import os
import random
import shutil
import numpy as np
from abc import ABC
from typing import List, Tuple, Union
from datetime import timedelta
from collections import defaultdict

import deepspeed
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed as dist
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import _LRScheduler
from peft import PeftModel, get_peft_model_state_dict

from deepspeed.ops.adam import deepspeedCPUAdam, FusedAdam

from rlhf.utils.distributed_sampler import DistributedSampler
from rlhf.models import Actor
from rlhf.models.ring_attn_utils import get_ring_attn_group, set_ring_attn_group

from .deepspeed_utils import (
    get_eval_ds_config,
    get_optimizer_grouped_parameters,
    get_train_ds_config,
    _z3_params_to_fetch,
)

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


class DeepspeedStrategy(ABC):
    def __init__(
        self,
        seed: int = 42, 
        max_norm: float = 0.0,
        micro_train_batch_size: int = 1,
        train_batch_size: int = 1,
        zero_stage: int = 2,
        bf16: bool = True,
        args = None,
    ) -> None:
        super().__init__()
        
        self.args = args
        self.stage = zero_stage
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.bf16 = bf16
        self.seed = seed
        self.max_norm = max_norm
        self.adam_offload = getattr(args, "adam_offload", False)
        self.zpg = getattr(args, "zpg", 1)        
        self.grad_accum_dtype = getattr(args, "grad_accum_dtype", None)
        # disable_trace_cache
        self.disable_trace_cache = getattr(args, "disable_trace_cache", False)

        self.is_rlhf = False
        self.time_steps = defaultdict(int)
        
    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
def setup_distributed(self, timeout: timedelta = timedelta(minutes=30)) -> None:
    """设置分布式训练环境。
    
    此函数负责初始化分布式训练所需的各项配置,包括设置随机种子、GPU设备、分布式后端和梯度累积等。
    
    Args:
        timeout (timedelta, optional): 分布式初始化的超时时间。默认为30分钟。
        
    Notes:
        - 函数会根据环境变量LOCAL_RANK设置local_rank参数
        - 如果指定了local_rank,会将当前进程绑定到对应的GPU设备
        - 使用deepspeed初始化分布式后端
        - 设置环形注意力机制(ring attention)
        - 计算梯度累积步数
    """
    # 设置随机种子以确保可重现性
    self.set_seed(self.seed)
    
    # 从环境变量获取local_rank
    if self.args.local_rank == -1 and "LOCAL_RANK" in os.environ:
        self.args.local_rank = int(os.environ["LOCAL_RANK"])
    
    # 如果指定了local_rank,将进程绑定到对应GPU
    if self.args.local_rank != -1:
        torch.cuda.set_device(self.args.local_rank)
    
    # 初始化分布式训练后端
    deepspeed.init_distributed(timeout=timeout)
    
    # 设置环形注意力机制
    self.setup_ring_attn()
    
    # 获取总进程数
    self.world_size = dist.get_world_size()
    
    # 计算梯度累积步数
    # accumulated_gradient = (总批次大小 * 环形注意力大小) / (微批次大小 * 总进程数)
    self.accumulated_gradient = (
        self.train_batch_size * self.ring_attn_size // 
        self.micro_train_batch_size // self.world_size
    )
    
    def setup_ring_attn(self) -> None:
        """设置环形注意力机制(Ring Attention)。
        
        该函数负责配置分布式训练中的环形注意力机制,包括设置环形大小、创建进程组、
        并替换Hugging Face的Flash Attention实现。
        
        Notes:
            - ring_attn_size: 环形注意力组中的进程数
            - 如果ring_attn_size为1,则禁用环形注意力
            - 将所有进程分成多个大小为ring_attn_size的组
            - 每个组使用NCCL后端创建通信组
            - 替换默认的Flash Attention实现为环形版本
        """
        # 获取环形注意力大小,默认为1
        self.ring_attn_size = getattr(self.args, "ring_attn_size", 1)
        
        # 如果环形大小为1,则禁用环形注意力并返回
        if self.ring_attn_size == 1:
            self.ring_attn_size == 0
            return

        # 获取注意力头步长,默认为1
        ring_head_stride = getattr(self.args, "ring_head_stride", 1)
        
        # 将进程划分为多个环形组
        for i in range(dist.get_world_size() // self.ring_attn_size):
            # 计算当前环形组中的进程ranks
            ring_head_ranks = list(
                range(
                    i * self.ring_attn_size,
                    (i + 1) * self.ring_attn_size,
                )
            )
            
            # 使用NCCL后端创建新的进程组
            group = dist.new_group(ranks=ring_attn_ranks, backend="nccl")
            
            # 如果当前进程在该环形组中
            if dist.get_rank() in ring_head_ranks:
                # 设置环形注意力组
                set_ring_attn_group(group)
                # 获取进程在组内的rank
                self.ring_head_rank = dist.get_rank(group=group)
        
        # 导入并替换HuggingFace的Flash Attention实现
        from ring_flash_attn import substitute_hf_flash_attn
        substitute_hf_flash_attn(self.ring_attn_group, ring_head_stride)
    
    @property
    def ring_attn_group(self):
        """获取环形注意力通信组。
        
        这是一个属性装饰器方法,用于获取当前进程所属的环形注意力通信组。
        
        Returns:
            ProcessGroup: 返回由get_ring_attn_group()获取的分布式通信组对象。
            如果环形注意力未启用,可能返回None。
        
        Notes:
            - 使用@property装饰器,使得这个方法可以像属性一样被访问
            - 实际的通信组获取逻辑封装在get_ring_attn_group()函数中
            - 通信组用于环形注意力机制中进程间的通信
        """
        return get_ring_attn_group()
    
    def create_optimizer(self, model: Union[nn.Module, Actor], **kwargs) -> Optimizer:
        """创建优化器实例。
        
        根据配置创建适合的Adam优化器,支持CPU卸载和参数融合优化。
        
        Args:
            model (Union[nn.Module, Actor]): 需要优化的模型。
                如果是Actor实例,会提取其内部的model属性。
            **kwargs: 优化器的额外参数。
                必须包含weight_decay参数。
                其他可能的参数包括learning_rate、beta1、beta2等。
        
        Returns:
            Optimizer: 返回配置好的优化器实例,可能是deepspeedCPUAdam或FusedAdam。
        
        Notes:
            - 当adam_offload=True时使用deepspeedCPUAdam进行CPU卸载
            - 当adam_offload=False时使用FusedAdam进行融合优化
            - 使用get_optimizer_grouped_parameters来对模型参数进行分组
        """
        # 如果输入是Actor实例,提取其内部model
        if isinstance(model, Actor):
            model = model.model
            
        # 根据是否进行CPU卸载选择优化器类型
        AdamOptimizer = deepspeedCPUAdam if self.adam_offload else FusedAdam
        
        # 获取分组后的模型参数
        optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        
        # 创建并返回优化器实例
        optim = AdamOptimizer(optim_params, **kwargs)
        return optim

    def backward(self, loss: torch.Tensor, model: Union[nn.Module, Actor], 
                optimizer: Optimizer, **kwargs) -> None:
        """执行反向传播计算梯度。
        
        对给定的损失进行反向传播,计算模型参数的梯度。
        
        Args:
            loss (torch.Tensor): 需要反向传播的损失值。
            model (Union[nn.Module, Actor]): 需要计算梯度的模型。
                如果是Actor实例,会提取其内部的model属性。
            optimizer (optim.Optimizer): 用于更新模型参数的优化器实例。
                当前版本未使用此参数,但保留以保持接口一致性。
            **kwargs: 额外的参数。
                当前版本未使用,但保留以支持可能的扩展。
        
        Notes:
            - 函数会检查模型类型,确保使用正确的模型实例
            - 实际的反向传播通过model.backward()实现
            - optimizer参数当前未使用,但作为标准接口的一部分保留
        """
        # 如果输入是Actor实例,提取其内部model
        if isinstance(model, Actor):
            model = model.model
        
        # 执行反向传播
        model.backward(loss)
        
    def optimizer_step(
            self, 
            optimizer: Optimizer,
            model: Union[nn.Module, Actor],
            scheduler: Optional[_LRScheduler],
            name: str = "model",
            **kwargs,
        ) -> None:
        """执行优化器的参数更新步骤。
        
        根据之前计算的梯度更新模型参数。
        
        Args:
            optimizer (optim.Optimizer): 用于更新模型参数的优化器实例。
                当前版本未使用此参数,但保留以保持接口一致性。
            model (Union[nn.Module, Actor]): 需要更新参数的模型。
                如果是Actor实例,会提取其内部的model属性。
            scheduler (_LRScheduler, optional): 学习率调度器。
                当前版本未使用此参数,但保留以保持接口一致性。
            name (str, optional): 模型的标识名称。
                默认为"model"。当前版本未使用此参数。
            **kwargs: 额外的参数。
                当前版本未使用,但保留以支持可能的扩展。
        
        Notes:
            - 函数会检查模型类型,确保使用正确的模型实例
            - 实际的参数更新通过model.step()实现
            - optimizer和scheduler参数当前未使用,但作为标准接口的一部分保留
        """
        # 如果输入是Actor实例,提取其内部model
        if isinstance(model, Actor):
            model = model.model
        
        # 执行参数更新步骤
        model.step()
        
    def setup_dataloader(
            self,
            replay_buffer: Dataset,
            batch_size: int,
            pin_memory: bool = False,
            shuffle: bool = True,
            collate_fn: Optional[Callable] = None,
            drop_last: bool = True,
            sampler: Optional[Sampler] = None,
            consumed_samples: int = 0,
        ) -> DataLoader:
        """配置并创建分布式数据加载器。
        
        注意: 这是纯DDP模式,每个rank上的replay buffer数据是不同的。
        
        Args:
            replay_buffer (Dataset): 数据集对象,当前rank上的经验回放缓冲区。
                在DDP模式下,每个rank拥有自己独立的数据。
            batch_size (int): 每个批次的样本数量。
                这是每个rank上的局部批次大小。
            pin_memory (bool, optional): 是否将数据固定在内存中。
                默认为False。在使用GPU时设为True可能提高性能。
            shuffle (bool, optional): 是否打乱数据顺序。
                默认为True。仅在未指定sampler时有效。
            collate_fn (Callable, optional): 数据整理函数。
                默认为None,使用默认的整理方法。
            drop_last (bool, optional): 是否丢弃最后一个不完整的批次。
                默认为True。
            sampler (Sampler, optional): 自定义的采样器实例。
                在DDP模式下,如未指定会创建特殊的采样器。
            consumed_samples (int, optional): 已消耗的样本数量。
                默认为0,用于恢复训练时的数据加载位置。
        
        Returns:
            DataLoader: 配置好的PyTorch数据加载器实例。
        
        Notes:
            - 这是纯DDP模式实现,每个rank独立维护自己的数据
            - 不需要使用DistributedSampler来分割数据,因为数据已经是分开的
            - sampler的主要作用是控制数据顺序和断点续训
            - 全局批次大小 = batch_size * 进程数
        """
        # 如果没有提供采样器,创建分布式采样器
        if sampler is None:
            # 计算实际的进程数和rank,考虑环形注意力大小
            num_replicas = dist.get_world_size() // self.ring_attn_size
            rank = dist.get_rank() // self.ring_attn_size
            
            # 创建分布式采样器
            sampler = DistributedSampler(
                replay_buffer,
                num_replicas=num_replicas,  # 在DDP模式下实际上每个rank是独立的
                rank=rank,
                shuffle=shuffle,
                seed=self.seed,
                drop_last=drop_last,
                consumed_samples=consumed_samples,
            )
        
        # 创建并返回数据加载器
        return DataLoader(
            replay_buffer,
            batch_size=batch_size,  # 这是每个rank上的局部批次大小
            sampler=sampler,
            drop_last=drop_last,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
    
    def _unwrap_model(self, model: Union[nn.Module, Actor, DistributedDataParallel]) -> nn.Module:
        """解包装模型,获取原始的模型实例。
        
        递归地解除模型的各层封装,返回最内层的实际模型实例。
        
        Args:
            model: 需要解包装的模型,可能的类型包括:
                - nn.Module: 原始PyTorch模型
                - Actor: 自定义的Actor封装器
                - DistributedDataParallel: PyTorch的分布式封装器
                或者其他具有.module属性的封装器类型
        
        Returns:
            nn.Module: 解除所有封装后的原始模型实例
        
        Notes:
            - 支持递归解包装,可以处理多层封装的情况
            - 主要处理三种情况:
            1. Actor封装
            2. DDP或类似的带.module属性的封装
            3. 原始模型
        """
        # 处理Actor封装的情况
        if isinstance(model, Actor):
            # 递归解包装Actor内部的模型
            return self._unwrap_model(model.model)
        # 处理DDP等带.module属性的封装
        elif hasattr(model, "module"):
            # 返回.module属性指向的实际模型
            return model.module
        # 处理原始模型的情况
        else:
            # 直接返回原始模型
            return model
            
    def prepare(
        self,
        *model_or_model_optim_pairs: ModelOrModelOptimPair,
        is_rlhf: bool = False,
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        """
        准备模型、优化器和调度器用于训练。
        
        此函数可以处理单个模型或多个模型-优化器对的初始化。它支持两种输入形式：
        1. 单个模型对象
        2. 包含(模型,优化器,调度器)的三元组

        Args:
            *model_or_model_optim_pairs (ModelOrModelOptimPair): 
                可变参数,接受以下两种形式:
                - 单个模型对象
                - (model, optimizer, scheduler)三元组
            is_rlhf (bool, optional): 
                是否使用人类反馈强化学习模式。默认为False。

        Returns:
            Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
                - 当输入单个模型/三元组时,返回处理后的单个ModelOrModelOptimPair
                - 当输入多个模型/三元组时,返回处理后的ModelOrModelOptimPair列表
                
        Raises:
            AssertionError: 当输入的三元组长度不为3时抛出异常
            
        Examples:
            >>> # 单个模型
            >>> model = prepare(single_model)
            >>> # 多个模型-优化器对
            >>> models = prepare((model1, optim1, sched1), (model2, optim2, sched2))
        """
        ret = []  # 初始化返回列表
        self.is_rlhf = is_rlhf  # 设置RLHF标志
        
        # 遍历所有输入的模型或模型-优化器对
        for arg in model_or_model_optim_pairs:
            if isinstance(arg, tuple):  # 如果是元组形式(模型,优化器,调度器)
                assert len(arg) == 3, f"Expect (model, optimizer, `scheduler`)"
                if arg[0] is not None:  # 如果模型不为空
                    ret.append(self._ds_init_train_model(*arg))
                else:  # 如果模型为空,添加空三元组
                    ret.append((None, None, None))
            else:  # 如果是单个模型对象
                ret.append(self._ds_init_train_model(arg))
        
        # 如果只有一个结果返回单个结果,否则返回列表
        return ret[0] if len(ret) == 1 else ret

    def _ds_init_train_model(
        self,
        model: Union[Actor, nn.Module],
        optim: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None
    ) -> Tuple[Union[Actor, nn.Module], Optional[Optimizer], Optional[_LRScheduler]]:
        """
        初始化DeepSpeed训练模型配置。

        该函数使用DeepSpeed框架初始化模型、优化器和调度器，支持Actor模型和普通模型。
        DeepSpeed用于实现分布式训练和模型并行化。

        Args:
            model: 要初始化的模型,可以是Actor类型或普通PyTorch模型
            optim: 优化器实例,用于模型参数的优化,默认为None
            scheduler: 学习率调度器,用于动态调整学习率,默认为None

        Returns:
            包含以下元素的元组:
            - model: 初始化后的模型
            - optim: 初始化后的优化器
            - scheduler: 初始化后的学习率调度器

        Note:
            - Actor类型模型会特殊处理其内部model属性
            - DeepSpeed配置根据模型类型动态获取
            - 依赖DeepSpeed框架进行分布式训练初始化
        """
        # 确定模型类型并获取配置
        is_actor = isinstance(model, Actor)
        ds_config = self.get_ds_train_config(is_actor)
        
        # DeepSpeed初始化
        engine, optim, _, scheduler = deepspeed.initialize(
            model=model.model if is_actor else model,
            args={"local_rank": self.args.local_rank},
            config=ds_config,
            dist_init_required=True
        )

        # 更新模型引用
        if is_actor:
            model.model = engine
        else:
            model = engine

        return model, optim, scheduler
            
    def get_ds_train_config(self, is_actor: bool) -> Dict[str, Any]:
        """
        获取DeepSpeed训练配置参数。

        根据不同的训练场景(RLHF、Actor模型等)生成对应的DeepSpeed配置字典,
        包括训练批次大小、模型并行化、混合精度训练等参数设置。

        Args:
            is_actor: 是否为Actor模型类型

        Returns:
            包含DeepSpeed训练配置的字典,主要配置项包括:
            - train_micro_batch_size_pre_gpu: 每个GPU的微批次大小
            - train_batch_size: 总训练批次大小
            - 其他配置项: offload、混合精度、梯度裁剪等参数

        Note:
            - 在RLHF预训练场景下,Actor模型的batch_size会翻倍
            - 最终的train_batch_size会乘以ring_attn_size进行扩展
            - 配置参数通过get_train_ds_config获取基础配置后再进行调整
        """
        # 获取基础DeepSpeed配置
        ds_config = get_train_ds_config(
            offload=False,  # 是否开启显存优化
            adam_offload=self.adam_offload,  # 是否将优化器状态卸载到CPU
            stage=self.stage,  # DeepSpeed优化阶段
            bf16=self.bf16,  # 是否使用BF16混合精度
            max_norm=self.max_norm,  # 梯度裁剪最大范数
            zpg=self.zpg,  # 零参数梯度
            grad_accum_dtype=self.grad_accum_dtype,  # 梯度累积数据类型
            disable_trace_cache=self.disable_trace_cache,  # 是否禁用跟踪缓存
        )
        
        # 设置每个GPU的微批次大小
        ds_config["train_micro_batch_size_pre_gpu"] = self.micro_train_batch_size
        
        # 计算总训练批次大小
        train_batch_size = self.train_batch_size
        
        # RLHF预训练场景下Actor模型的特殊处理
        if self.is_rlhf and is_actor and self.args.pretrain_data is not None:
            train_batch_size *= 2  # 批次大小翻倍
        
        # 考虑ring attention大小,设置最终训练批次大小
        ds_config["train_batch_size"] = train_batch_size * self.ring_attn_size
        
        return ds_config
    
    def _ds_init_eval_model(self, model: Optional[Union[Actor, nn.Module]]) -> Optional[Union[Actor, nn.Module]]:
        """
        初始化用于评估的DeepSpeed模型配置。

        为评估阶段配置DeepSpeed模型,支持Actor类型和普通PyTorch模型。
        相比训练配置,评估配置更加轻量,主要聚焦于推理性能优化。

        Args:
            model: 要初始化的模型,可以是以下类型:
                - Actor: Actor类型模型
                - nn.Module: 普通PyTorch模型
                - None: 空值,直接返回
        
        Returns:
            初始化后的模型,类型与输入相同,如果输入为None则返回None

        Note:
            - 支持模型的offload特性,通过model._offload属性控制
            - Actor类型模型会特殊处理其内部model属性
            - 使用DeepSpeed的评估专用配置
        """
        # 如果模型为None,直接返回
        if not model:
            return model

        # 判断模型类型
        is_actor = isinstance(model, Actor)
        
        # 获取评估配置,考虑模型的offload属性
        ds_config = self.get_ds_eval_config(
            offload=getattr(model, "_offload", False)
        )

        # DeepSpeed初始化
        engine, *_ = deepspeed.initialize(
            model=model.model if is_actor else model,  # 根据模型类型选择正确的模型对象
            args={"local_rank": self.args.local_rank},  # 设置本地rank
            config=ds_config,  # 使用评估配置
            dist_init_required=True  # 要求进行分布式初始化
        )

        # 更新模型引用
        if is_actor:
            model.model = engine  # Actor类型更新内部model
        else:
            model = engine  # 普通模型直接更新引用

        return model
    
    def moving_average(
        self,
        model: nn.Module,
        model_ema: nn.Module,
        beta: float = 0.992,
        device: str = "cpu"
    ) -> None:
        """
        对模型参数执行指数移动平均(EMA)更新。

        使用指数移动平均来更新目标模型(model_ema)的参数,通过源模型(model)的参数
        和历史参数的加权平均实现。支持正常模式和DeepSpeed ZeRO-3优化模式。

        Args:
            model: 源模型,提供最新的参数值
            model_ema: 目标模型,其参数将被EMA更新
            beta: EMA的衰减率,决定历史参数的权重,默认0.992
            device: 计算设备,默认为"cpu"

        Note:
            - 每accumulated_gradient步才进行一次更新
            - ZeRO-3模式下使用特殊的参数gather机制
            - 只更新requires_grad=True的参数
            - EMA更新公式: param_new = (1-beta) * param_current + beta * param_old

        Examples:
            >>> ema_updater = MovingAverage()
            >>> ema_updater.moving_average(model, model_ema, beta=0.999)
        """
        # 更新EMA步数计数器
        self.time_steps["ema"] += 1

        # 检查是否达到累积梯度的步数
        if self.time_steps["ema"] % self.accumulated_gradient == 0:
            with torch.no_grad():
                # 遍历源模型和目标模型的对应参数
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    if param.requires_grad:
                        if self.stage != 3:
                            # 普通模式: 直接执行EMA更新
                            data = param.data.to(device)
                            param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)
                        else:
                            # ZeRO-3模式: 使用特殊的参数gather机制
                            params_to_fetch = _z3_params_to_fetch([param, param_ema])
                            # 确保参数完全gathered后再更新
                            with deepspeed.zero.GatheredParameters(
                                params_to_fetch, 
                                enabled=len(params_to_fetch) > 0
                            ):
                                data = param.data.to(device)
                                param_ema.data.copy_((1 - beta) * data + beta * param_ema.data)
    
def load_model(
    self,
    model: nn.Module,
    path: str,
    map_location: Union[str, torch.device] = "cpu",
    strict: bool = False,
    key_replace_fn: Optional[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]] = None
) -> None:
    """
    加载模型参数状态。

    从指定路径加载模型状态字典并应用到模型。支持模型参数键的自定义替换,
    可以处理分布式封装的模型。

    Args:
        model: 待加载参数的PyTorch模型
        path: 模型状态字典的存储路径
        map_location: 参数加载的目标设备,可以是字符串或torch.device对象,默认"cpu"
        strict: 是否严格匹配键值,为True时键不匹配会报错,默认False
        key_replace_fn: 可选的状态字典键替换函数,用于修改加载的键名,默认None

    Note:
        - 会自动处理DistributedDataParallel等封装的模型
        - 支持通过key_replace_fn修改状态字典的键名
        - 非严格模式下允许键不完全匹配
        - 使用torch.load加载状态字典

    Examples:
        >>> model = MyModel()
        >>> loader = ModelLoader()
        >>> # 基本加载
        >>> loader.load_model(model, "model.pth")
        >>> # 使用键替换函数
        >>> loader.load_model(model, "model.pth", 
        ...                  key_replace_fn=lambda d: {k.replace('module.',''): v 
        ...                                           for k,v in d.items()})
    """
    # 获取原始模型(移除分布式封装等)
    unwrapped_model = self._unwrap_model(model)

    # 加载状态字典
    state_dict = torch.load(path, map_location=map_location)

    # 如果提供了键替换函数,执行键替换
    if key_replace_fn:
        state_dict = key_replace_fn(state_dict)

    # 将状态字典加载到模型
    unwrapped_model.load_state_dict(state_dict, strict=strict)
    
    def save_model(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        output_dir: str,
        **kwargs
    ) -> None:
        """
        保存模型、分词器和配置文件。

        支持分布式训练场景下的模型保存,处理PEFT模型和普通模型,
        同时保存相关的配置文件和训练代码。

        Args:
            model: 要保存的PyTorch模型
            tokenizer: 与模型配套的分词器
            output_dir: 保存目录路径
            **kwargs: 传递给save_pretrained的额外参数

        Note:
            - 只在rank 0进程执行保存操作
            - 支持DeepSpeed ZeRO-3参数收集
            - 特殊处理PEFT模型的保存
            - 自动复制训练相关的Python文件

        保存内容:
            1. 模型状态字典
            2. 模型配置文件
            3. 分词器
            4. 训练相关的Python文件
        """
        # 确保输出目录存在(仅rank 0)
        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)
        
        # 获取原始模型
        model_to_save = self._unwrap_model(model)
        
        # 收集模型参数
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            # ZeRO-3参数收集
            params_to_fetch = _z3_params_to_fetch([v])
            with deepspeed.zero.GatheredParameters(params_to_fetch, enabled=len(params_to_fetch) > 0):
                vv = v.data.cpu()
                if self.is_rank_0():
                    output_state_dict[k] = vv
        
        # rank 0执行保存操作
        if self.is_rank_0():
            # 获取完整状态字典
            state_dict = model_to_save.state_dict()
            
            # 补充未收集的参数
            for k, v in model_to_save.named_parameters():
                if k not in state_dict:
                    continue
                vv = v.data.cpu()
                output_state_dict[k] = vv
            
            # 验证参数完整性
            state_dict_keys = set(state_dict.keys())
            output_state_dict_keys = set(output_state_dict.keys())
            
            # 处理词嵌入权重共享
            if getattr(model_to_save.config, "tie_word_embeddings", False) and "lm_head.weight" in state_dict_keys:
                state_dict_keys.remove("lm_head.weight")
            
            # 确保所有必需的键都存在
            assert state_dict_keys.issubset(
                output_state_dict_keys
            ), f"mismatch keys {output_state_dict_keys.symmetric_difference(state_dict_keys)}"
            
            # 保存模型
            if isinstance(model_to_save, PeftModel):
                # PEFT模型的特殊保存
                model_to_save.save_pretrained(output_dir, **kwargs)
                if self.stage == 3:
                    torch.save(
                        get_peft_model_state_dict(model_to_save, output_state_dict),
                        os.path.join(output_dir, "adapter_model.bin")
                    )
            else:
                # 普通模型保存
                model_to_save.save_pretrained(output_dir, state_dict=output_state_dict, **kwargs)
            
            # 保存配置
            output_config_file = os.path.join(output_dir, "config.json")
            model_to_save.config.to_json_file(output_config_file)
            
            # 保存分词器
            tokenizer.save_pretrained(output_dir)
            
            # 复制训练相关的Python文件
            train_from_model_path = model_to_save.config._name_or_path
            if os.path.exists(train_from_model_path):
                for filename in os.listdir(train_from_model_path):
                    if filename.endswith(".py"):
                        shutil.copy(
                            os.path.join(train_from_model_path, filename),
                            os.path.join(output_dir, filename)
                        )
    
    def all_reduce(
        self,
        data: Union[Dict, torch.Tensor, float, int],
        op: str = "mean"
    ) -> Union[Dict, torch.Tensor, float]:
        """
        执行分布式全规约(all-reduce)操作。

        支持对字典、张量和标量数据进行分布式规约计算,
        可以执行平均值、最大值、求和三种规约操作。

        Args:
            data: 要进行规约的数据,支持以下类型:
                - Dict: 字典类型,会递归处理其值
                - torch.Tensor: PyTorch张量
                - float/int: 标量值
            op: 规约操作类型,必须是以下之一:
                - "mean": 求平均值
                - "max": 求最大值
                - "sum": 求和
                默认为"mean"

        Returns:
            规约后的结果,类型与输入数据相同:
                - Dict: 规约后的字典
                - torch.Tensor: 规约后的张量
                - float: 规约后的标量值

        Raises:
            AssertionError: 当op参数不是允许的值时

        Note:
            - 自动处理CPU和GPU张量
            - 对非张量数据会先转换为张量再处理
            - 字典数据会递归处理每个值
        """
        # 检查操作类型是否合法
        assert op in ("mean", "max", "sum")

        # 处理字典类型数据
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_reduce(v, op)
            return ret
        else:
            # 处理非字典类型数据
            is_tensor = True
            # 非张量数据转换为张量
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
                is_tensor = False
            
            # 检查是否为CPU张量
            is_cpu_tensor = data.device.type == "cpu"

            # CPU张量转移到当前GPU
            if is_cpu_tensor:
                data = data.to(torch.cuda.current_device())

            # 平均值操作预处理
            if op == "mean":
                data /= self.world_size

            # 执行all-reduce操作
            dist.all_reduce(
                data,
                op=dict.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM
            )

            # CPU张量移回CPU
            if is_cpu_tensor:
                data = data.cpu()

            # 返回适当的数据类型
            return data.item() if not is_tensor else data
    
    def all_gather(
        self,
        data: Union[Dict, torch.Tensor, float, int]
    ) -> Union[Dict, torch.Tensor]:
        """
        执行分布式全收集(all-gather)操作。

        从所有进程收集数据并在每个进程上返回完整的数据集合。
        支持字典、张量和标量类型的数据收集。

        Args:
            data: 要收集的数据,支持以下类型:
                - Dict: 字典类型,会递归处理其值
                - torch.Tensor: PyTorch张量
                - float/int: 标量值

        Returns:
            收集后的数据:
                - Dict: 收集后的字典,包含所有进程的数据
                - torch.Tensor: 连接后的张量,包含所有进程的数据

        Note:
            - 自动处理CPU和GPU张量
            - 非张量数据会被转换为张量处理
            - 字典数据会递归处理每个值
            - 返回的张量按进程rank顺序连接
            - 保持输入数据的设备位置(CPU/GPU)

        Examples:
            >>> # 单值收集
            >>> local_value = torch.tensor([1.0])
            >>> gathered = all_gather(local_value)  # size = world_size
            >>> 
            >>> # 字典收集
            >>> local_dict = {'a': 1.0, 'b': torch.tensor([2.0])}
            >>> gathered_dict = all_gather(local_dict)
        """
        # 处理字典类型数据
        if isinstance(data, dict):
            ret = {}
            for k, v in data.items():
                ret[k] = self.all_gather(v)
            return ret
        else:
            # 处理非字典类型数据
            # 将标量转换为张量
            if not isinstance(data, torch.Tensor):
                data = torch.Tensor([data])
            
            # 记录原始设备类型
            is_cpu_tensor = data.device.type == "cpu"

            # 为每个进程创建接收缓冲区
            ret = [
                torch.zeros_like(data).to(torch.cuda.current_device()) 
                for _ in range(self.world_size)
            ]

            # 执行all-gather操作
            dist.all_gather(
                ret,  # 接收列表
                data.to(torch.cuda.current_device())  # 发送数据
            )

            # 连接结果并处理设备位置
            return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)
            
    def print(
        self,
        *msg: Any
    ) -> None:
        """
        在分布式环境中进行rank 0进程的打印。

        只在主进程(rank 0)打印消息,避免在多进程环境下重复打印。
        可接受任意数量的参数,与Python内置print函数使用方式相同。

        Args:
            *msg: 要打印的消息,支持任意类型和数量的参数

        Note:
            - 只在rank 0进程执行打印
            - 完全兼容Python内置print的参数传递方式
            - 用于分布式训练中的日志打印

        Examples:
            >>> # 单个消息
            >>> trainer.print("Training started")
            >>> 
            >>> # 多个参数
            >>> epoch = 1
            >>> loss = 0.123
            >>> trainer.print("Epoch:", epoch, "Loss:", loss)
            >>>
            >>> # 格式化字符串
            >>> trainer.print(f"Epoch {epoch} completed with loss {loss:.4f}")
        """
        # 检查是否为rank 0进程
        if self.is_rank_0():
            # 使用Python内置print函数打印消息
            print(*msg)

    def is_rank_0(self) -> bool:
        """
        检查当前进程是否为主进程(rank 0)。

        在分布式环境中判断当前进程是否为主进程，
        用于控制只应在主进程执行的操作。

        Returns:
            bool: 如果当前进程是rank 0则返回True，否则返回False

        Note:
            - 使用PyTorch分布式接口获取进程rank
            - 常用于控制打印、保存模型等操作
            - 在非分布式环境下总是返回True

        Examples:
            >>> if trainer.is_rank_0():
            ...     print("This only prints on main process")
            ...     save_model()
        """
        return dist.get_rank() == 0

    def get_rank(self) -> int:
        """
        获取当前进程的rank值。

        返回在分布式环境中当前进程的rank编号，
        rank是从0开始的整数，标识每个进程。

        Returns:
            int: 当前进程的rank值
                - 0: 主进程
                - >0: 其他工作进程
                - 非分布式环境下返回0

        Note:
            - 使用PyTorch分布式接口获取进程rank
            - rank值在进程组内唯一
            - rank 0通常作为主进程使用

        Examples:
            >>> rank = trainer.get_rank()
            >>> print(f"Current process rank: {rank}")
        """
        return dist.get_rank()
    
    def save_ckpt(
        self,
        model: deepspeed.DeepSpeedEngine,
        save_dir: str,
        tag: Optional[str] = None,
        max_num: int = 3,
        max_mem: int = 1000,
        client_state: Dict = {},
        save_latest: bool = True
    ) -> None:
        """
        保存模型检查点，并管理检查点存储空间。

        在保存新检查点时会自动管理存储空间，当检查点数量或总大小超过限制时，
        会删除最旧的检查点。支持DeepSpeed分布式训练框架。

        Args:
            model: DeepSpeed模型对象
            save_dir: 检查点保存的目录路径
            tag: 检查点标签，用于标识特定检查点，默认为None
            max_num: 最大保存的检查点数量，默认为3
            max_mem: 检查点存储最大内存限制(GB)，默认为1000GB
            client_state: 需要随检查点一起保存的客户端状态字典，默认为空字典
            save_latest: 是否保存最新检查点的标记，默认为True

        Raises:
            AssertionError: 当模型不是DeepSpeed引擎实例时抛出

        Note:
            - 只在rank 0进程执行清理操作
            - 自动管理存储空间
            - 支持分布式训练
            - 保证检查点完整性

        Space Management:
            1. 检查点数量控制: 保持不超过max_num个检查点
            2. 存储空间控制: 总大小不超过max_mem GB
            3. 删除策略: 优先删除最早的检查点
        """
        # 确保模型是DeepSpeed引擎实例
        assert isinstance(model, deepspeed.DeepSpeedEngine)

        # rank 0进程执行清理操作
        if self.is_rank_0():
            # 创建保存目录
            os.makedirs(save_dir, exist_ok=True)
            # 转换GB到字节
            MAX_SIZE = max_mem * 1024**3  

            # 循环检查和清理空间
            while True:
                # 获取所有子目录并按修改时间排序
                subdirs = sorted(
                    [
                        (os.path.join(save_dir, d), os.path.getmtime(os.path.join(save_dir, d)))
                        for d in os.listdir(save_dir)
                        if os.path.isdir(os.path.join(save_dir, d))
                    ],
                    key=lambda x: x[1],
                )

                # 计算所有检查点的总大小
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, f))
                    for subdir, _ in subdirs
                    for dirpath, _, filenames in os.walk(subdir)
                    for f in filenames
                )

                # 检查是否需要清理
                if len(subdirs) >= max_num or total_size > MAX_SIZE:
                    # 删除最旧的检查点
                    oldest_dir = subdirs[0][0]
                    if os.path.exists(oldest_dir):
                        shutil.rmtree(oldest_dir)
                        self.print(f"Deleted oldest ckpt {oldest_dir}")
                else:
                    break

        # 等待所有进程完成清理
        dist.barrier()
        # 保存新的检查点
        model.save_checkpoint(save_dir, tag=tag, client_state=client_state, save_latest=save_latest)
        
    def load_ckpt(
        self,
        model: deepspeed.DeepSpeedEngine,
        load_dir: str,
        tag: Optional[str] = None,
        load_module_strict: bool = True,
        laod_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
        load_module_only: bool = False,
    ) -> Tuple[str, Dict]:
        """
        加载模型检查点及相关状态。

        从指定目录加载DeepSpeed模型检查点，支持加载模型参数、优化器状态和学习率调度器状态。
        提供灵活的加载选项，可以选择性加载不同组件的状态。

        Args:
            model: DeepSpeed模型对象
            load_dir: 检查点加载目录路径
            tag: 检查点标签，用于加载特定检查点，默认为None
            load_module_strict: 是否严格加载模型参数，默认为True
            laod_optimizer_states: 是否加载优化器状态，默认为True
            load_lr_scheduler_states: 是否加载学习率调度器状态，默认为True
            load_module_only: 是否只加载模型参数，默认为False

        Returns:
            Tuple[str, Dict]: 返回加载的检查点路径和状态字典
                - str: 成功加载的检查点完整路径
                - Dict: 包含加载的状态信息的字典

        Raises:
            AssertionError: 当模型不是DeepSpeed引擎实例时抛出
            Exception: 当检查点加载失败时抛出

        Note:
            - 支持DeepSpeed分布式训练框架
            - 可选择性加载不同组件状态
            - 提供严格/非严格加载模式
            - 自动处理设备映射

        Examples:
            >>> # 完整加载检查点
            >>> path, states = trainer.load_ckpt(model, "checkpoints/")
            >>> 
            >>> # 只加载模型参数
            >>> path, states = trainer.load_ckpt(
            ...     model, 
            ...     "checkpoints/",
            ...     load_module_only=True
            ... )
        """
        # 确保模型是DeepSpeed引擎实例
        assert isinstance(model, deepspeed.DeepSpeedEngine)

        # 尝试加载检查点
        load_path, states = model.load_checkpoint(
            load_dir,
            tag,
            load_module_strict=load_module_strict,
            laod_optimizer_states=laod_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
            load_module_only=load_module_only,
        )

        # 检查加载是否成功
        if load_path is None:
            raise Exception(f"[deepspeed] failed to resume from checkpoint {load_dir}")

        return load_path, states