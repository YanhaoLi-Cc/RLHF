import os
from abc import ABC
from tqdm import tqdm

import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.optim import Optimizer
from ..models import GPTLMLoss

class SFTTrainer(ABC):
    """监督微调(Supervised Fine-Tuning)训练器的基类。

    该类实现了大语言模型监督微调的核心训练逻辑，支持分布式训练、
    混合精度训练、梯度裁剪等高级特性。

    Args:
        model (torch.nn.Module): 待训练的模型。
        strategy (Strategy): 训练策略对象,控制分布式训练、混合精度等特性。
        optim (Optimizer): 用于模型训练的优化器。
        train_dataloader (DataLoader): 训练数据集的数据加载器。
        eval_dataloader (DataLoader): 评估数据集的数据加载器。
        scheduler (Scheduler): 学习率调度器,用于动态调整训练速率。
        max_norm (float, defaults to 1.0): 梯度裁剪的最大范数,用于防止梯度爆炸。
        batch_size (int, defaults to 1): 训练的批次大小。
        max_epochs (int, defaults to 2): 最大训练轮数。
        tokenizer (Tokenizer, optional): 用于处理输入数据的分词器。
        pretrain_mode (bool, defaults to False): 是否使用预训练模式。

    Attributes:
        model (torch.nn.Module): 训练的模型实例
        strategy (Strategy): 训练策略实例
        optimizer (Optimizer): 优化器实例
        scheduler (Scheduler): 学习率调度器实例
        train_dataloader (DataLoader): 训练数据加载器
        eval_dataloader (DataLoader): 验证数据加载器
        tokenizer (Tokenizer): 分词器实例
        epochs (int): 训练轮数
        batch_size (int): 批次大小
        max_norm (float): 梯度裁剪阈值
        args (Namespace): 训练参数配置
        pretrain_mode (bool): 是否为预训练模式
        loss_fn (LossFunction): 损失函数实例
        aux_loss (bool): 是否使用辅助损失
        packing_samples (bool): 是否启用样本打包
        _wandb: Weights & Biases日志记录器
        _tensorboard: TensorBoard日志记录器
    """

    def __init__(
        self,
        model: torch.nn.Module,
        strategy: Strategy,
        optim: Optimizer,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader, 
        scheduler: Scheduler,
        max_norm: float = 1.0,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer: Optional[Tokenizer] = None,
        pretrain_mode: bool = False,
    ) -> None:
        """初始化SFT训练器。

        Args:
            model (torch.nn.Module): 待训练的模型。
            strategy (Strategy): 训练策略对象。
            optim (Optimizer): 优化器实例。
            train_dataloader (DataLoader): 训练数据加载器。
            eval_dataloader (DataLoader): 验证数据加载器。
            scheduler (Scheduler): 学习率调度器。
            max_norm (float, defaults to 1.0): 梯度裁剪阈值。
            batch_size (int, defaults to 1): 训练批次大小。
            max_epochs (int, defaults to 2): 最大训练轮数。
            tokenizer (Tokenizer, optional): 分词器实例。
            pretrain_mode (bool, defaults to False): 是否使用预训练模式。
        """
        super().__init__()
        
        # 基础训练组件
        self.model = model
        self.strategy = strategy
        self.optimizer = optim
        self.scheduler = scheduler
        
        # 数据相关配置
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.tokenizer = tokenizer
        
        # 训练超参数
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.args = strategy.args
        self.pretrain_mode = pretrain_mode
        
        # 损失函数配置
        self.loss_fn = GPTLMLoss(
            ring_attn_group=self.strategy.ring_attn_group
        )
        self.aux_loss = self.aux_loss_coef > 1e-8
        
        # 训练优化配置
        self.packing_samples = strategy.args.packing_samples
        
        # 日志工具初始化
        self._wandb = None
        self._tensorboard = None
        
        # 配置日志工具
        if self.strategy.args.use_wandb and self.strategy.is_rank_0():
            # print("wandb not implement")
            import wandb

            self._wandb = wandb
            if not wandb.api.api_key:
                wandb.login(key=strategy.args.use_wandb)
            wandb.init(
                entity=strategy.args.wandb_org,
                project=strategy.args.wandb_project,
                group=strategy.args.wandb_group,
                name=strategy.args.wandb_run_name,
                config=strategy.args.__dict__,
                reinit=True,
            )

            wandb.define_metric("train/global_step")
            wandb.define_metric("train/*", step_metric="train/global_step", step_sync=True)
            wandb.define_metric("eval/global_step")
            wandb.define_metric("eval/*", step_metric="eval/global_step", step_sync=True)
        
        if self.strategy.args.use_tensorboard:
            print("tensorboard not implement")

    def fit(self, args: Namespace, consumed_samples: int = 0, num_update_steps_per_epoch: Optional[int] = None) -> None:
        """执行模型训练流程。

        Args:
            args (Namespace): 训练配置参数。
            consumed_samples (int, defaults to 0): 已处理的样本数。
            num_update_steps_per_epoch (int, optional): 每轮的更新步数。

        """
        # 设置评估和保存步骤
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch # 每轮评估一次
        if args.save_steps == -1:
            args.save_steps = float("inf") # 不保存检查点
            
        # 计算初始步数和轮数
        step = consumed_samples // args.train_batch_size * step.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        # 配置训练进度条
        epoch_bar = tqdm(
            range(start_epoch, self.epochs),
            desc = "Train epoch",
            disable = not self.strategy.is_rank_0(),
        )

        # 开始训练循环                
        for epoch in range(start_epoch, self.epochs):
            # 设置分布式采样器的epoch
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch,
                    consumed_samples=0 if epoch > start_epoch else consumed_samples
                )
            
            # 配置步骤进度条
            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc = f"Train step of epoch {epoch}",
                disable = not self.strategy.is_rank_0(),
            )
            
            # 训练模式
            self.model.train()
            loss_mean = 0

            # 训练步骤
            for prompt_id_lens, inputs, attention_masks, infos in self.train_dataloader:
                # 数据预处理
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                
                # 前向传播
                if self.strategy.ring_attn_group is None:
                    output = self.model(inputs, attention_mask=attention_mask, return_output=True)
                else:
                    output = self.model(
                        inputs,
                        attention_mask=attention_mask,
                        return_output=True,
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=infos["input_length"],
                    )
                    
                # 准备标签
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )
                
                # 计算辅助损失
                if self.aux_loss:
                    aux_loss = output.aux_loss
                else:
                    aux_loss = 0
                
                # 处理非预训练模式
                if not self.pretrain_mode:
                    if self.packing_samples:
                        index = 0
                        for input_length, source_len in zip(infos["input_length"], prompt_id_lens):
                            labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                            index += input_length
                    else:
                        for label, source_len in zip(labels, prompt_id_lens):
                            label[:source_len] = self.loss_fn.IGNORE_INDEX
                
                # 计算损失并更新
                gpt_loss = self.loss_fn(output.logits, labels)
                loss = gpt_loss + aux_loss * self.aux_loss_coef
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)
                
                # 更新损失统计
                loss_mean = loss_mean * 0.9 + gpt_loss.item() * 0.1
                logs_dict = {
                    "gpt_loss": gpt_loss.item(),
                    "loss_mean": loss_mean,
                    "lr": self.scheduler.get_last_lr()[0],
                }
                if self.aux_loss:
                    logs_dict["aux_loss"] = aux_loss.item()

                # 更新进度条
                logs_dict = self.strategy.all_reduce(logs_dict)
                step_bar.set_postfix(logs_dict)
                step_bar.update()
                
                # 保存日志和检查点
                if step % self.strategy.accumulated_gradient == 0:
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, step_bar, logs_dict, client_states)
                
                step += 1
                
            epoch_bar.update()
            
        # 关闭日志记录器
        if self._wandb is not None and self.strategy.is_rank_0():
            self._wandb.finish()
        if self._tensorboard is not None and self.strategy.is_rank_0():
            self._tensorboard.finish()
            
    def save_logs_and_checkpoints(
        self,
        args,
        global_step: int,
        step_bar: tqdm,
        logs_dict: Dict[str, float] = {},
        client_states: Dict[str, Any] = {}
    ) -> None:
        """保存训练日志和模型检查点。

        根据配置的步数间隔,执行以下操作:
        1. 记录训练日志到wandb或tensorboard
        2. 执行模型评估
        3. 保存模型检查点

        Args:
            args (Namespace): 训练配置参数,包含日志记录、评估和保存的相关配置。
            global_step (int): 当前的全局训练步数。
            step_bar (tqdm): 训练进度条对象。
            logs_dict (Dict[str, float], optional): 待记录的训练指标字典。默认为空字典。
            client_states (Dict[str, Any], optional): 客户端状态信息。默认为空字典。
        """
        # 记录训练日志
        if global_step % args.logging_steps == 0:
            if self._wandb is not None and self.strategy.is_rank_0():
                # 转换日志格式并记录到wandb
                logs = {
                    "train/%s" % k: v 
                    for k, v in {**logs_dict, "global_step": global_step}.items()
                }
                self._wandb.log(logs)
            elif self._tensorboard is not None and self.strategy.is_rank_0():
                # 记录到tensorboard
                for k, v in logs_dict.items():
                    self._tensorboard.add_scalar(f"train/{k}", v, global_step)
        
        # 执行模型评估        
        if global_step % args.eval_steps == 0:
            if len(self.eval_dataloader) > 0:
                self.evaluate(self.eval_dataloader, global_step)
        
        # 保存模型检查点
        if global_step % args.save_ckpt == 0:
            tag = f"global_step{global_step}"
            self.strategy.save_ckpt(
                self.model.model,
                args.ckpt_path,
                tag,
                args.max_ckpt_num,
                args.max_ckpt_mem,
                client_states
            )
    
    def evaluate(self, eval_dataloader, steps=0):
        times = 0
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc="Eval stage of steps %d" % steps,
                disable=not self.strategy.is_rank_0(),
            )

            for prompt_id_lens, inputs, attention_masks, infos in eval_dataloader:
                if self.packing_samples:
                    inputs = inputs.to(torch.cuda.current_device())
                    attention_mask = attention_masks.to(torch.cuda.current_device())
                else:
                    inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                    attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)

                if self.strategy.ring_attn_group is None:
                    output = self.model(inputs, attention_mask=attention_mask, return_output=True)
                else:
                    output = self.model(
                        inputs, 
                        attention_mask=attention_mask, 
                        return_output=True,
                        ring_attn_group=self.strategy.ring_attn_group,
                        packed_seq_lens=infos["input_length"],
                    )
                    
                # loss function
                labels = torch.where(
                    attention_mask.bool(),
                    inputs,
                    self.loss_fn.IGNORE_INDEX,
                )

                if not self.pretrain_mode:
                    if self.packing_samples:
                        index = 0
                        for input_length, source_len in zip(infos["input_length"], prompt_id_lens):
                            labels[0][index : index + source_len] = self.loss_fn.IGNORE_INDEX
                            index += input_length
                    else:
                        for label, source_len in zip(labels, prompt_id_lens):
                            label[:source_len] = self.loss_fn.IGNORE_INDEX

                loss = self.loss_fn(output.logits, labels)

                times += 1
                loss_sum += loss.item()
                bar_dict = {"eval gpt_loss": loss_sum / times}
                step_bar.update()
                logs = self.strategy.all_reduce(bar_dict)
                step_bar.set_postfix(logs)

            if self.strategy.is_rank_0():
                if self._wandb is not None:
                    logs = {"eval/%s" % k: v for k, v in {**logs, "global_step": steps}.items()}
                    self._wandb.log(logs)
                elif self._tensorboard is not None:
                    for k, v in logs.items():
                        self._tensorboard.add_scalar(f"eval/{k}", v, steps)
        self.model.train()  # reset model state
        