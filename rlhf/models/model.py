from typing import Optional, Union

import deepspeed
import torch
import torch.nn as nn
from flash_attn.utils.distributed import all_gather
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import LoraLayer
from transformers import AutoConfig, AutoModel, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from rlhf.utils.logging_utils import init_logger

from .ring_attn_utils import convert_ring_attn_params
from .utils import reset_position_ids

logger = init_logger(__name__)


# Construct transformer with a value head for sequence classification.
# https://github.com/huggingface/transformers/blob/405b56269812056d9593869e22b7b264d806cb1e/src/transformers/models/llama/modeling_llama.py#L1254
def get_llm_for_sequence_regression(
    model_name_or_path: str,  # 预训练模型的路径
    model_type: str,          # 模型类型,optional"reward"或"critic" 
    *,
    bf16=True,               # 是否启用 bfloat16 精度
    load_in_4bit=False,      # 是否加载为4位精度 
    lora_rank=0,             # LoRA 适配的秩
    lora_alpha=16,           # LoRA 的 alpha 参数
    target_modules=None,     # LoRA 目标模块列表
    lora_dropout=0,          # LoRA 层的 dropout 率
    normalize_reward=False,   # 是否归一化奖励值
    use_flash_attention_2=False,  # 是否使用 Flash Attention 2.0
    ds_config: dict = None,  # 启用 ZeRO-3 时用于跨多 GPU 分区模型的 Deepspeed 配置
    init_value_head: bool = False,  # 是否初始化值头
    value_head_prefix="score",      # 值头的前缀
    device_map=None,         # 模型加载的设备映射
    packing_samples=False,   # 训练时是否打包样本
    **kwargs,
) -> nn.Module:
    """获取带有序列回归头的 transformer 模型。
    
    该函数加载预训练的 transformer 模型并附加一个用于序列回归的线性层。

    参数:
        model_name_or_path (str): 预训练模型的路径
        model_type (str): 模型类型,optional"reward"或"critic"
        bf16 (bool, optional): 启用 bfloat16 精度。默认为 True
        load_in_4bit (bool, optional): 以 4 位精度加载模型。默认为 False
        lora_rank (int, optional): LoRA 适配的秩。默认为 0
        lora_alpha (int, optional): LoRA 的 alpha 参数。默认为 16
        target_modules (list, optional): LoRA 的目标模块列表。默认为 None
        lora_dropout (float, optional): LoRA 层的 dropout 率。默认为 0
        normalize_reward (bool, optional): 是否归一化奖励值。默认为 False
        use_flash_attention_2 (bool, optional): 是否使用 Flash Attention 2.0。默认为 False
        ds_config (dict, optional): 启用 ZeRO-3 时用于跨多 GPU 分区模型的 Deepspeed 配置。默认为 None
        init_value_head (bool, optional): 是否初始化值头。默认为 False
        value_head_prefix (str, optional): 值头的前缀。默认为"score"
        device_map (dict, optional): 模型加载的设备映射。默认为 None
        packing_samples (bool, optional): 训练时是否打包样本。默认为 False

    返回:
        nn.Module: 带有序列回归头的预训练 transformer 模型
    """
    # 验证模型类型是否合法
    assert (
        model_type == "critic" or model_type == "reward"
    ), f"invalid model_type: {model_type}, should be critic or reward."

    # 加载模型配置
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    config.normalize_reward = normalize_reward
    config._attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"

    # 优先使用模型配置中的 value_head_prefix
    value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)
    logger.info(f"set value_head_prefix to `{value_head_prefix}`")

    # 获取基础模型类
    base_class = AutoModel._model_mapping[type(config)]
    base_pretrained_class = base_class.__base__
    if model_type == "reward":
        cls_class = _get_reward_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)
    else:
        cls_class = _get_critic_model(base_pretrained_class, base_class, value_head_prefix, packing_samples)

    # Note: dschf is defined in function scope to avoid global effects
    # https://huggingface.co/docs/transformers/main_classes/deepspeed#nontrainer-deepspeed-integration
    # 配置 DeepSpeed
    if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
        dschf = HfDeepSpeedConfig(ds_config)
    else:
        dschf = None

    # 配置 4 位量化
    if load_in_4bit:
        assert bf16, "we only support bnb_4bit_compute_dtype = bf16"
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        nf4_config = None

    # 加载预训练模型
    model = cls_class.from_pretrained(
        model_name_or_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        quantization_config=nf4_config,
        device_map=device_map,
        **kwargs,
    )

    # 配置 LoRA
    if lora_rank > 0:
        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        # 设置 4 位量化时的数据类型
        if load_in_4bit:
            for name, module in model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if "norm" in name:
                    module = module.to(torch.float32)
                if value_head_prefix in name or "embed_tokens" in name:
                    if hasattr(module, "weight"):
                        module = module.to(torch.bfloat16)

    # 配置 MoE - 平衡损失
    model_config = model.config.to_dict()
    if "output_router_logits" in model_config:
        print("[MoE] set output_router_logits as True")
        model.config.output_router_logits = True

    # 禁用缓存
    model.config.use_cache = False

    # 注意: 仅用于奖励模型训练,需要手动初始化 value_head
    # 因为 deepspeed.zero.Init() 不会初始化它们
    if init_value_head:
        value_head = getattr(model, value_head_prefix)
        if dschf is not None:
            logger.info("initialize value_head for ZeRO-3 reward model training.")
            with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
        else:
            value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    return model


def _get_reward_model(base_pretrained_model, base_llm_model, value_head_prefix="score", packing_samples=False):
    """获取奖励模型类
    
    该函数创建一个继承自预训练模型的奖励模型类,用于给序列打分。

    参数:
        base_pretrained_model: 基础预训练模型类
        base_llm_model: 基础语言模型类
        value_head_prefix (str): 值头层的前缀名,默认为"score" 
        packing_samples (bool): 是否打包样本,默认为 False

    返回:
        RewardModel: 继承自基础预训练模型的奖励模型类
    """
    class RewardModel(base_pretrained_model):
        """奖励模型类
        
        继承自基础预训练模型,添加了value_head层用于打分。
        """
        
        # 支持梯度检查点
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            """初始化奖励模型
            
            参数:
                config (AutoConfig): 模型配置
            """
            # 调用父类初始化
            super().__init__(config)
            
            # 设置基础语言模型
            setattr(self, self.base_model_prefix, base_llm_model(config))

            # 设置value_head层
            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            # 是否打包样本
            self.packing_samples = packing_samples

            # 设置奖励值归一化相关参数
            self.normalize_reward = config.normalize_reward
            # 注册均值和标准差缓冲区,不持久保存
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # 从配置文件加载均值和标准差
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,            # 输入的token id序列
            attention_mask: Optional[torch.Tensor] = None, # 注意力掩码
            return_output=False,                           # 是否返回模型中间输出
            ring_attn_group=None,                          # 环形注意力分组 
            packed_seq_lens=None,                          # 打包序列的长度信息
        ) -> torch.Tensor:
            """奖励模型的前向传播函数
            
            Args:
                input_ids (torch.LongTensor): 输入序列的token id
                attention_mask (torch.Tensor, optional): 注意力掩码矩阵
                return_output (bool): 是否返回模型的中间输出结果
                ring_attn_group: 环形注意力机制的分组信息
                packed_seq_lens: 打包后序列的长度信息
                
            Returns:
                torch.Tensor: 如果return_output为False,只返回奖励值;否则返回(奖励值, 模型输出)的元组
            """
            if not self.packing_samples:
                # 非打包样本模式:根据attention_mask计算position_ids
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                # 打包样本模式:转换attention_mask为position_ids
                if ring_attn_group is not None:
                    # 使用环形注意力时转换参数
                    input_ids, attention_mask, position_ids = convert_ring_attn_params(
                        input_ids, attention_mask, packed_seq_lens, ring_attn_group
                    )
                else:
                    # 重置position_ids
                    position_ids = reset_position_ids(attention_mask)
                # 打包样本模式下显式忽略attention_mask
                attention_mask = None

            # 通过基础模型获取隐藏状态
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            # 通过值头网络计算值
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)

            if self.packing_samples:
                # 打包样本模式下的奖励计算
                if ring_attn_group is not None:
                    reward = all_gather(values, ring_attn_group).reshape(1, -1)
                else:
                    reward = values
                packed_seq_lens = torch.tensor(packed_seq_lens, device=values.device)
                # 计算每个序列的结束索引
                eos_indices = packed_seq_lens.cumsum(dim=0) - 1
                reward = reward.squeeze(0).gather(dim=0, index=eos_indices)
            else:
                # 非打包样本模式下的奖励计算
                # 找到每个序列的最后一个非pad token位置
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)

            # 测试阶段且需要归一化时,对奖励进行归一化
            if not self.training and self.normalize_reward:
                reward = (reward - self.mean) / self.std

            return (reward, outputs) if return_output else reward

    return RewardModel


def _get_critic_model(base_pretrained_model, base_llm_model, value_head_prefix="score", packing_samples=False):
    """获取评论家(Critic)模型类
    
    Args:
        base_pretrained_model: 基础预训练模型类
        base_llm_model: 基础语言模型类
        value_head_prefix (str): 值头层的前缀名,默认为"score"
        packing_samples (bool): 是否打包样本,默认为False
    
    Returns:
        CriticModel: 继承自基础预训练模型的评论家模型类
    """
    class CriticModel(base_pretrained_model):
        """评论家模型类
        
        继承自基础预训练模型,用于评估动作值。支持梯度检查点功能。
        """
        supports_gradient_checkpointing = True

        def __init__(self, config: AutoConfig):
            """初始化评论家模型
            
            Args:
                config (AutoConfig): 模型配置对象
            """
            super().__init__(config)
            # 设置基础语言模型
            setattr(self, self.base_model_prefix, base_llm_model(config))
            
            # 设置value head层
            self.value_head_prefix = value_head_prefix
            setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

            # 是否打包样本
            self.packing_samples = packing_samples

            # 设置奖励归一化相关参数
            self.normalize_reward = config.normalize_reward
            self.register_buffer("mean", torch.zeros(1), persistent=False)
            self.register_buffer("std", torch.ones(1), persistent=False)

            # 从配置文件加载均值和标准差
            if hasattr(config, "mean"):
                self.mean[0] = config.mean
                self.std[0] = config.std

        def forward(
            self,
            input_ids: torch.LongTensor = None,                   # 输入序列的token ids
            num_actions: Optional[Union[int, list[int]]] = None,  # 动作数量
            attention_mask: Optional[torch.Tensor] = None,        # 注意力掩码
            return_output=False,                                  # 是否返回模型中间输出
            packed_seq_lens=None,                                 # 打包序列的长度信息
        ) -> torch.Tensor:
            """评论家模型的前向传播函数
            
            Args:
                input_ids: 输入序列的token ids
                num_actions: 需要评估的动作数量,可以是整数或整数列表
                attention_mask: 注意力掩码矩阵
                return_output: 是否返回模型的中间输出
                packed_seq_lens: 打包序列的长度信息
            
            Returns:
                torch.Tensor: 动作值或模型输出
            """
            if not self.packing_samples:
                # 非打包样本模式:计算position_ids
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                # 打包样本模式:重置position_ids
                position_ids = reset_position_ids(attention_mask)
                attention_mask = None

            # 获取模型输出和最后的隐藏状态
            outputs = getattr(self, self.base_model_prefix)(
                input_ids, attention_mask=attention_mask, position_ids=position_ids
            )
            last_hidden_states = outputs["last_hidden_state"]
            # 计算除最后一个token外的所有token的值
            values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)[:, :-1]

            # 如果需要,对值进行归一化
            if self.normalize_reward:
                values = (values - self.mean) / self.std

            # 如果没有指定动作数量,返回模型输出
            if num_actions is None:
                assert return_output
                return outputs

            if not self.packing_samples:
                # 非打包模式:获取最后num_actions个token的值
                action_values = values[:, -num_actions:]
            else:
                # 打包模式:根据每个序列的动作数量和长度提取相应的值
                assert isinstance(num_actions, list) and len(num_actions) == len(packed_seq_lens)
                action_values = []
                offset = 0
                for num_action, seq_len in zip(num_actions, packed_seq_lens):
                    start, end = max(0, offset + seq_len - num_action - 1), offset + seq_len - 1
                    action_values.append(values[:, start:end])
                    offset += seq_len
                action_values = torch.cat(action_values, dim=1)

            if return_output:
                return (action_values, outputs)
            else:
                return action_values

    return CriticModel
