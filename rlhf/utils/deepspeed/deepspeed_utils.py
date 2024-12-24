# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def get_train_ds_config(
    offload: bool,
    adam_offload: bool = True,
    stage: int = 2,
    bf16: bool = True, 
    max_norm: float = 1.0,
    zpg: int = 8,
    grad_accum_dtype: Optional[str] = None,
    disable_trace_cache: bool = False,
) -> Dict[str, Any]:
    """
    生成DeepSpeed训练配置字典。

    根据输入参数配置DeepSpeed训练参数，包括ZeRO优化、混合精度训练、
    梯度裁剪等设置。支持CPU卸载和多种优化选项。

    Args:
        offload: 是否启用模型参数CPU卸载
        adam_offload: 是否将优化器状态卸载到CPU，默认True
        stage: ZeRO优化阶段(1-3)，默认2
        bf16: 是否启用bfloat16混合精度训练，默认True
        max_norm: 梯度裁剪阈值，默认1.0
        zpg: ZeRO参数分组大小，默认8
        grad_accum_dtype: 梯度累积数据类型，默认None
        disable_trace_cache: 是否禁用Trace缓存，默认False

    Returns:
        Dict[str, Any]: DeepSpeed配置字典，包含以下主要部分：
            - zero_optimization: ZeRO优化配置
            - bf16: 混合精度训练配置
            - gradient_clipping: 梯度裁剪配置
            - data_types: 数据类型配置
            - 其他训练相关配置

    Note:
        - ZeRO优化支持三个阶段
        - 支持CPU卸载以节省GPU内存
        - 提供自动和手动内存管理选项
        - 支持梯度累积和混合精度训练

    Configuration Details:
        1. ZeRO优化:
           - 参数分片
           - 优化器状态分片
           - 自动内存管理
        
        2. 内存优化:
           - CPU卸载
           - 梯度量化
           - 缓存控制

        3. 训练优化:
           - 混合精度
           - 梯度裁剪
           - 梯度累积
    """
    # 确定参数卸载设备
    device = "cpu" if offload else "none"

    # 配置ZeRO优化字典
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {"device": device},
        "offload_optimizer": {
            "device": "cpu" if adam_offload else "none",
            "pin_memory": True,
        },
        # 自动配置内存管理参数
        "sub_group_size": "auto",
        "stage3_max_live_parameters": "auto",
        "stage3_max_reuse_distance": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "reduce_bucket_size": "auto",
        # ZeRO参数分组
        "zero_hpz_partition_size": zpg,
        # 关闭权重和梯度量化
        "zero_quantized_weights": False,
        "zero_quantized_gradients": False,
    }

    # 处理trace缓存禁用
    if disable_trace_cache:
        zero_opt_dict["stage3_prefetch_bucket_size"] = 0
        zero_opt_dict["stage3_max_live_parameters"] = 0
        zero_opt_dict["stage3_max_reuse_distance"] = 0
    
    # 返回完整配置字典
    return {
        "steps_per_print": 100,  # 打印频率
        "zero_optimization": zero_opt_dict,  # ZeRO优化配置
        "bf16": {
            "enabled": bf16,  # 混合精度训练设置
        },
        "gradient_clipping": max_norm,  # 梯度裁剪阈值
        "prescale_gradients": False,  # 关闭梯度预缩放
        "wall_clock_breakdown": False,  # 关闭时间分析
        "data_types": {
            "grad_accum_dtype": grad_accum_dtype,  # 梯度累积类型
        },
    }
    
def get_eval_ds_config(
    offload: bool,
    stage: int = 0,
    bf16: bool = True,
) -> Dict[str, Any]:
    """
    生成DeepSpeed评估/推理配置字典。

    配置用于模型评估和推理阶段的DeepSpeed参数，支持CPU卸载和混合精度。
    相比训练配置更加轻量，主要focus在推理效率和内存优化。

    Args:
        offload: 是否启用模型参数CPU卸载
        stage: ZeRO优化阶段，默认0（不启用）
        bf16: 是否启用bfloat16混合精度，默认True

    Returns:
        Dict[str, Any]: DeepSpeed评估配置字典，包含以下主要部分：
            - zero_optimization: ZeRO优化配置
            - bf16: 混合精度配置
            - gradient_clipping: 梯度裁剪配置（评估时通常不需要）
            - 其他评估相关配置

    Note:
        - 评估配置通常比训练配置简单
        - 主要关注推理效率和内存使用
        - 支持CPU卸载以处理大模型
        - 可选择性启用混合精度

    Configuration Details:
        1. 内存优化:
           - 参数CPU卸载
           - 持久化阈值控制
           - 内存钉扎(pin_memory)

        2. 性能优化:
           - 混合精度推理
           - ZeRO优化（可选）
           
        3. 监控选项:
           - 步骤打印
           - 时钟分析
    """
    # 配置ZeRO优化字典
    zero_opt_dict = {
        "stage": stage,  # ZeRO优化阶段
        "stage3_param_persistence_threshold": "auto",  # 参数持久化阈值
        "offload_param": {
            "device": "cpu" if offload else "none",  # 卸载设备选择
            "pin_memory": True,  # 启用内存钉扎
        },
    }

    # 返回完整配置字典
    return {
        "steps_per_prit": 100,  # 打印频率
        "zero_optimization": zero_opt_dict,  # ZeRO优化配置
        "bf16": {
            "enabled": bf16,  # 混合精度配置
        },
        "gradient_clipping": 1.0,  # 梯度裁剪（评估时通常不使用）
        "prescale_gradients": False,  # 关闭梯度预缩放
        "wall_clock_breakdown": False,  # 关闭时间分析
    }
    
def get_optimizer_grouped_parameters(
    model: torch.nn.Module,
    weight_decay: float,
    no_decay_name_list: List[str] = [
        "bias",
        "layer_norm.weight",
        "layernorm.weight", 
        "norm.weight",
        "ln_f.weight"
    ]
) -> List[Dict[str, Any]]:
    """
    将模型参数分组用于优化器，对不同参数使用不同的权重衰减策略。

    将模型参数分为两组:
    1. 需要权重衰减的参数（如普通权重矩阵）
    2. 不需要权重衰减的参数（如偏置项、层归一化参数等）

    Args:
        model: PyTorch模型对象
        weight_decay: 权重衰减系数
        no_decay_name_list: 不需要权重衰减的参数名列表，默认包含:
            - bias: 偏置项
            - layer_norm.weight: 层归一化权重
            - layernorm.weight: 层归一化权重（别名）
            - norm.weight: 归一化权重
            - ln_f.weight: 最终层归一化权重

    Returns:
        List[Dict[str, Any]]: 参数分组列表，每组包含:
            - params: 参数列表
            - weight_decay: 权重衰减值

    Note:
        - 只处理requires_grad=True的参数
        - 通过参数名匹配确定分组
        - 支持自定义无衰减参数列表
        - 适用于Adam类优化器

    Examples:
        >>> model = MyModel()
        >>> param_groups = get_optimizer_grouped_parameters(
        ...     model, 
        ...     weight_decay=0.01
        ... )
        >>> optimizer = torch.optim.AdamW(param_groups)
    """
    # 构建参数分组
    optimizer_grouped_parameters = [
        {
            # 需要权重衰减的参数组
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            # 不需要权重衰减的参数组
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]

    return optimizer_grouped_parameters

def _z3_params_to_fetch(param_list: List[torch.nn.Parameter]) -> List[torch.nn.Parameter]:
    """
    在ZeRO-3优化中筛选需要获取的参数。

    在分布式训练中，使用ZeRO-3优化时，筛选出当前设备上不可用且需要从其他设备获取的模型参数。
    通过检查参数的ds_id（DeepSpeed ID）和ds_status（参数状态）来确定。

    Args:
        param_list: 模型参数列表

    Returns:
        List[torch.nn.Parameter]: 需要获取的参数列表，包含所有满足以下条件的参数:
            - 具有ds_id属性（表明是ZeRO管理的参数）
            - ds_status为NOT_AVAILABLE（表明参数当前不在本地设备）

    Note:
        - 仅用于ZeRO-3优化
        - 检查参数的DeepSpeed属性
        - 用于参数预取优化
        - 支持分布式训练场景

    Implementation Details:
        1. 参数筛选条件：
           - 存在ds_id属性
           - 状态为NOT_AVAILABLE

    Examples:
        >>> # 在ZeRO-3训练中使用
        >>> params_to_fetch = _z3_params_to_fetch(model.parameters())
        >>> # 进行参数预取操作
        >>> fetch_parameters(params_to_fetch)
    """
    return [p for p in param_list 
            if hasattr(p, "ds_id") and p.ds_status == ZeroParamStatus.NOT_AVAILABLE]