from typing import Optional, Tuple, Union, List, Dict, Any

import torch
import torch.nn.functional as F


def log_probs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """计算目标标签序列在模型预测分布中的对数概率。
    
    该函数将模型的logits输出转换为概率分布，并提取目标标签对应的对数概率。
    主要用于语言模型训练、策略梯度计算等场景。
    
    Args:
        logits (torch.Tensor): 模型输出的原始logits值
            shape: (batch_size, sequence_length, vocab_size)
            example: torch.randn(32, 128, 50257)  # GPT模型输出
            
        labels (torch.Tensor): 目标标签的索引序列
            shape: (batch_size, sequence_length)
            example: torch.tensor([[1,2,3], [0,2,1]])
            values: 取值范围应在[0, vocab_size-1]之间
            
    Returns:
        torch.Tensor: 标签序列对应的对数概率
            shape: (batch_size, sequence_length)
            values: 通常在[-inf, 0]范围内，越接近0表示概率越大
            
    Examples:
        >>> # 创建示例数据
        >>> batch_size, seq_len, vocab_size = 2, 3, 5
        >>> logits = torch.randn(batch_size, seq_len, vocab_size)
        >>> labels = torch.tensor([[1,2,3], [0,2,1]])
        >>> log_probs = log_probs_from_logits(logits, labels)
        >>> print(log_probs.shape)  # torch.Size([2, 3])
        
    Notes:
        1. 使用log_softmax而不是softmax后取log，可以提高数值稳定性
        2. gather操作用于高效收集指定索引位置的值
        3. 输出的对数概率可直接用于计算交叉熵损失
    """
    # 计算对数概率分布
    # 在词表维度(dim=-1)上进行log_softmax，保持batch和序列长度维度不变
    # 输出shape: (batch_size, sequence_length, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # 扩展标签维度以匹配log_probs
    # 在最后增加一个维度，用于后续的gather操作
    # 输出shape: (batch_size, sequence_length, 1)
    labels_expanded = labels.unsqueeze(-1)
    
    # 收集标签对应的对数概率
    # 使用gather在词表维度上选择目标标签对应的概率值
    # 输出shape: (batch_size, sequence_length, 1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels_expanded)
    
    # 移除gather操作引入的额外维度
    # 输出shape: (batch_size, sequence_length)
    return log_probs_labels.squeeze(-1)

# Reset positions for packed samples
# For example
# Input: attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2, 3, 3, 0]])
# Output: position_ids  = torch.tensor([[0, 1, 2, 0, 1, 2, 0, 1, 0]])
def reset_position_ids(attention_mask):
    position_ids = torch.zeros_like(attention_mask, dtype=torch.long)
    for i in range(attention_mask.size(0)):
        mask = attention_mask[i]
        seq_num = mask.max().item()
        for index in range(1, seq_num + 1):
            sample_mask = mask == index
            sample_length = sample_mask.sum().item()
            position_ids[i, sample_mask] = torch.arange(sample_length, device=mask.device)
    return position_ids


def unpacking_samples(values: torch.Tensor, packed_seqlens: list[int]):
    values = values.squeeze(0)
    unpacked_values = []
    offset = 0
    for seqlen in packed_seqlens:
        unpacked_values.append(values[offset : offset + seqlen])
        offset += seqlen
    return unpacked_values


def masked_mean(tensor: torch.Tensor, mask: Optional[torch.Tensor], dim: int = None) -> torch.Tensor:
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)
