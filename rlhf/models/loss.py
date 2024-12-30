from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean

class GPTLMLoss(nn.Module):
    """GPT Language Model Loss计算模块
    
    计算GPT语言模型的损失函数,支持分布式训练的RingAttention机制

    Args:
        ring_attn_group: 分布式训练的进程组,用于RingAttention。默认为None表示不使用RingAttention
    """
    
    def __init__(self, ring_attn_group=None):
        """初始化GPTLMLoss
        
        Args:
            ring_attn_group: 分布式训练的进程组,用于RingAttention
        """
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index = self.IGNORE_INDEX)
        self.ring_attn_group = ring_attn_group
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """前向计算
        
        Args:
            logits: 模型预测的logits,shape为[batch_size, seq_len, vocab_size] 
            labels: 目标标签,shape为[batch_size, seq_len]

        Returns:
            torch.Tensor: 计算得到的loss标量值
            
        Note:
            使用RingAttention时,每个进程只处理部分序列长度的数据
            如果某个batch的标签全为IGNORE_INDEX,返回0 loss以维持梯度流动
        """
        # RingAttention分布式训练
        if self.ring_attn_group is not None:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start_idx = self.ring_attn_rank * seq_len_per_process
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)
            labels = labels[..., start_idx:end_idx]

            shift_logits = logits[..., :-1, :].contiguous()  # 预测t时刻的词
            shift_labels = labels[..., 1:].contiguous()  # 使用t+1时刻的词作为标签

            # 如果标签全为IGNORE_INDEX,CrossEntropyLoss会返回nan
            if torch.all(shift_labels == self.IGNORE_INDEX):
                # 返回logits均值乘0以保持梯度流动
                loss = shift_logits.mean() * 0
            else:
                loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # 跨进程规约求和取平均
            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.ring_attn_group)
            loss = loss / self.ring_attn_world_size
        # 不使用RingAttention的普通训练
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss
    
    