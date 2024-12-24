from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean

class GPTLMLoss(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    
    def __init__(self, ring_attn_group=None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index = self.IGNORE_INDEX)
        self.ring_attn_group = ring_attn_group
        
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.ring_attn_group is not None:
            pass
        else:
            shift_logits = logits[..., :-1].contiguous
            shift_labels = labels[..., 1:].contiguous
            
            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
        return loss
    
    