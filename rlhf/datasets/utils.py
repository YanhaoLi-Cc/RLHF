import torch
import torch.nn.functional as F


def zero_pad_sequences(sequences, side: str = "left", value=0):
    """对序列进行零填充以使所有序列具有相同长度
    
    Args:
        sequences: 张量序列的列表,每个张量可以有不同的长度
        side: 填充位置,"left"表示在序列左侧填充,"right"表示在右侧填充
        value: 用于填充的值,默认为0
        
    Returns:
        torch.Tensor: 填充后的序列stack成的张量,shape为[batch_size, max_seq_len, ...]
    """
    assert side in ("left", "right")
    # 获取所有序列中的最大长度
    max_len = max(seq.size(-1) for seq in sequences)
    padded_sequences = []
    for seq in sequences:
        # 计算需要填充的长度
        pad_len = max_len - seq.size(-1)
        # 根据side参数确定填充位置
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        # 对序列进行填充并添加到列表中
        padded_sequences.append(F.pad(seq, padding, value=value))
    # 将填充后的序列stack成一个批量张量
    return torch.stack(padded_sequences, dim=0)


def exist_and_not_none(d, key):
    """检查字典中是否存在指定的键且其值不为None
    
    Args:
        d: 要检查的字典
        key: 要检查的键
        
    Returns:
        bool: 如果键存在且其值不为None则返回True,否则返回False
    """
    return key in d and not d[key] is None
