import torch
import torch.distributed as dist
import torch.nn.functional as F

# 初始化全局变量
RING_ATTN_GROUP: Optional[dist.ProcessGroup] = None
"""
环形注意力机制的进程组全局变量。
用于跨模块共享分布式训练中的进程组信息。
初始值为None，需要在使用前通过set_ring_attn_group设置。
"""

def set_ring_attn_group(group: dist.ProcessGroup) -> None:
    """
    设置环形注意力机制使用的分布式进程组。

    为环形注意力机制设置全局进程组，使得不同模块可以共享相同的分布式通信组。
    这个进程组将用于协调分布式训练中的通信和计算。

    Args:
        group: PyTorch分布式进程组对象，用于环形注意力的分布式计算

    Note:
        - 全局单例模式
        - 线程不安全
        - 需要在使用前设置
        - 用于分布式训练

    Example:
        >>> # 创建进程组
        >>> world_size = 4
        >>> ranks = list(range(world_size))
        >>> group = dist.new_group(ranks)
        >>> 
        >>> # 设置环形注意力进程组
        >>> set_ring_attn_group(group)
        >>> 
        >>> # 验证设置
        >>> assert get_ring_attn_group() is not None

    Warning:
        - 确保在分布式环境初始化后调用
        - 避免重复设置
        - 注意进程组兼容性
    """
    global RING_ATTN_GROUP
    RING_ATTN_GROUP = group

def get_ring_attn_group() -> Optional[dist.ProcessGroup]:
    """
    获取当前设置的环形注意力进程组。

    返回当前环形注意力机制使用的全局分布式进程组。
    如果进程组未设置，返回None。

    Returns:
        Optional[dist.ProcessGroup]: 当前设置的分布式进程组，未设置时为None

    Note:
        - 返回全局单例
        - 只读访问
        - 线程不安全
        - 用于查询配置

    Example:
        >>> # 获取进程组
        >>> group = get_ring_attn_group()
        >>> 
        >>> # 使用进程组
        >>> if group is not None:
        ...     rank = dist.get_rank(group)
        ...     world_size = dist.get_world_size(group)
        ...     print(f"Rank {rank} in group of size {world_size}")

    Warning:
        - 使用前检查返回值
        - 注意并发访问
        - 确保初始化状态
    """
    return RING_ATTN_GROUP

def reset_ring_attn_position_ids(
    start: int, 
    end: int, 
    packed_seq_lens: List[int]
) -> torch.Tensor:
    """
    计算打包序列中指定范围的位置编码。

    对于打包的多个序列，根据每个序列的长度生成相应的位置编码。
    位置编码在每个序列内部从0开始递增，不同序列之间独立计数。
    
    Args:
        start: 起始位置索引
        end: 结束位置索引
        packed_seq_lens: 打包序列的长度列表

    Returns:
        torch.Tensor: 生成的位置编码张量，形状为[1, end-start]
            - 每个位置的值表示该位置在其所属序列中的相对位置
            - 设备与当前CUDA设备相同
            - 数据类型为torch.long

    Example:
        >>> lengths = [3, 2, 4, 1]  # 四个序列的长度
        >>> pos_ids = reset_ring_attn_position_ids(2, 8, lengths)
        >>> print(pos_ids)
        # tensor([[2, 0, 1, 0, 1, 2]])
        # 解释:
        # - 第一个序列(长度3): 位置2
        # - 第二个序列(长度2): 位置0,1
        # - 第三个序列(长度4): 位置0,1,2
        # 总共覆盖索引2到8的范围

    Note:
        - 用于Transformer模型中的位置编码
        - 支持序列打包训练
        - 处理长文本切分
        - 维护序列边界信息

    Implementation Details:
        1. 创建输出张量:
           - 形状为[1, end-start]
           - 初始化为0
           - 放置在当前CUDA设备

        2. 遍历序列:
           - 计算每个序列在目标范围内的部分
           - 为该部分生成位置编码
           - 处理序列重叠和边界情况

        3. 优化考虑:
           - 提前终止条件
           - 边界检查
           - 内存效率
    """
    # 创建用于存储位置编码的张量
    position_ids = torch.zeros(
        (1, end - start),  # 形状[1, 长度]
        dtype=torch.long,  # 数据类型
        device=torch.cuda.current_device()  # 设备位置
    )

    # 记录当前处理位置的偏移量
    offset = 0

    # 遍历每个序列的长度
    for seqlen in packed_seq_lens:
        # 计算当前序列在目标范围内的起止位置
        seq_start = max(offset, start)  # 序列起始位置
        seq_end = min(offset + seqlen, end)  # 序列结束位置

        # 如果当前序列部分落在目标范围内
        if seq_start < seq_end:
            # 生成该序列的位置编码
            position_ids[0, seq_start - start : seq_end - start] = torch.arange(
                seq_start - offset,  # 序列内相对起始位置
                seq_end - offset     # 序列内相对结束位置
            )

        # 更新偏移量
        offset += seqlen
        
        # 如果已处理完目标范围，提前终止
        if offset >= end:
            break

    return position_ids

def update_ring_attn_params(
    packed_seq_lens: List[int], 
    total_seq_len: int
) -> None:
    """
    更新环形注意力机制的参数，计算累积序列长度并更新到ring_flash_attn。

    为当前前向传播计算累积序列长度(cumulative sequence lengths)，并将其传递给
    替代的ring_flash_attn实现。这个函数主要用于优化packed sequence的注意力计算。

    Args:
        packed_seq_lens: 打包序列的长度列表，每个元素表示一个序列的实际长度
        total_seq_len: 总序列长度，可能大于packed_seq_lens的和（因为padding）

    Raises:
        AssertionError: 当RING_ATTN_GROUP未初始化时抛出
        RuntimeError: 当累积序列长度计算失败时抛出

    Note:
        - 用于环形注意力机制的参数更新
        - 处理打包序列的长度信息
        - 支持序列padding
        - 与ring_flash_attn配合使用

    Technical Details:
        1. 累积序列长度计算:
           - 使用torch.cumsum计算
           - 转换为int32类型
           - 在首尾添加padding
        
        2. 数据处理流程:
           - 验证输入
           - 计算累积长度
           - 添加padding
           - 更新参数

        3. 内存优化:
           - 使用GPU计算
           - 控制数据类型
           - 高效padding操作

    Example:
        >>> # 基本使用
        >>> seq_lens = [3, 4, 2]  # 三个序列的长度
        >>> total_len = 10  # 总长度（包含padding）
        >>> update_ring_attn_params(seq_lens, total_len)
        
        >>> # 实际效果
        >>> # 输入序列长度: [3, 4, 2]
        >>> # 累积长度: [0, 3, 7, 9, 10]  
        >>> #           └─ [起始=0, 第一个序列后=3, 第二个序列后=7, ...]
    """
    # 验证RING_ATTN_GROUP是否已初始化
    assert RING_ATTN_GROUP is not None, "RING_ATTN_GROUP must be initialized before calling this function"

    # 计算累积序列长度
    cu_seqlens = torch.cumsum(
        torch.tensor(
            packed_seq_lens,  # 输入序列长度列表
            device=torch.cuda.current_device(),  # 使用当前CUDA设备
            dtype=torch.int32  # 使用int32类型
        ),
        dim=-1,  # 在最后一个维度上累加
        dtype=torch.int32  # 确保输出类型为int32
    )

    # 在累积长度序列的首尾添加padding
    # 1. 在开头添加0: [s1,s2,s3] -> [0,s1,s2,s3]
    # 2. 在末尾添加total_seq_len: [0,s1,s2,s3] -> [0,s1,s2,s3,total_seq_len]
    cu_seqlens = F.pad(
        F.pad(cu_seqlens, (1, 0), value=0),  # 首部padding
        (0, 1), value=total_seq_len  # 尾部padding
    )

    # 导入并更新ring_flash_attn参数
    from ring_flash_attn import update_ring_flash_attn_params
    update_ring_flash_attn_params(cu_seqlens, RING_ATTN_GROUP)
    

def convert_ring_attn_params(
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    packed_seq_lens: List[int],
    ring_attn_group: dist.ProcessGroup
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    转换和准备环形注意力机制所需的参数。在分布式环境中为每个rank分配并处理序列数据切片。

    该函数实现了序列数据的分布式切分，为环形注意力机制准备必要的参数。
    每个rank负责处理整体序列的一个子集，并生成相应的注意力掩码和位置编码。

    Args:
        sequences: 输入序列张量，形状为[batch_size, sequence_length, ...]
        attention_mask: 注意力掩码张量，形状为[batch_size, sequence_length]
        packed_seq_lens: 打包序列的长度列表
        ring_attn_group: 分布式进程组对象，用于环形注意力计算

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 包含以下经过处理的张量:
            - sequences: 当前rank负责的序列切片
            - attention_mask: 对应的注意力掩码切片
            - position_ids: 生成的位置编码

    Technical Details:
        1. 分布式参数计算:
           - 获取当前rank ID
           - 计算进程组大小
           - 确定序列切分范围

        2. 数据切分处理:
           - 计算每个rank的序列长度
           - 提取对应的序列切片
           - 生成相应的掩码和位置编码

        3. 参数更新:
           - 更新环形注意力参数
           - 处理序列边界
           - 维护全局一致性

    Example:
        >>> # 分布式环境中使用
        >>> model = DistributedModel()
        >>> sequences = torch.randn(2, 1000, 768)  # 批大小为2，序列长1000
        >>> mask = torch.ones(2, 1000)  # 注意力掩码
        >>> seq_lens = [400, 600]  # 两个序列的实际长度
        >>> 
        >>> # 转换参数
        >>> seq_slice, mask_slice, pos_ids = convert_ring_attn_params(
        ...     sequences, mask, seq_lens, dist_group
        ... )
        >>> 
        >>> # 使用转换后的参数
        >>> output = model(seq_slice, mask_slice, pos_ids)
    """
    # 获取当前rank在环形组中的ID
    ring_attn_rank = dist.get_rank(group=ring_attn_group)
    
    # 获取环形组的总进程数
    ring_attn_size = dist.get_world_size(group=ring_attn_group)
    
    # 计算总序列长度
    total_seq_len = sequences.numel()
    
    # 计算每个rank处理的序列长度
    local_seq_len = total_seq_len // ring_attn_size
    
    # 计算当前rank负责的序列范围
    start = ring_attn_rank * local_seq_len  
    end = (ring_attn_rank + 1) * local_seq_len
    
    # 提取当前rank的序列切片
    sequences = sequences[:, start:end]
    
    # 提取对应的注意力掩码切片
    attention_mask = attention_mask[:, start:end]
    
    # 生成位置编码
    position_ids = reset_ring_attn_position_ids(start, end, packed_seq_lens)
    
    # 更新环形注意力参数
    update_ring_attn_params(packed_seq_lens, total_seq_len)
    
    return sequences, attention_mask, position_ids