# 导入必要的库
import math
from typing import Iterator, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.utils.data.dataset import Dataset 
from torch.utils.data.sampler import Sampler

# 声明该模块公开的类
__all__ = ["DistributedSampler"]

# 定义协变类型变量
_T_co = TypeVar("_T_co", covariant=True)

class DistributedSampler(Sampler[_T_co]):
    """分布式采样器,用于将数据集限制加载到数据集的子集

    主要用于配合torch.nn.parallel.DistributedDataParallel使用。
    在这种情况下,每个进程可以传入一个DistributedSampler实例作为DataLoader的采样器,
    专门加载原始数据集的一个子集。

    注意:
        假设数据集大小是固定的,且任何实例总是以相同的顺序返回相同的元素。

    参数:
        dataset: 用于采样的数据集
        num_replicas (int, 可选): 参与分布式训练的进程数。默认从当前分布式组获取world_size
        rank (int, 可选): 当前进程在num_replicas中的序号。默认从当前分布式组获取
        shuffle (bool, 可选): 如果为True(默认值),采样器会打乱索引
        seed (int, 可选): 当shuffle=True时用于打乱采样器的随机种子。
                         在分布式组中所有进程应该使用相同的值。默认: 0
        drop_last (bool, 可选): 如果为True,采样器会丢弃数据集末尾的数据以确保能被副本数整除。
                               如果为False,采样器会添加额外的索引使数据能被副本数整除。默认: False
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None, 
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        consumed_samples = 0,
    ) -> None:
        # 如果未指定进程数,则从当前分布式组获取
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("需要分布式包可用")
            num_replicas = dist.get_world_size()
            
        # 如果未指定rank,则从当前分布式组获取
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("需要分布式包可用")
            rank = dist.get_rank()
            
        # 验证rank值的有效性
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"无效的rank {rank}, rank应该在区间[0, {num_replicas - 1}]内")

        # 初始化实例变量
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        # 计算每个进程应该分配的样本数
        # 如果启用drop_last且数据集长度不能被进程数整除
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # 向上取整以确保每个rank收到相同数量的数据
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        # 计算总样本数        
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.consumed_indicies = consumed_samples // self.num_replicas

    def __iter__(self) -> Iterator[_T_co]:
        """返回一个迭代器用于遍历采样的索引"""
        if self.shuffle:
            # 基于epoch和seed确定性地打乱
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # 添加额外的样本使其能被进程数整除
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # 移除尾部数据使其能被进程数整除
            indices = indices[:self.total_size]
        
        assert len(indices) == self.total_size

        # 对数据进行切片,每个进程获取其对应的部分
        indices = indices[self.rank:self.total_size:self.num_replicas]# skip consumed_samples
        indices = indices[self.consumed_indicies :]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        """返回采样器的长度"""
        return self.num_samples - self.consumed_indicies

    def set_epoch(self, epoch: int, consumed_samples: int=0) -> None:
        """设置采样器的epoch
        
        当shuffle=True时,这确保了所有副本在每个epoch使用不同的随机顺序。
        否则,这个采样器的下一次迭代将产生相同的顺序。

        参数:
            epoch (int): epoch编号
        """
        self.epoch = epoch
        self.consumed_indicies = consumed_samples // self.num_replicas