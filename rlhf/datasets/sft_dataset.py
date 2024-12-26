from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import zero_pad_sequences

from typing import Optional, Tuple, Union, List, Dict, Any


def preprocess_data(
    data: Dict[str, str],
    input_template: Optional[str] = None,
    input_key: str = "input",
    output_key: Optional[str] = None,
    apply_chat_template: Optional[Callable] = None
) -> Tuple[str, str]:
    """
    预处理训练数据，将输入数据转换为模型所需的prompt和response格式。

    该函数支持两种主要的预处理模式:
    1. 聊天模板模式(使用apply_chat_template)
    2. 简单模板模式(使用input_template)

    Args:
        data: 包含输入输出数据的字典
        input_template: 输入文本的格式化模板，可选
        input_key: 输入数据在字典中的键名，默认为"input"
        output_key: 输出数据在字典中的键名，可选
        apply_chat_template: 聊天模板应用函数，可选

    Returns:
        Tuple[str, str]: 包含处理后的(prompt, response)对

    Technical Details:
        1. 聊天模板处理模式:
           - 使用apply_chat_template函数处理
           - 支持有无输出的场景
           - 自动计算response部分

        2. 简单模板处理模式:
           - 直接使用input_template格式化
           - 支持空response的预训练场景
           - 保持原始数据结构
    """
    if apply_chat_template:
        if output_key:
            # 带输出的聊天模板处理
            # 1. 生成带generation_prompt的输入模板
            prompt = apply_chat_template(
                data[input_key],
                tokenizer=False,
                add_generation_prompt=True
            )
            # 2. 生成完整对话并截取response部分
            full_chat = apply_chat_template(
                data[input_key] + data[output_key],
                tokenizer=False
            )
            response = full_chat[len(prompt):]
        else:
            # 无输出的聊天模板处理(预训练场景)
            # 1. 使用除最后一个token外的输入生成prompt
            prompt = apply_chat_template(
                data[input_key][:-1],
                tokenizer=False,
                add_generation_prompt=True
            )
            # 2. 使用完整输入生成对话并截取response
            full_chat = apply_chat_template(
                data[input_key],
                tokenizer=False
            )
            response = full_chat[len(prompt):]
    else:
        # 简单模板处理模式
        prompt = data[input_key]
        if input_template:
            # 使用模板格式化输入
            prompt = input_template.format(prompt)
        # 处理输出(预训练时为空字符串)
        response = data[output_key] if output_key else ""
        
    return prompt, response


class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning (SFT)数据集类。
    
    用于处理和管理模型监督微调训练所需的数据集。支持预训练和微调两种模式，
    并提供并行数据处理能力。

    Args:
        dataset: 原始数据集对象
        tokenizer (Callable): 分词器对象，用于文本标记化
        max_length (int): 序列最大长度限制
        strategy: 训练策略对象，包含配置参数
        input_template (Optional[str]): 输入模板，默认为None
        pretrain_mode (bool): 是否为预训练模式，默认False
        num_processors (int): 并行处理器数量，默认8
        multiple_of (int): 序列长度倍数约束，默认1

    Attributes:
        tokenizer: 分词器实例
        strategy: 训练策略实例
        pretrain_mode: 预训练模式标志
        max_length: 最大序列长度
        multiple_of: 序列长度倍数
        input_template: 输入模板
        input_key: 输入数据键名
        output_key: 输出数据键名
        apply_chat_template: 聊天模板应用函数
        prompts: 处理后的提示文本
        response: 处理后的响应文本
        prompt_ids_lens: 提示文本的token长度

    Technical Details:
        1. 数据处理流程:
           - 加载原始数据集
           - 并行处理数据
           - 过滤无效数据
           - 保存处理结果

        2. 配置管理:
           - 支持模板配置
           - 动态参数获取
           - 模式切换

    Example:
        >>> # 创建数据集实例
        >>> tokenizer = AutoTokenizer.from_pretrained("model_name")
        >>> dataset = load_dataset("dataset_name")
        >>> sft_dataset = SFTDataset(
        ...     dataset=dataset,
        ...     tokenizer=tokenizer,
        ...     max_length=512,
        ...     strategy=training_strategy,
        ...     input_template="问题: {}\n回答:"
        ... )
        >>> 
        >>> # 访问数据
        >>> prompt, response = sft_dataset[0]
        >>> print(f"Prompt length: {len(prompt)}")

    Note:
        - 支持并行处理
        - 自动数据过滤
        - 灵活配置选项
        - 内存效率优化
    """

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: Callable,
        max_length: int,
        strategy: Any,
        input_template: Optional[str] = None,
        pretrain_mode: bool = False,
        num_processors: int = 8,
        multiple_of: int = 1,
    ) -> None:
        """
        初始化SFTDataset实例。

        Args:
            dataset: 输入数据集
            tokenizer: 分词器对象
            max_length: 最大序列长度
            strategy: 训练策略对象
            input_template: 输入模板(可选)
            pretrain_mode: 预训练模式标志
            num_processors: 并行处理器数量
            multiple_of: 序列长度倍数约束

        Note:
            - 初始化过程包含数据处理
            - 自动配置参数读取
            - 并行处理优化
        """
        super().__init__()
        
        # 基本配置
        self.tokenizer = tokenizer
        self.strategy = strategy
        self.pretrain_mode = pretrain_mode
        self.max_length = max_length
        self.multiple_of = multiple_of
        
        # 模板配置
        self.input_template = input_template
        # 从策略参数中获取配置
        self.input_key = getattr(self.strategy.args, "input_key", None)
        self.output_key = getattr(self.strategy.args, "output_key", None)
        self.apply_chat_template = getattr(
            self.strategy.args,
            "apply_chat_template",
            False
        )
        
        # 配置聊天模板
        if self.apply_chat_template:
            self.apply_chat_template = self.tokenizer.apply_chat_template
            tokenizer_chat_template = getattr(
                self.strategy.args,
                "tokenizer_chat_template",
                None
            )
            if tokenizer_chat_template:
                self.tokenizer.chat_template = tokenizer_chat_template
        
        # 并行处理数据集
        processed_dataset = dataset.map(
            self.process_data,
            remove_columns=dataset.column_names,
            num_proc=num_processors
        )
        # 过滤无效数据
        processed_dataset = processed_dataset.filter(
            lambda x: x["prompt"] is not None
        )
        
        # 保存处理结果
        self.prompts = processed_dataset["prompt"]
        self.responses = processed_dataset["response"]
        self.prompt_ids_lens = processed_dataset["prompt_ids_len"]
        
    def process_data(self, data: Dict[str, str]) -> Dict[str, Optional[Union[str, int]]]:
        """
        处理单条训练数据。支持预训练和微调两种模式，对输入数据进行标记化和长度验证。

        主要功能包括：
        1. 数据预处理和模板应用
        2. 标记化处理
        3. 长度验证和过滤
        4. 预训练/微调模式适配

        Args:
            data: 输入数据字典，包含原始文本

        Returns:
            Dict包含处理后的数据:
                - prompt: 处理后的提示文本，无效时为None
                - response: 处理后的回应文本
                - prompt_ids_len: prompt的token长度

        Technical Details:
            1. 预处理流程:
            - 应用模板(微调模式)
            - 文本标记化
            - 长度验证
            
            2. 模式处理:
            - 预训练模式: 简单处理
            - 微调模式: 完整处理流程
            
            3. 验证标准:
            - prompt非空
            - response非空(微调模式)
            - 长度限制检查

        Example:
            >>> # 微调模式处理
            >>> data = {
            ...     "input": "计算1+1等于几?",
            ...     "output": "1+1=2"
            ... }
            >>> result = process_data(data)
            >>> print(result)
            {
                "prompt": "问题: 计算1+1等于几?\n答案:",
                "response": "1+1=2",
                "prompt_ids_len": 15
            }
            
            >>> # 预训练模式处理
            >>> data = {"text": "这是一段预训练文本"}
            >>> result = process_data(data)
            >>> print(result)
            {
                "prompt": "这是一段预训练文本",
                "response": "",
                "prompt_ids_len": 0
            }
        """
        # 预处理数据
        prompt, response = preprocess_data(
            data,
            # 预训练模式不使用模板
            None if self.pretrain_mode else self.input_template,
            self.input_key,
            self.output_key,
            # 预训练模式不使用聊天模板
            apply_chat_template=None 
                if self.pretrain_mode 
                else self.apply_chat_template,
        )
        
        if not self.pretrain_mode:
            # 微调模式: 执行标记化和长度验证
            prompt_token = self.tokenizer(
                prompt,
                max_length=self.max_length,
                padding=False,
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False,
            )
            
            # 计算prompt的token长度
            prompt_ids_len = prompt_token["attention_mask"].int().sum().item()
            
            # 验证数据有效性:
            # 1. prompt非空
            # 2. response非空
            # 3. prompt长度在限制范围内(预留2个token给答案)
            if not prompt or not response or prompt_ids_len >= self.max_length - 2:
                prompt = None
        else:
            # 预训练模式: 不计算token长度
            prompt_ids_len = 0

        return {
            "prompt": prompt,
            "response": response,
            "prompt_ids_len": prompt_ids_len
        }
    
    def __len__(self):
        length = len(self.prompts)
        return length
    
    def __getitem__(self, idx: int) -> Tuple[int, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        获取处理后的单条训练数据。

        该方法实现了Dataset的__getitem__接口，用于获取经过标记化和处理的训练样本。
        支持预训练和微调两种模式，并确保特殊token(如EOS)的正确添加。

        Args:
            idx (int): 数据索引

        Returns:
            Tuple包含:
                - int: prompt的token长度
                - torch.Tensor: 输入token ID序列
                - torch.Tensor: attention mask序列
                - Dict: 包含原始输入、输出和长度信息的字典
        """
        # 获取基础数据
        prompt_ids_len = self.prompt_ids_lens[idx]
        prompt = self.prompts[idx]
        response = self.responses[idx]
        
        if not self.pretrain_mode:
            # 微调模式: 合并prompt和response
            # 1. 连接文本
            # 2. 移除尾部换行
            # 3. 确保EOS token存在
            text = (prompt + response).rstrip("\n")
            if not text.endswith(self.tokenizer.eos_token):
                text += " " + self.tokenizer.eos_token
        else:
            # 预训练模式: 直接使用prompt
            text = prompt

        # 标记化处理
        input_token = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=False,
            truncation=True,
            return_tensors="pt",
            add_special_tokens=False,
        )

        if not self.pretrain_mode:
            # 微调模式特殊处理:
            # 确保EOS token不被截断
            # 1. 设置最后一个token为EOS
            # 2. 确保attention mask有效
            input_token["input_ids"][0][-1] = self.tokenizer.eos_token_id
            input_token["attention_mask"][0][-1] = True
        
        # 构建信息字典
        info = {
            "input": prompt,
            "output": response,
            "input_length": input_token["attention_mask"].int().sum().item()
        }

        return (
            prompt_ids_len,
            input_token["input_ids"],
            input_token["attention_mask"],
            info
        )
    
    def collate_fn(self, item_list: List[Tuple[int, torch.Tensor, torch.Tensor, Dict]]) -> Tuple[
        List[int],
        torch.Tensor,
        torch.Tensor,
        Dict[str, List[str]]
    ]:
        """
        数据批处理整理函数。将多个样本整理成批量训练所需的格式。

        Args:
            item_list: 数据样本列表，每个样本包含:
                - prompt_ids_len (int): prompt的token长度
                - input_id (torch.Tensor): 输入token序列
                - attention_mask (torch.Tensor): 注意力掩码
                - info (Dict): 原始文本信息

        Returns:
            Tuple包含:
                - List[int]: prompt长度列表
                - torch.Tensor: 填充后的输入token序列
                - torch.Tensor: 填充后的注意力掩码
                - Dict: 原始文本信息字典
                    - "input": 输入文本列表
                    - "output": 输出文本列表
        """
        # 初始化收集器
        prompt_ids_lens = []  # prompt长度列表
        input_ids = []        # 输入token序列列表
        attention_masks = []  # 注意力掩码列表
        infos = {            # 原始文本信息
            "input": [],     # 输入文本列表
            "output": []     # 输出文本列表
        }

        # 收集批量数据
        for prompt_ids_len, input_id, attention_mask, info in item_list:
            # 收集prompt长度
            prompt_ids_lens.append(prompt_ids_len)
            # 收集输入序列
            input_ids.append(input_id)
            # 收集注意力掩码
            attention_masks.append(attention_mask)
            # 收集原始文本
            infos["input"].append(info["input"])
            infos["output"].append(info["output"])

        # 序列填充
        # 使用right padding策略对齐序列长度
        input_ids = zero_pad_sequences(
            input_ids,
            "right",
            self.tokenizer.pad_token_id
        )
        # 对齐注意力掩码
        attention_masks = zero_pad_sequences(
            attention_masks,
            "right"
        )
        
        return prompt_ids_lens, input_ids, attention_masks, infos
    
    def packing_collate_fn(self, item_list: List[Tuple[int, torch.Tensor, torch.Tensor, Dict]]) -> Tuple[
        List[int],
        torch.Tensor,
        torch.Tensor,
        Dict[str, List[int]]
    ]:
        """
        打包式批处理整理函数。将多个样本打包成单个序列，以提高计算效率。

        主要功能:
        1. 序列打包合并
        2. 位置索引标记
        3. 长度对齐padding
        4. 信息收集整理

        Args:
            item_list: 数据样本列表，每个样本包含:
                - prompt_ids_len (int): prompt的token长度
                - input_id (torch.Tensor): 输入token序列
                - attention_mask (torch.Tensor): 注意力掩码
                - info (Dict): 序列信息

        Returns:
            Tuple包含:
                - List[int]: prompt长度列表
                - torch.Tensor: 打包后的输入序列 [1, seq_len]
                - torch.Tensor: 打包后的位置索引 [1, seq_len]
                - Dict: 信息字典
                    - "input_length": 输入长度列表
        """
        # 初始化收集器
        packed_input_ids = []      # 打包的输入序列
        packed_attention_masks = [] # 打包的位置索引
        prompt_ids_lens = []       # prompt长度列表
        infos = {                  # 信息字典
            "input_length": []     # 输入长度列表
        }

        # 为每个序列分配唯一的位置索引
        index = 1
        # 处理每个样本
        for prompt_ids_len, input_id, attention_mask, info in item_list:
            # 展平并收集输入序列
            packed_input_ids.append(input_id.flatten())
            # 创建相同长度的位置索引
            packed_attention_masks.append(
                torch.full_like(input_id.flatten(), index)
            )
            # 收集prompt长度
            prompt_ids_lens.append(prompt_ids_len)
            # 收集输入长度信息
            infos["input_length"].append(info["input_length"])
            # 递增位置索引
            index += 1

        # 连接所有序列并添加批量维度
        packed_input_ids = torch.cat(
            packed_input_ids,
            dim=0
        ).unsqueeze(0)
        packed_attention_masks = torch.cat(
            packed_attention_masks,
            dim=0
        ).unsqueeze(0)
        
        # 处理长度约束
        if (self.multiple_of > 1 and 
            packed_input_ids.numel() % self.multiple_of != 0):
            # 计算需要填充的长度
            padding_len = self.multiple_of - (
                packed_input_ids.numel() % self.multiple_of
            )
            # 填充输入序列
            packed_input_ids = F.pad(
                packed_input_ids,
                (0, padding_len),
                value=self.tokenizer.pad_token_id
            )
            # 填充位置索引
            packed_attention_masks = F.pad(
                packed_attention_masks,
                (0, padding_len),
                value=0
            )

        return (
            prompt_ids_lens,
            packed_input_ids,
            packed_attention_masks,
            infos
        )