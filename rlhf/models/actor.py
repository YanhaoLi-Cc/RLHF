from typing import Optional, Tuple, Union

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import LoraLayer

import torch.nn as nn
import torch
import torch.distributed as dist

from .utils import log_probs_from_logits, reset_position_ids
from .ring_attn_utils import convert_ring_attn_params

class Actor(nn.Module):
    """Actor模型的基类
    
    该类用于封装预训练模型或已有模型，使其能够作为Actor模型使用。
    支持直接传入模型实例或模型路径，并提供了多种优化选项。

    Args:
        pretrain_or_model (Union[str, nn.Module]): 
            预训练模型实例或模型路径
            - 如果是字符串，将从HuggingFace加载对应的预训练模型
            - 如果是nn.Module实例，将直接使用该模型
        bf16 (bool, optional): 
            是否启用bfloat16精度计算，默认为True
        ds_config (dict, optional): 
            DeepSpeed配置字典，用于分布式训练优化
        device_map (dict, optional): 
            设备映射配置，用于指定模型各部分加载到的设备
        packing_samples (bool, optional): 
            控制训练过程中是否对样本进行打包。
            - 启用时，多个序列会被打包在一起以优化内存使用和训练效率。
            - 默认值: False。
        **kwargs: 
            额外的关键字参数

    注意:
        1. 当使用预训练模型路径时，会自动处理MoE（Mixture of Experts）模型的特殊配置
        2. 模型加载后会禁用推理缓存以优化训练性能
    """
    def __init__(
        self,
        pretrain_or_model: nn.Module,
        use_flash_attention_2: bool = False,
        bf16: bool = True,
        load_in_4bit: bool = False,
        lora_rank: int = 0,
        lora_alpha: int = 16,
        lora_dropout: int = 0,
        target_modules = None,
        ds_config: dict = None,
        device_map: dict = None,
        packing_samples=False,
        **kwargs   
    ) -> None:
        super().__init__()
        
        if isinstance(pretrain_or_model, str):
            attn_implementation = "flash_attention_2" if use_flash_attention_2 else "eager"
            
            # DeepSpeed ZeRO-3配置说明：
            # 如需使用DeepSpeed ZeRO-3，需要在模型实例化前创建HfDeepSpeedConfig对象
            # 详见：https://huggingface.co/docs/transformers/deepspeed#non-trainer-deepspeed-integration
            #
            # 注意：相关配置代码已注释，如需使用请取消注释并适当配置
            if ds_config is not None and ds_config['zero_optimization']['stage'] == 3:
                dschf = HfDeepSpeedConfig(ds_config)
            else:
                dschf = None
            
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
                
            # 从预训练模型路径加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrain_or_model,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                quantization_config=nf4_config,
                torch_dtype=torch.bfloat16 if bf16 else "auto",
                device_map=device_map,
            )
            
            # LoRA
            if lora_rank > 0:
                # https://github.com/huggingface/peft/issues/137
                self.model.enable_input_require_grads()
                lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=target_modules,
                    lora_dropout=lora_dropout,
                    bias="none",
                )
                self.model = get_peft_model(self.model, lora_config)

                if load_in_4bit:
                    for name, module in self.model.named_modules():
                        if isinstance(module, LoraLayer):
                            module = module.to(torch.bfloat16)
                        if "norm" in name:
                            module = module.to(torch.float32)
                        if "lm_head" in name or "embed_tokens" in name:
                            if hasattr(module, "weight"):
                                module = module.to(torch.bfloat16)
            
            # 处理MoE（Mixture of Experts）模型的特殊配置
            model_config = self.model.config.to_dict()
            if "output_router_logits" in model_config:
                print("[MoE] set output_router_logitis as True")
                self.model.config.output_router_logitis = True
            
            # 禁用模型推理缓存以优化训练性能    
            self.model.config.use_cache = False 
            
            # packing samples using Flash Attention 2
            self.packing_samples = packing_samples               
        else:
            # 直接使用传入的模型实例
            self.model = pretrain_or_model
        
    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor, 
        **kwargs
    ) -> Union[
        Tuple[torch.LongTensor, torch.LongTensor],
        Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor],
    ]:
        """使用预训练模型生成序列并进行后处理

        Args:
            input_ids (torch.Tensor): 输入序列的token ID
            **kwargs: 可选的生成参数
                - top_k (int, optional): Top-K采样的K值
                - top_p (float, optional): Top-P采样的概率阈值
                - do_sample (bool, optional): 是否使用采样策略，默认True
                - temperature (float, optional): 采样温度，默认1.0
                - num_beams (int, optional): 束搜索的束宽，默认1
                - attention_mask (torch.Tensor, optional): 注意力掩码
                - eos_token_id (int, optional): 结束标记的ID
                - pad_token_id (int, optional): 填充标记的ID
                - min_new_tokens (int, optional): 最少生成的新token数量，默认1
                - max_new_tokens (int, optional): 最多生成的新token数量
                - max_length (int, optional): 序列最大长度

        Returns:
            Union[
                Tuple[torch.LongTensor, torch.LongTensor],
                Tuple[torch.LongTensor, torch.LongTensor, torch.BoolTensor]
            ]: 返回处理后的序列和相关掩码
                - sequences: 生成的序列
                - attention_mask: 注意力掩码
                - action_mask: (可选) 动作掩码
        """
        # 1. 构建生成参数字典
        generate_args = {
            "input_ids": input_ids,
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "do_sample": kwargs.get("do_sample", True),
            "early_stopping": True,
            "temperature": kwargs.get("temperature", 1),
            "use_cache": True,
            "num_beams": kwargs.get("num_beams", 1),
            "attention_mask": kwargs.get("attention_mask"),
            "eos_token_id": kwargs.get("eos_token_id"),
            "pad_token_id": kwargs.get("pad_token_id"),
            "min_new_token": kwargs.get("min_new_tokens", 1),
        }
        
        # 2. 添加可选的长度控制参数
        if kwargs.get("max_new_tokens", None):
            generate_args["max_new_tokens"] = kwargs.get("max_new_tokens")
        if kwargs.get("max_length", None):
            generate_args["max_length"] = kwargs.get("max_length")

        # 3. 使用模型生成序列
        sequences = self.model.generate(**generate_args)
        
        # 4. 提取token ID并进行序列处理
        eos_token_id = generate_args['eos_token_id']
        pad_token_id = generate_args['pad_token_id']
        
        return self.process_sequences(
            sequences, 
            input_ids.size(1), 
            eos_token_id, 
            pad_token_id
        )
    
    def process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        """处理输入序列：生成注意力掩码并调整序列中的EOS标记位置
        
        Args:
            sequences: 输入序列张量
            input_len: 输入序列长度
            eos_token_id: 结束标记的ID
            pad_token_id: 填充标记的ID
            
        Returns:
            tuple: (处理后的序列, 注意力掩码, 动作掩码)
        """
        # 1. 生成初始注意力掩码
        # - 找出既不是EOS token也不是PAD token的位置
        # - ne()返回布尔张量，标记不等于指定值的位置为True
        # - 使用按位与(&)确保两个条件都满足
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)
        
        # 2. 处理EOS标记位置
        # - 通过fliplr()水平翻转张量，从右向左扫描
        # - argmax找到最后一个有效token的位置
        # - clamp(min=1)确保索引至少为1
        # - scatter_在找到的位置插入EOS标记
        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)
        
        # 3. 处理特殊模型（Llama3和Qwen2）的中间EOS标记
        # 3.1 找到每个序列第一个有效token的位置
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        
        # 3.2 创建位置索引矩阵
        # - 生成范围[0, seq_length)的索引序列
        # - 扩展到与输入序列相同的batch size
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        
        # 3.3 更新注意力掩码
        # - 只保留从第一个token到EOS token之间的位置
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)
        
        # 4. 处理强化学习所需的状态和动作序列
        # 在强化学习中，每个步骤包含：
        # - state_i: 当前token
        # - action_i: 下一个token的选择
        # - state_i+1: 选择的下一个token（等于action_i）
        #
        # 示例序列 "B -> C -> D" 的状态-动作对：
        # B(state) + C(action) -> C(next_state)
        # C(state) + D(action) -> D(next_state)
        
        # 4.1 提取状态序列（从input_len-1到倒数第二个位置）
        state_seq = sequences[:, input_len - 1 : -1]
        
        # 4.2 生成动作掩码
        # - 标记可以采取动作的位置（非EOS和非PAD的位置）
        # - 确保第一个位置始终可以采取动作
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1
        
        return sequences, attention_mask, action_mask
    
    def forward(
        self,
        sequences: torch.LongTensor,
        num_actions: Optional[Union[int, list[int]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output: bool = False,
        ring_attn_group: Optional[dist.ProcessGroup] = None,
        packed_seq_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """执行模型的前向传播，计算动作的对数概率。

        Args:
            sequences (torch.LongTensor): 输入序列tokens
                shape: (batch_size, sequence_length)
            num_actions (Optional[Union[int, list[int]]]): 要提取的动作数量
                - int: 所有序列使用相同的动作数量
                - list[int]: 为每个序列指定不同的动作数量
                - None: 直接返回模型输出
            attention_mask (Optional[torch.Tensor]): 注意力掩码
                shape: (batch_size, sequence_length)
                values: 1表示实际token，0表示padding
            return_output (bool): 是否返回模型的原始输出
            ring_attn_group (Optional[dist.ProcessGroup]): 环形注意力组
            packed_seq_lens (Optional[list[int]]): 打包序列的长度列表

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Any]]: 
                - 如果return_output=False: 返回动作的对数概率
                - 如果return_output=True: 返回(动作的对数概率, 模型输出)

        Examples:
            示例注意力掩码:
            attention_mask = [[1, 1, 1, 0, 0],    # 序列1：3个有效token，2个padding
                            [1, 1, 0, 0, 0]]     # 序列2：2个有效token，3个padding
        """
        if not self.packing_samples:
            # 计算位置编码
            # cumsum在最后一维度上累加，减1使位置编码从0开始
            position_ids = attention_mask.long().cumsum(-1) - 1
            # 输出示例:
            # [[0, 1, 2, 2, 2],
            #  [0, 1, 1, 1, 1]]

            # 处理padding位置的位置编码
            # 将padding位置(attention_mask=0)的position_ids设置为1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 输出示例:
            # [[0, 1, 2, 1, 1],
            #  [0, 1, 1, 1, 1]]
        else:
            if ring_attn_group is not None:
                sequences, attention_mask, position_ids = convert_ring_attn_params(
                    sequences, attention_mask, packed_seq_lens, ring_attn_group
                )
            else:
                position_ids = reset_position_ids(attention_mask)
            # explicitly ignore attention_mask for packing_samples
            attention_mask = None

        # 执行模型前向传播
        output = self.model(
            sequences, 
            attention_mask=attention_mask, 
            position_ids=position_ids
        )

        if num_actions is None:
            assert return_output, "当num_actions为None时，必须设置return_output=True"
            return output

        # 计算对数概率
        # 使用错位的方式计算下一个token的概率：当前token预测下一个token
        log_probs = log_probs_from_logits(
            output['logits'][:, :-1, :],  # 去掉最后一个token的预测
            sequences[:, 1:]              # 去掉第一个token的标签
        )

        # 提取指定数量的动作概率
        if not self.packing_samples:
            action_log_probs = log_probs[:, -num_actions:]
        else:
            pass

        if return_output:
            return (action_log_probs, output)
        return action_log_probs
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs={"use_reentrant": False}):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
        
    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()
    
    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()