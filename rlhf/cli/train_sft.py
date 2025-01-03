
import argparse
import math
import os 
from datetime import datetime
from argparse import Namespace

from transformers.trainer import get_scheduler

from rlhf.models import Actor
from rlhf.trainer import SFTTrainer
from rlhf.datasets import SFTDataset
from rlhf.utils import blending_datasets, get_strategy, get_tokenizer

def train(args: argparse.Namespace) -> None:
    """
    执行强化学习模型的训练流程。
    
    该函数实现了完整的训练流程，包括：
    1. 设置分布式训练策略
    2. 初始化和配置Actor模型
    3. 准备训练和评估数据集
    4. 配置优化器和学习率调度器
    5. 执行训练循环
    6. 保存训练后的模型
    
    Args:
        args (argparse.Namespace): 包含所有训练配置参数的命名空间对象，包括：
            - pretrain (str): 预训练模型路径
            - flash_attn (bool): 是否使用Flash Attention 2
            - bf16 (bool): 是否使用BF16精度
            - load_in_4bit (bool): 是否使用4位量化
            - lora_rank (int): LoRA秩
            - lora_alpha (float): LoRA alpha参数
            - target_modules (list): 目标模块列表
            - lora_dropout (float): LoRA dropout率
            - gradient_checkpointing (bool): 是否启用梯度检查点
            - learning_rate (float): 学习率
            - max_samples (int): 最大样本数
            - max_len (int): 最大序列长度
            - train_batch_size (int): 训练批次大小
            - max_epochs (int): 最大训练轮数
            - save_path (str): 模型保存路径
            
    Returns:
        None: 函数不返回任何值，但会将训练好的模型保存到指定路径
        
    Raises:
        RuntimeError: 当分布式训练设置失败时抛出
        ValueError: 当参数配置无效时抛出
        
    Examples:
        >>> args = get_train_args()  # 获取训练参数
        >>> train(args)  # 执行训练
    """
    # 获取并设置分布式训练策略
    strategy = get_strategy(args)
    strategy.setup_distributed()
    
    # 初始化Actor模型
    model = Actor(
        args.pretrain,                    # 预训练模型路径
        use_flash_attention_2=args.flash_attn,  # 是否使用Flash Attention 2
        bf16=args.bf16,                   # 是否使用BF16精度
        load_in_4bit=args.load_in_4bit,   # 是否使用4位量化
        lora_rank=args.lora_rank,         # LoRA秩
        lora_alpha=args.lora_alpha,       # LoRA alpha参数
        target_modules=args.target_modules,# 目标模块
        lora_dropout=args.lora_dropout,    # LoRA dropout率
        ds_config=strategy.get_ds_train_config(is_actor=True),  # DeepSpeed配置
        packing_samples=args.packing_samples,  # 是否打包样本
    )

    # 配置tokenizer
    tokenizer = get_tokenizer(args.pretrain, model.model, "right", strategy, 
                            use_fast=not args.disable_fast_tokenizer)
    strategy.print(model)

    # 启用梯度检查点（如果指定）
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": args.gradient_checkpointing_use_reentrant}
        )

    # 配置优化器
    optim = strategy.create_optimizer(model, lr=args.learning_rate, 
                                    betas=args.adam_betas, weight_decay=args.l2)

    # 准备训练和评估数据集
    train_data, eval_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        strategy,
        args.seed,
        max_count=args.max_samples,
        train_split=args.train_split,
        eval_split=args.eval_split,
    )
    
    # 限制数据集大小
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
    
    # 创建训练和评估数据集实例
    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
        multiple_of=args.ring_attn_size,
    )
    eval_dataset = SFTDataset(
        eval_data,
        tokenizer,
        args.max_len,
        strategy,
        pretrain_mode=args.pretrain_mode,
        input_template=args.input_template,
        multiple_of=args.ring_attn_size,
    )

    # 配置数据加载器
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,  # shuffle
        True,  # drop_last
        train_dataset.packing_collate_fn if args.packing_samples else train_dataset.collate_fn,
    )
    eval_dataloader = strategy.setup_dataloader(
        eval_dataset,
        args.micro_train_batch_size,
        True,  # shuffle
        False, # drop_last
        eval_dataset.packing_collate_fn if args.packing_samples else eval_dataset.collate_fn,
    )

    # 计算训练步数并配置学习率调度器
    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = math.ceil(args.max_epochs * num_update_steps_per_epoch)

    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    # 准备模型、优化器和调度器
    (model, optim, scheduler) = strategy.prepare((model, optim, scheduler))

    # 加载检查点（如果存在）
    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    # 创建保存目录
    os.makedirs(args.save_path, exist_ok=True)

    # 配置训练器
    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        max_norm=args.max_norm,
        pretrain_mode=args.pretrain_mode,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        tokenizer=tokenizer,
    )

    # 开始训练
    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)

    # 在rank0上保存模型检查点
    strategy.save_model(model, tokenizer, args.save_path)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Checkpoint
    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_sft")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=1e8)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)

    # DeepSpeed
    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--bf16", action="store_true", default=False, help="Enable bfloat16")
    parser.add_argument("--zpg", type=int, default=1, help="ZeRO++ max partition size")
    parser.add_argument("--adam_offload", action="store_true", default=False, help="Offload Adam Optimizer")
    parser.add_argument("--flash_attn", action="store_true", default=False, help="Enable FlashAttention2")
    parser.add_argument("--grad_accum_dtype", type=str, default=None, help="Adam grad accum data type")
    parser.add_argument("--disable_trace_cache", action="store_true", default=False)
    parser.add_argument("--gradient_checkpointing_use_reentrant", action="store_true", default=False)
    parser.add_argument("--disable_fast_tokenizer", action="store_true", default=True)

    # SFT
    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--pretrain_mode", action="store_true", default=False, help="Use pretrain loss")
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--l2", type=float, default=0, help="weight decay loss")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam optimizer")

    # ring-attention
    parser.add_argument("--ring_attn_size", type=int, default=1, help="Ring attention group size")
    parser.add_argument(
        "--ring_head_stride",
        type=int,
        default=1,
        help="the number of heads to do ring attention each time. "
        "It should be a divisor of the number of heads. "
        "A larger value may results in faster training but will consume more memory.",
    )

    # LoRA
    parser.add_argument("--load_in_4bit", action="store_true", default=False)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--target_modules", type=str, nargs="*", default="all-linear")
    parser.add_argument("--lora_dropout", type=float, default=0)

    # packing SFT samples without CrossAttention
    parser.add_argument("--packing_samples", action="store_true", default=False)

    # custom dataset
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0", help="sampling probs for datasets")
    parser.add_argument("--train_split", type=str, default="train", help="train split of the HF dataset")
    parser.add_argument("--eval_split", type=str, default="test", help="test split of the dataset")

    parser.add_argument("--input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument("--output_key", type=str, default=None, help="JSON dataset key")
    parser.add_argument("--input_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument(
        "--apply_chat_template", action="store_true", default=False, help="Use HF tokenizer chat template"
    )
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=1e8, help="Max number of samples")
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens for the samples")

    # wandb parameters
    parser.add_argument("--use_wandb", type=str, default=None)
    parser.add_argument("--wandb_org", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="rlhf")
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default="sft_%s" % datetime.now().strftime("%m%dT%H:%M"),
    )

    # TensorBoard parameters
    parser.add_argument("--use_tensorboard", type=str, default=None, help="TensorBoard logging path")

    args = parser.parse_args()

    if args.input_template and "{}" not in args.input_template:
        print("[Warning] {} not in args.input_template, set to None")
        args.input_template = None

    if args.input_template and "\\n" in args.input_template:
        print(
            "[Warning] input_template contains \\n chracters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    if args.packing_samples and not args.flash_attn:
        print("[Warning] Please --flash_attn to accelerate when --packing_samples is enabled.")
        args.flash_attn = True

    # TODO: [packing samples]
    if args.ring_attn_size > 1:
        assert args.packing_samples, "packing_samples must be enabled when using ring attention"
    
    train(args)
