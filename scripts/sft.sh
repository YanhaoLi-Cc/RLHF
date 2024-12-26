set -x
export HF_HOME="/home/liyanhao/huggingface_cache"

read -r -d '' training_commands <<EOF
rlhf.cli.train_sft \
   --max_len 2048 \
   --dataset Open-Orca/OpenOrca \
   --input_key question \
   --output_key response \
   --train_batch_size 256 \
   --micro_train_batch_size 2 \
   --max_samples 5000 \
   --pretrain /home/liyanhao/luo_hm/RLHF/models/vicuna-7b-v1.5 \
   --save_path ./checkpoint/vicuna-7b-v1.5-sft \
   --save_steps -1 \
   --logging_steps 1 \
   --eval_steps -1 \
   --zero_stage 3 \
   --max_epochs 1 \
   --bf16 \
   --flash_attn \
   --learning_rate 5e-6 \
   --load_checkpoint \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY 
EOF
    # --wandb [WANDB_TOKENS]
    # --packing_samples

if [[ ${1} != "slurm" ]]; then
    deepspeed --include localhost:0,1,2,3 --module $training_commands
fi