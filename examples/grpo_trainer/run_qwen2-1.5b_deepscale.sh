export PYTHONPATH="/data/yangzhenfei/verl/deepscaler:$PYTHONPATH"


export CUDA_VISIBLE_DEVICES=2,3
#export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1

# export SWANLAB_API_KEY="bkHIjbNcQgol4Cldsp8IT"
export SWANLAB_API_KEY="gYbI1egFijpdmf4K0uYoS"
export WANDB_API_KEY='a552f7169e3e6889c5c9bb37306e265b0168ae02'
#export WANDB_API_KEY="4c3245840727bab2a2d846451f3bafbf7fa08e1e"

export HYDRA_FULL_ERROR=1

#WANDB_MODE="offline"
# source /home/lvshangke/anaconda3/etc/profile.d/conda.sh 
# source activate verl

# conda activate verl_env
# cd /home/lvshangke/verl-main
set -x

# export VLLM_ATTENTION_BACKEND=XFORMERS

# gsm8k_train_path=$HOME/data/gsm8k/train.parquet
# gsm8k_test_path=$HOME/data/gsm8k/test.parquet
# math_train_path=$HOME/data/math/train.parquet
# math_test_path=$HOME/data/math/test.parquet
deepscaler_train_path=/data/yangzhenfei/DisCO/datasets/deepscaler/data/train.parquet
math=/data/yangzhenfei/DisCO/datasets/deepscaler/data/math.parquet
olympiad=/data/yangzhenfei/DisCO/datasets/deepscaler/data/olympiad_bench.parquet
amc=/data/yangzhenfei/DisCO/datasets/deepscaler/data/amc.parquet
aime=/data/yangzhenfei/DisCO/datasets/deepscaler/data/aime.parquet
model_path=/data/yangzhenfei/llm_checkpoint/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# train_files="['$gsm8k_train_path', '$math_train_path']"
train_files="['$deepscaler_train_path']"
#test_files="["/home/lvshangke/dataset/deepscaler_noprompt/math.parquet","/home/lvshangke/dataset/deepscaler_noprompt/olympiad_bench.parquet","/home/lvshangke/dataset/deepscaler_noprompt/amc.parquet","/home/lvshangke/dataset/deepscaler_noprompt/aime.parquet"]"
test_files="['$math', '$olympiad', '$amc', '$aime']"

# train_batch_size=128
# val_batch_size=32
# ppo_mini_batch_size=32
# ppo_micro_batch_size_per_gpu=1


train_batch_size=1
val_batch_size=1
ppo_mini_batch_size=1
ppo_micro_batch_size_per_gpu=1

#ray stop 
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$deepscaler_train_path" \
    data.val_files="$aime" \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$model_path  \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='['console','swanlab']' \
    trainer.project_name='verl_grpo_cosreward' \
    trainer.experiment_name='deepseekdistill_1.5b_4gpu' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=30 $@
#/home/lvshangke/verl-main/checkpoints/verl_grpo_cosreward/deepscaler_lc_truly_penalty_length_control
# verl-main/checkpoints/verl_grpo_cosreward/deepscaler_lc_1_repub_penalty_0_agg