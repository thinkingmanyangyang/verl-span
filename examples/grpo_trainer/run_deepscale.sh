#!/bin/bash
export CUDA_VISIBLE_DEVICES=4,5,6,7

set -x

# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1

deepscaler_train_path=/data/yangzhenfei/DisCO/datasets/deepscaler/data/train.parquet
math=/data/yangzhenfei/DisCO/datasets/deepscaler/data/math.parquet
olympiad=/data/yangzhenfei/DisCO/datasets/deepscaler/data/olympiad_bench.parquet
amc=/data/yangzhenfei/DisCO/datasets/deepscaler/data/amc.parquet
aime=/data/yangzhenfei/DisCO/datasets/deepscaler/data/aime.parquet
model_path=/data/yangzhenfei/llm_checkpoint/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

MODEL_PATH="/data/yangzhenfei/llm_checkpoint/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Train over a single node, 8 A100-80GB GPUs.
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$deepscaler_train_path \
    data.val_files=$aime \
    data.train_batch_size=128 \
    data.val_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH  \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.strategy="fsdp2" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=10240 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','swanlab'] \
    trainer.project_name='DisCO' \
    trainer.experiment_name='1.5B-deepscale-4-gpu' \
    trainer.balance_batch=False  \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=5 \
    trainer.default_hdfs_dir=null \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.total_epochs=10 "${@:1}"