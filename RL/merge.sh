export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

python3 verl/scripts/model_merger.py \
    --backend fsdp \
    --is-value-model \
    --hf_model_path original_model_path \
    --local_dir actor_path \
    --target_dir save_path


