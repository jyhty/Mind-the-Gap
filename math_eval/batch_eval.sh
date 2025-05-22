set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1

MODELS=(
  "model-path1"
  "model-path2"
)
OUTPUT_DIRS=(
  "model-path1"
  "model-path2"
)
DATA_NAMES=(
  "amc23,gsm8k,math500,olympiadbench,odyssey,gaokao2023en"
  "amc23,gsm8k,math500,olympiadbench,odyssey,gaokao2023en"
)

for i in "${!MODELS[@]}"; do
    MODEL_NAME="${MODELS[i]}"
    OUTPUT_DIR="${OUTPUT_DIRS[i]}"
    DATA_NAME="${DATA_NAMES[i]}"

    echo "Running model: $MODEL_NAME"
    echo "Output directory: $OUTPUT_DIR"
    echo "Data name: $DATA_NAME"

    python3 -u eval.py \
      --model_name ${MODEL_NAME} \
      --output_dir ${OUTPUT_DIR}\
      --data_name ${DATA_NAME} \
      --max_tokens 2048 \
      --temperature 0 \
      --n_sampling 4 \
#      --few_shot
done
