set -ex
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1

MODELS=(
    "your model path"
)
OUTPUT_DIRS=(
    "your output dir"
)
DATA_NAMES=(
    "folio,pw,ruletaker,logicqa,reclor"
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
      --max_tokens 4096 \
      --temperature 0
done
