set -ex
export CUDA_VISIBLE_DEVICES=0
export NCCL_P2P_DISABLE=1

OUTPUT_DIRS=(
    "your_dir"
)
DATA_NAMES=(
    "folio,pw,ruletaker,logicqa,reclor"
)

for i in "${!OUTPUT_DIRS[@]}"; do
    OUTPUT_DIR="${OUTPUT_DIRS[i]}"
    DATA_NAME="${DATA_NAMES[i]}"

    echo "Output directory: $OUTPUT_DIR"
    echo "Data name: $DATA_NAME"

    python3 -u acc.py \
      --output_dir ${OUTPUT_DIR}\
      --data_name ${DATA_NAME}
done
