set -ex
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_P2P_DISABLE=1

MODEL_NAME="model-path1"
OUTPUT_DIR="model-path1"
DATA_NAME="amc23,gaokao2023en,gsm8k,math500,olympiadbench,odyssey"

python3 -u eval.py \
    --model_name ${MODEL_NAME} \
    --output_dir ${OUTPUT_DIR}\
    --data_name ${DATA_NAME} \
    --max_tokens 2048 \
    --temperature 0 \
    --n_sampling 4