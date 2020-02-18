INFER_DATA_FILE="./data/stsa.binary/dev.tsv"
OUTPUT_DIR="./aug_output/"
ARGS_PATH="./output/training_args.bin"
PRETRAINED_MODEL_PATH="./output/checkpoint-500/pytorch_model.bin"
CONFIG_PATH="./output/config.json"
MODEL_TYPE="bert"
MODEL_NAME_OR_PATH="bert-base-uncased"
BATCH_SIZE=1

python augmentation.py\
    --data_file=${INFER_DATA_FILE}\
    --output_dir=${OUTPUT_DIR}\
    --args_path=${ARGS_PATH}\
    --pretrained_model_path=${PRETRAINED_MODEL_PATH}\
    --config_path=${CONFIG_PATH}\
    --per_gpu_infer_batch_size=${BATCH_SIZE}\
    --mlm\
    --fp16
