TRAIN_DATA_FILE="./data/stsa.binary/train.tsv" 
OUTPUT_DIR="./output/"
MODEL_TYPE="bert"
EVAL_DATA_FILE="./data/stsa.binary/dev.tsv"
MODEL_NAME_OR_PATH="bert-base-uncased"

python train_mlmwithcls.py\
    --train_data_file=${TRAIN_DATA_FILE}\
    --output_dir=${OUTPUT_DIR}\
    --model_type=${MODEL_TYPE}\
    --eval_data_file=${EVAL_DATA_FILE}\
    --model_name_or_path=${MODEL_NAME_OR_PATH}\
    --do_train\
    --do_eval\
    --evaluate_during_training\
    --do_lower_case\
    --per_gpu_train_batch_size=32\
    --per_gpu_eval_batch_size=32\
    --learning_rate=1e-4\
    --num_train_epochs=10.0\
    --logging_steps=30\
    --overwrite_output_dir\
    --mlm\
    --fp16
