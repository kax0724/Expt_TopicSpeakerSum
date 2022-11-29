#!/usr/bin/env bash
export PYTHONPATH="../":"${PYTHONPATH}"
today=`date "+%Y-%m-%d-%H-%M-%S"`

# Set --gradient_accumulation_steps  so that effective batch size is 256 (2*128, 4*64, 8*32, 16*16)
python finetune.py \
    --learning_rate=1e-4 \
    --do_train \
    --do_predict \
    --val_check_interval 0.25 \
    --max_source_length 512 --max_target_length 128 \
    --freeze_embeds --label_smoothing 0.1 --adafactor --task summarization \
    --train_batch_size 2 \
    --eval_batch_size 2 \
    --n_train -1 \
    --n_val -1 \
    --n_test -1 \
    --data_dir "/home/acc12119do/dialogsum/samsum_dataset" \
    --output_dir "output/${today}" \
    --gpus 1 --logger_name wandb \
    --model_name_or_path facebook/bart-large-xsum \
    --tokenizer_name facebook/bart-large \
    --warmup_steps 500 \
    "$@"
