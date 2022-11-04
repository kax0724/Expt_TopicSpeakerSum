# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"
DATA_DIR=./data/samsum_dataset3 \
OUTPUT_DIR=./output/topspeak \


# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path
python finetune.py \
    --learning_rate=3e-5 \
    --new_params_learning_rate=3e-5 \
    --fp16 \
    --gpus 1 \
    --do_train \
    --do_predict \
    --n_val 1000 \
    --val_check_interval 0.25 \
    --label_smoothing 0.1 \
    --max_source_length 512 \
    --max_target_length 100 \
    --adafactor \
    --task summarization \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --n_train -1 \
    --n_val -1 \
    --n_test -1 \
    --gpus 1 \
    --logger_name wandb \
    --use_speaker_embeds \
    --partial_embed \
    --data_dir $DATA_DIR \
    --train_batch_size 32 \
    --eval_batch_size 32 \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs 100 \
    --model_name_or_path google/pegasus-xsum \
    --save_top_k 5 \
    --early_stopping_patience 50 \
    --warmup_steps 10 \
    --speaker_embed_scale 10 \
    --val_max_target_length 100 \
    --test_max_target_length 100 \
    --max_length 100 \
    --min_length 10 \
    $@
