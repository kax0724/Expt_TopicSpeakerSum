# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"

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
    --data_dir /home/naraki/dialogsum/samsum_dataset_fixed \
    --output_dir output/2021-05-12-16-13-26 \
    --model_name_or_path google/pegasus-xsum \
    --gpus 1 \
    --logger_name wandb \
    --use_speaker_embeds \
    --partial_embed \
    --speaker_embed_scale 10 \
    --val_max_target_length 100 \
    --test_max_target_length 100 \
    --max_length 100 \
    --min_length 10 \
    --fixedspecialtoken \
    $@
