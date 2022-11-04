# Expt_DialogSum
Experimental Environment for Dialog Summarization


## Finetuning Training params
To override the pretrained model's training params, you can pass them to ./finetune.sh:

```
DATA_DIR=./data/samsum_dataset3 \
OUTPUT_DIR=./output/topspeak \

finetune.sh \
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
```