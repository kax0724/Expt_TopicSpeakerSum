# Add parent directory to python path to access lightning_base.py
export PYTHONPATH="../":"${PYTHONPATH}"
# the proper usage is documented in the README, you need to specify data_dir, output_dir and model_name_or_path

DATA_DIR=./data/samsum_dataset3 \
OUTPUT_DIR=./output/topspeak \

python runeval.py \
    model_dir $OUTPUT_DIR \
    dataset_dir $DATA_DIR  \
    --val \
    $@