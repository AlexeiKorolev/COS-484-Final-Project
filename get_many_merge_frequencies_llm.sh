#!/bin/bash
# filepath: run_dump_frequencies.sh

EXPERIMENT_DIR="/mnt/a/Projects/COS 484/Final Project/tokenizer-attack/experiments/defence_custom"
CORPUS_DIR="/mnt/a/Projects/COS 484/Final Project/tokenizer-attack/small_oscar"
MODEL_NAME="V3"

# List your language codes here (should match your Python list)
# languages_to_download=(ru uk bg sr ja ko hi bn ta te ur)
# languages_to_download=(ru es fr de zh)
# languages_to_download=(ru uk bg sr ja ko hi bn ta te ur es fr de zh)
# languages_to_download=(ja)
# languages_to_download=(uk bg hi bn ta)
# languages_to_download=(uk bg hi bn ta code en)
languages_to_download=(en ru zh)

for lang in "${languages_to_download[@]}"
do
    echo "Processing language: $lang"
    python -m dump_frequencies \
        --experiment_dir "$EXPERIMENT_DIR" \
        --lang_code "$lang" \
        --corpus_dir "$CORPUS_DIR" \
        --model_name "$MODEL_NAME"
done