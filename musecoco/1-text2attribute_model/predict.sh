#!/bin/bash

echo "Starting predict.sh script"

set -e # stops script if error

declare -a genres=("classical" "country" "jazz" "pop" "rock" "traditional")
declare -a moods=("angry" "exciting" "fear" "funny" "happy" "lazy" "magnificent" "quiet" "romantic" "sad" "warm")


for g in "${genres[@]}"
do 
    for m in "${moods[@]}"
    do 
    echo "test file"
    echo "./data/prompts/${g}/${m}_${g}_prompts.json"
    python muzic/musecoco/1-text2attribute_model/main.py \
    --do_predict \
    --model_name_or_path=XinXuNLPer/MuseCoco_text2attribute \
    --test_file="data/prompts/${g}/${m}_${g}_prompts.json" \
    --attributes="muzic/musecoco/1-text2attribute_model/data/att_key.json" \
    --num_labels="muzic/musecoco/1-text2attribute_model/num_labels.json" \
    --output_dir="muzic/musecoco/1-text2attribute_model/generation/${g}/${m}" \
    --overwrite_output_dir

    python muzic/musecoco/1-text2attribute_model/stage2_pre.py \
    --test_file="data/prompts/${g}/${m}_${g}_prompts.json" \
    --predictions="muzic/musecoco/1-text2attribute_model/generation/${g}/${m}/predict_attributes.json" \
    --probabilites="muzic/musecoco/1-text2attribute_model/generation/${g}/${m}/softmax_probs.json" \
    --attributes="muzic/musecoco/1-text2attribute_model/data/att_key.json" \
    --output_file="muzic/musecoco/2-attribute2music_model/data/controlability/${g}/infer-${m}.bin"

    done
done

