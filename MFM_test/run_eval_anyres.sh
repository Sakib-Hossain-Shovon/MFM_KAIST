#!/bin/bash

export CUDA_VISIBLE_DEVICES=2,3
# MODEL_PATH=".checkpoints/llava-v1.6-rmfjausvicuna-7b-mfm-lora-anyres"
# Use the full path instead of starting with .
# MODEL_PATH= /data2/sakib/MFM_test/.checkpoints/lavares/lavanayres
MODEL_PATH=".checkpoints/lavares/lavanayres"
MODEL_BASE="liuhaotian/llava-v1.6-vicuna-7b"
DATA_ROOT="/data2/sakib/datasets/VisA"

python -u eval_llava_anyres.py \
    --model_path $MODEL_PATH \
    --model_base $MODEL_BASE \
    --visa_root $DATA_ROOT \
    --batch_size 1 \
    --conv_mode llava_v1 \
    --image_aspect_ratio anyres \
    --mm_patch_merge_type spatial_unpad \
    --image_grid_pinpoints '[[336,336],[336,672],[672,336]]' \
    --output_dir ./eval_results_checkpoint_v1_6_anyres > evaluation_log_v1_6_anyres.txt 2>&1
