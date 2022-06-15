#!/bin/bash
# @Author: gunjianpan
# @Date:   2022-05-08 23:47:17
# @Last Modified by:   gunjianpan
# @Last Modified time: 2022-05-29 00:18:13

WORK_NUM=$(cat /proc/cpuinfo | grep "processor" | sort | uniq | wc -l)

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0,1,2 python -m torch.distributed.launch --nproc_per_node=3 vit_pruner.py \
    --output_dir ./vit_nn_pruning/ \
    --remove_unused_columns False \
    --do_eval \
    --do_train \
    --head_mask_str encoder1:2+3+4+6+7+8+9+11+12-encoder2:3+5+6+8+9+10-encoder3:5+6-encoder4:4+5+12-encoder5:3+4+9-encoder6:1+4+5+9+10+12-encoder7:3+4+5+6+8+9+11-encoder8:1+2+4+8+10-encoder9:1+2+3+7+8+11+12-encoder10:1+4+5+6+7+8+9+11-encoder11:2+3+4+5+6+7+8+9+10+11+12-encoder12:2+11 \
    --learning_rate 0.003 \
    --num_train_epochs 3 \
    --final_threshold 1.0 \
    --block_size_row 64 \
    --block_size_col 768 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 256 \
    --dataloader_num_workers ${WORK_NUM} \
    --dataloader_pin_memory False \
    --logging_strategy steps \
    --logging_steps 500 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --load_best_model_at_end True \
    --save_total_limit 3 \
    --model_name_or_path google/vit-base-patch16-384 \
    --distil_teacher_name_or_path google/vit-base-patch16-384 \
    --seed 1337