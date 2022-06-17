#!/usr/bin/env bash
    # Example on GTAV
     python -m torch.distributed.launch --nproc_per_node=2 train.py \
        --dataset gtav \
        --val_dataset bdd100k cityscapes synthia mapillary \
        --wild_dataset imagenet \
        --arch network.deepv3.DeepR50V3PlusD \
        --city_mode 'train' \
	--sgd \
        --lr_schedule poly \
        --lr 0.0025 \
        --poly_exp 0.9 \
        --max_cu_epoch 10000 \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --crop_size 768 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --rrotate 0 \
        --max_iter 60000 \
        --bs_mult 4 \
        --gblur \
        --color_aug 0.5 \
        --fs_layer 1 1 1 0 0 \
        --cont_proj_head 256 \
        --wild_cont_dict_size 393216 \
        --lambda_cel 0.1 \
        --lambda_sel 1.0 \
        --lambda_scr 10.0 \
        --date 0101 \
        --exp r50os16_gtav_wildnet \
        --ckpt ./logs/ \
        --tb_path ./logs/

