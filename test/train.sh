#!/bin/sh

python -m keras_segmentation train \
 --train_images "data/mars_data_20210923/train/images/" \
 --train_annotations "data/mars_data_20210923/train/annotations/" \
 --val_images "data/mars_data_20210923/val/images/" \
 --val_annotations "data/mars_data_20210923/val/annotations" \
 --n_classes=3 \
 --input_height 480 \
 --input_width 480 \
 --epochs 50 \
 --validate  \
 --checkpoints_path "logs/checkpoints/" \
 --model_name "vgg_unet" \
 --do_augment  \
 --read_image_type 0 \
 --steps_per_epoch 512
 --val_steps_per_epoch 512 \
 --batch_size 20 
 