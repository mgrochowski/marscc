#!/bin/sh

python -m keras_segmentation evaluate_model \
 --checkpoints_path "models/vgg_unet_2021-09-29_132526.566811/vgg_unet" \
 --images_path "data/mars_data_20210923/test_0.1/images/" \
 --segs_path 'data/mars_data_20210923/test_0.1/annotations/' \
 --read_image_type 0


