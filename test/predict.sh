#!/bin/sh

python -m keras_segmentation predict \
 --checkpoints_path "models/vgg_unet_2021-09-29_132526.566811/vgg_unet" \
 --input_path 'data/mars_data_20210923/test_0.1/images/unnamed_testing_2_patch_011_00720_00480_r0.10.png' \
 --output_path 'out.png' \
 --read_image_type 0

