#!/bin/sh

python -m keras_segmentation predict \
 --checkpoints_path "models/vgg_unet_2021-09-29_132526.566811/vgg_unet" \
 --input_path 'data/mars_data_20210923/test_0.1/images/' \
 --output_path 'prediction_output/' \
 --read_image_type 0

