#!/bin/sh

input_file='data/mars_test_500K/unnamed_testing_1.png'
checkpoint_path='models/vgg_unet_2021-09-29_132526.566811/vgg_unet'
output_dir='log/detection_output_vgg_unet'
resize_ratio=0.1


python predict_and_detect.py --input_file "${input_file}" \
                             --checkpoint_path "${checkpoint_path}" \
                             --resize_ratio ${resize_ratio} --overlap 0 \
                             --input_width 480 \
                             --input_height 480 \
                             --output_dir "${output_dir}"
