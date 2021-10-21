#!/bin/bash

python generate_training_data.py --input_file data/mars_images_scale_1_500K/Coprates1.png \
                                 --mask_file data/mars_images_scale_1_500K/Coprates1_mask.png \
                                 --output_width 500 --output_height 500 --overlap 50 \
                                 --output_dir tmp/generate_training_data_output/ \
                                 --resize_ratio 0.1 --debug
