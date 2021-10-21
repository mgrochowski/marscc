#!/bin/sh

python detect.py --input_file data/mars_images_scale_1_500K/Coprates1_mask.png \
                       --input_image data/mars_images_scale_1_500K/Coprates1.png \
                       --min_area 10000 \
                       --min_perimeter  600 \
                       --min_solidity  0.98 \
                       --output_dir  tmp/detect_cones_output/
