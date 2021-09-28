from __future__ import print_function, division, absolute_import, unicode_literals
from utils.image import image_to_labelmap
import click
import cv2

import numpy as np
# import pandas as pd
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)
from skimage.measure import label, regionprops, regionprops_table

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage import data
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.color import label2rgb
from pathlib import Path


@click.command()
@click.option('--input_file', default=None, help='Input file (mask image)')
@click.option('--min_area', default=10, help='Minimum object area [in ptx]')
@click.option('--min_perimeter', default=5, help='Minimum object perimeter [in ptx]')
@click.option('--min_solidity', default=0.5, help='Minimum object solidity')
@click.option('--output_file', default='output.png', help='Output image with detected regions')
def run(input_file, min_area=10, min_perimeter=5, min_solidity=0.5, output_file='output.png'):
    # img_mask='data/mask_0.png'

    image = cv2.imread(input_file)

    if image is None:
        raise Exception('Cant open file %s' % input_file)

    h, w, _ = image.shape
    # plt.imshow(image)
    # plt.show()

    labels = image_to_labelmap(image)
    # plt.imshow(label)

    results = detect_cones_and_craters(labels, min_area=min_area, min_perimeter=min_perimeter, min_solidity=min_solidity, output_file=output_file)



def detect_cones_and_craters(labels, min_area=10, min_perimeter=5, min_solidity=0.5, output_file='output.png'):

    color = ['red', 'green']
    detected = {}

    # detect
    for i, label in enumerate(['cone', 'crater']):
        label_image = detectoin(labels, label=i + 1)
        res = regionprops(label_image)

        detected[label] = res
        print('%s detected %d objects (color %s)' % (label, len(res), color[i]))


    # min_area = 0
    # min_perimeter = 0
    # min_solidity = 0

    # filter
    for i, label in enumerate(['cone', 'crater']):

        res = detected[label]
        res_filtered = []
        for region in res:

            # take regions with large enough areas
            if region.area >= min_area and region.perimeter >= min_perimeter and region.solidity >= min_solidity:
                res_filtered.append(region)
            else:
                print('Removing region, area %d, perimeter %f, solidity %f' % (region.area, region.perimeter, region.solidity))

        detected[label] = res_filtered

    # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(labels)

    for i, label in enumerate(['cone', 'crater']):
        res = detected[label]
        for region in res:

            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor=color[i], linewidth=2)
            ax.add_patch(rect)

    plt.savefig(output_file)

    # print (sorted by perimeter)
    for i, label in enumerate(['cone', 'crater']):
        res = detected[label]
        print('\n%s\n' % label)
        print('Region      area                       bbox          centroid        perimeter  solidity  ')

        for region in sorted(res,  key=lambda x: getattr(x, 'perimeter')):
            print('%5d  %9d  %25s  %8.1f %8.1f  %8.1f  %5.3f ' % (
            region.label, region.area, str(region.bbox), region.centroid[0], region.centroid[1], region.perimeter,
            region.solidity))

    return detected


from skimage.measure import label as label_region


def detectoin(x, label=1):
    label_img = (x == label).astype(np.int32)

    # apply threshold
    bw = closing(label_img, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label_region(cleared, background=0)
    return label_image



if __name__ == '__main__':

    run()
