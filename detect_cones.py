from __future__ import print_function, division, absolute_import, unicode_literals

import imageio

from utils.image import image_to_labelmap, label_map
import click
import cv2

import numpy as np
# import pandas as pd
from skimage.io import imread, imshow
from skimage.color import rgb2gray, gray2rgb
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
from skimage.draw import rectangle_perimeter, set_color
from pathlib import Path


@click.command()
@click.option('--input_file', default=None, help='Input file with annotations (labels)')
@click.option('--input_image', default=None, help='Input file (image)')
@click.option('--min_area', default=10, help='Minimum object area [in ptx]')
@click.option('--min_perimeter', default=5, help='Minimum object perimeter [in ptx]')
@click.option('--min_solidity', default=0.5, help='Minimum object solidity')
@click.option('--output_dir', default='detect_cones_output', help='Output directory')
def run(input_file, input_image=None, min_area=10, min_perimeter=5, min_solidity=0.5, output_dir='detect_cones_output'):

    image = cv2.imread(input_file)
    if image is None:
        raise Exception('Cant open file %s' % input_file)

    h, w, _ = image.shape
    # plt.imshow(image)
    # plt.show()

    labels = image_to_labelmap(image)
    # plt.imshow(label)


    results = detect_cones_and_craters(labels, min_area=min_area, min_perimeter=min_perimeter, min_solidity=min_solidity)
    log = print_detections(results)

    o_dir = Path(output_dir)
    o_dir.mkdir(parents=True, exist_ok=True)

    i_name = Path(input_file).stem
    file_path = str(o_dir / i_name) + '_regions.log'
    with open(file_path, 'w') as f:
        f.write(log)

    i_name = Path(input_file).stem
    labels_reg = draw_regions2(image, results)
    file_path = str(o_dir / i_name) + '_lab_regions.png'
    cv2.imwrite(file_path, labels_reg)

    if input_image is not None:
        inp_img = cv2.imread(input_image)

        if inp_img is None:
            print('Warrning: cant open file %s' % input_image)
        else:
            inp_img_reg = draw_regions2(inp_img, results)
            i_name = Path(input_image).stem
            file_path = str(o_dir / i_name) + '_img_regions.png'
            cv2.imwrite(file_path, inp_img_reg)

    print('Results saved in %s' % output_dir)



def detect_cones_and_craters(labels, min_area=10, min_perimeter=5, min_solidity=0.5):

    label_names = ['cone', 'crater']
    detected = {}

    # detect
    for label in label_names:
        label_image = detectoin(labels, label=label_map[label])
        res = regionprops(label_image)

        detected[label] = res
        print('%s detected %d objects' % (label, len(res)))

    # filter
    for label in label_names:

        res = detected[label]
        res_filtered = []
        for region in res:

            # take regions with large enough areas
            if region.area >= min_area and region.perimeter >= min_perimeter and region.solidity >= min_solidity:
                res_filtered.append(region)
            else:
                print('Removing region, area %d, perimeter %f, solidity %f' % (region.area, region.perimeter, region.solidity))

        detected[label] = res_filtered

    return detected


from skimage.measure import label as label_region

# obsolete
def draw_regions(image, detected, color=None):

    if color is None: color = ['red', 'green']

    # # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)

    for i, label in enumerate(['cone', 'crater']):
        res = detected[label]
        for region in res:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor=color[i], linewidth=2)
            ax.add_patch(rect)

    plt.savefig('output.png')



def draw_regions2(image, detected, color=None, thickness=3):

    if color is None: color = [[0, 0, 255], [0, 255, 0]]

    img_reg = image.copy()
    if len(img_reg.shape) == 2:
        img_reg = gray2rgb(img_reg)

    alpha = 0.5

    for i, label in enumerate(['cone', 'crater']):
        res = detected[label]
        for region in res:
            # draw rectangle around segmented objects
            minr, minc, maxr, maxc = region.bbox
            # rr, cc = rectangle_perimeter(start=(minr, minc), end=(maxr, maxc), shape=img_reg.shape)
            # set_color(img_reg, (rr, cc), color[i], alpha=alpha)
            cv2.rectangle(img_reg, (minc, minr), (maxc, maxr), color=color[i], thickness=thickness)

    return img_reg


def print_detections(detected):

    text = ''
    # print (sorted by perimeter)
    for i, label in enumerate(['cone', 'crater']):
        res = detected[label]
        text += '\n%s\n\n' % label
        text += 'Region      area                   bbox          centroid        perimeter  solidity\n'

        for region in sorted(res, key=lambda x: getattr(x, 'perimeter')):
            text += '%5d  %9d  %25s  %8.1f %8.1f  %8.1f  %5.3f\n' % (
                region.label, region.area, str(region.bbox), region.centroid[0], region.centroid[1], region.perimeter,
                region.solidity)
    print(text)
    return text




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
