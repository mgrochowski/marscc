from __future__ import print_function, division, absolute_import, unicode_literals

from pathlib import Path

import click
import cv2
import numpy as np
from skimage.color import label2rgb

from utils.image import image_to_labelmap, split_image, recognize_rgb_map


@click.command()
@click.option('--input_file', default=None, help='Input file')
@click.option('--mask_file', default=None, help='Mask image')
@click.option('--output_width', default=450, help='Width of output patch [in ptx]')
@click.option('--output_height', default=450, help='Height of output patch [in ptx]')
@click.option('--overlap', default=100, help='Patch overlapping size (in pixels or ratio)')
@click.option('--output_dir', default='output_dir', help='Output directory')
@click.option('--resize_ratio', default=1.0, help='Scaling ratio')
@click.option('--debug', default=False, is_flag=True, help='Create debug pictures')
def run(input_file, mask_file, output_width=450, output_height=450, overlap=100, resize_ratio=0.1,
        output_dir='output_dir', debug=False):

    generate(input_file=input_file, mask_file=mask_file, output_width=output_width, output_height=output_height,
             overlap=overlap, resize_ratio=resize_ratio, output_dir=output_dir, debug=debug)


def generate(input_file, mask_file, output_width=450, output_height=450, overlap=100, resize_ratio=0.1,
             output_dir='output_dir', debug=False):
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Exception('Cant open file %s' % input_file)

    h, w = image.shape[0], image.shape[1]

    mask_img = cv2.imread(mask_file)
    if mask_img is None:
        raise Exception('Cant open file %s' % mask_file)

    assert h == mask_img.shape[0] and w == mask_img.shape[1]
    print('Input size: %dx%d' % (h, w))

    if resize_ratio != 1.0:
        h_new = int(h * resize_ratio)
        w_new = int(w * resize_ratio)

        image = cv2.resize(image, (h_new, w_new), interpolation=cv2.INTER_AREA)
        mask_img = cv2.resize(mask_img, (h_new, w_new), interpolation=cv2.INTER_NEAREST_EXACT)
        print('New size: %dx%d' % (h_new, w_new))

    rgb_map = recognize_rgb_map(mask_img)
    labels = image_to_labelmap(mask_img, rgb_map=rgb_map)

    # code labels as blue channel in RGB image
    pad_zeros = np.zeros((labels.shape[0], labels.shape[1], 2)).astype('uint8')
    ann_img = np.concatenate((np.expand_dims(labels, axis=2), pad_zeros), axis=2)

    patches_img = split_image(image, output_width=output_width, output_height=output_height, overlap=overlap)
    patches_ann = split_image(ann_img, output_width=output_width, output_height=output_height, overlap=overlap)

    # save results
    # create output directories structure
    output_images = Path(output_dir + '/images/')
    output_annotations = Path(output_dir + '/annotations/')
    Path.mkdir(output_images, exist_ok=True, parents=True)
    Path.mkdir(output_annotations, exist_ok=True)

    output_debug = None
    if debug is True:
        output_debug = Path(output_dir + '/debug/')
        Path.mkdir(output_debug, exist_ok=True)

    img_name = Path(input_file).stem
    img_ext = Path(input_file).suffix

    # cv2.imwrite(str(output_images / img_name) + '_resized' + img_ext, image)
    # cv2.imwrite(str(output_annotations / img_name) + '_resized' + img_ext, mask_img)

    for i, patch in enumerate(patches_img):
        patch_x, patch_info = patch
        patch_y, patch_info_ann = patches_ann[i]

        assert patch_info == patch_info_ann

        file_name = '%s_patch_%03d_%05d_%05d_r%0.4f%s' % (img_name, i, patch_info[0], patch_info[1],
                                                          resize_ratio, img_ext)
        cv2.imwrite(str(output_images / file_name), patch_x)
        cv2.imwrite(str(output_annotations / file_name), patch_y)
        if debug is True:
            # patch_debug = np.hstack((patch_x, patch_y[:, :, 0]))
            patch_debug = label2rgb(patch_y[:, :, 0], image=patch_x, bg_label=0, alpha=0.5, kind='overlay')
            patch_debug = patch_debug * 255
            patch_debug = patch_debug.astype(np.uint8)

            if debug is True:
                cv2.imwrite(str(output_debug / file_name), patch_debug)

    print('Saved %d patches in: %s' % (len(patches_img), output_dir))


if __name__ == '__main__':
    run()
