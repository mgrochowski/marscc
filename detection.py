# process large input image and produce segmentation and list of detected objects

from __future__ import print_function, division, absolute_import, unicode_literals

from pathlib import Path

import click
import cv2
import numpy as np
from skimage.morphology import closing, square
from skimage.segmentation import clear_border

from utils.data import image_to_labelmap
from generate_training_data import split_image
from keras_segmentation_mod.predict import predict_multiple
from predict_ks import model_from_checkpoint_path

# @click.command()
# @click.option('--input_file', default=None, help='Input file')
# @click.option('--mask_file', default=None, help='Mask image')
# @click.option('--output_width', default=450, help='Width of output patch [in ptx]')
# @click.option('--output_height', default=450, help='Height of output patch [in ptx]')
# @click.option('--overlap', default=100, help='Patch overlaping size (in pixels or ratio)')
# @click.option('--output_file', default='output_file', help='Output file')
# @click.option('--resize_ratio', default=1.0, help='Scaling ratio')
# @click.option('--checkpoint_path', default=None, help='Path to model checkpoint')

def run(input_file, mask_file, output_width=450, output_height=450, overlap=100, resize_ratio=0.1,
        output_file='output.png', checkpoint_path='models/some_checkpoint'):

    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Exception('Cant open file %s' % input_file)

    h, w = image.shape[0], image.shape[1]

    if mask_file is not None:
        mask_img = cv2.imread(mask_file)

        if mask_img is None:
            raise Exception('Cant open file %s' % mask_file)

        assert h == mask_img.shape[0] and w == mask_img.shape[1]

    print('Input size: %dx%d' % (h, w))

    if resize_ratio != 1.0:
        h_new = int(h * resize_ratio)
        w_new = int(w * resize_ratio)

        image = cv2.resize(image, (h_new, w_new), interpolation=cv2.INTER_AREA)
        if mask_img is not None:
            mask_img = cv2.resize(mask_img, (h_new, w_new), interpolation=cv2.INTER_NEAREST_EXACT)

    if mask_img is not None:
        labels = image_to_labelmap(mask_img)

        # code labels as blue channel in RGB image
        pad_zeros = np.zeros((labels.shape[0], labels.shape[1], 2)).astype('uint8')
        ann_img = np.concatenate((np.expand_dims(labels, axis=2), pad_zeros), axis=2)

    patches = split_image(image, ann_img, output_width=output_width, output_height=output_height, overlap=overlap)

    input_images, input_ann = [], []
    for i, patch in enumerate(patches):
        patch_x, patch_y, patch_info = patch
        input_images.append(patch_x)
        input_ann.append(patch_y)

    model = model_from_checkpoint_path(checkpoint_path, 30)

    inps =input_images
    from keras_segmentation_mod.data_utils.data_loader import class_colors
    # class_colors =  ['red', 'green']

    # segmentation
    predictions = predict_multiple(model=model, inps=inps, inp_dir=None, out_dir=None,
                     checkpoints_path=None, overlay_img=False,
                     class_names=None, show_legends=False, colors=class_colors,
                     prediction_width=output_width, prediction_height=output_height, read_image_type=1)


    # join
    output_image = np.zeros((h_new, w_new))
    output_image2 = np.zeros((h_new, w_new))

    for i, patch in enumerate(patches):
        _, _, patch_info = patch
        sx, sy, ex, ey = patch_info
        output_image[sy:ey, sx:ex] = predictions[i]
        output_image2[sy:ey, sx:ex] = inps[i]

    a = 1
    # # save results
    # # create output directories structure
    # output_images = Path(output_dir + '/images/')
    # output_annotations = Path(output_dir + '/annotations/')
    # Path.mkdir(output_images, exist_ok=True, parents=True)
    # Path.mkdir(output_annotations, exist_ok=True)
    #
    # img_name = Path(input_file).stem
    # img_ext = Path(input_file).suffix
    #
    # # cv2.imwrite(str(output_images / img_name) + '_resized' + img_ext, image)
    # # cv2.imwrite(str(output_annotations / img_name) + '_resized' + img_ext, mask_img)
    # #
    # #
    # for i, patch in enumerate(patches):
    #     patch_x, patch_y, patch_info = patch
    #     file_name =  '%s_patch_%03d_%05d_%05d%s' % (img_name, i, patch_info[0], patch_info[1], img_ext)
    #     cv2.imwrite(str(output_images / file_name), patch_x)
    #     cv2.imwrite(str(output_annotations / file_name), patch_y)
    #
    # print('Saved %d patches to: %s' % (len(patches), output_dir))




if __name__ == '__main__':

    input_file = 'data\\test\\scale_500K\\unnamed_testing_1.png'
    mask_file = 'data\\test\\scale_500K\\unnamed_testing_1_mask.png'
    output_width = 480
    output_height = 480
    overlap = 0
    resize_ratio = 0.1
    output_file = 'detection_output.png'
    checkpoint_path = 'logs\\unet_mini_2021-09-27_003743.848447\\checkpoints\\unet_mini'
    run(input_file=input_file, mask_file=mask_file, output_width=output_width, output_height=output_height,
        overlap=overlap, resize_ratio=resize_ratio, output_file=output_file, checkpoint_path=checkpoint_path)
