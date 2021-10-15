# process large input image and produce segmentation and list of detected objects

from __future__ import print_function, division, absolute_import, unicode_literals

from pathlib import Path

import click
import cv2
import numpy as np

from detect import detect_cones_and_craters, print_detections, draw_regions2
from keras_segmentation.data_utils.data_loader import class_colors
from keras_segmentation.predict import predict_multiple, model_from_checkpoint_path
from utils.image import labelmap_to_image, split_image


@click.command()
@click.option('--input_file', default=None, help='Input image with Mars surface')
@click.option('--input_width', default=None, help='Model input width [in ptx]', type=int)
@click.option('--input_height', default=None, help='Model input height [in ptx]', type=int)
@click.option('--overlap', default=0, help='Patch overlaping size (in pixels or ratio)')
@click.option('--resize_ratio', default=1.0, help='Scaling ratio')
@click.option('--checkpoint_path', default=None, help='Path to model checkpoint')
@click.option('--output_dir', default='detection_output', help='Output directory')
def run(input_file, input_width=None, input_height=None, overlap=0, resize_ratio=0.1,
        output_dir='detection_output', checkpoint_path='models/some_checkpoint'):

    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Exception('Cant open file %s' % input_file)

    h, w = image.shape[0], image.shape[1]
    print('Image original size: %dx%d' % (h, w))

    model = model_from_checkpoint_path(checkpoint_path, input_width=input_width, input_height=input_height)
    print('Model input shape', model.input_shape)

    _, input_width, input_height, _ = model.input_shape

    h_new, w_new = h, w
    if resize_ratio != 1.0:
        h_new = int(h * resize_ratio)
        w_new = int(w * resize_ratio)

        image = cv2.resize(image, (h_new, w_new), interpolation=cv2.INTER_AREA)

    padding = max(input_height, input_width) - overlap
    patches = split_image(image, output_width=input_width, output_height=input_height, overlap=overlap,
                          padding=padding)

    input_images = []
    for i, patch in enumerate(patches):
        patch_x, patch_info = patch
        input_images.append(patch_x)

    inps = input_images

    # segmentation
    predictions = predict_multiple(model=model, inps=inps, inp_dir=None, out_dir=None,
                                   checkpoints_path=None, overlay_img=False,
                                   class_names=None, show_legends=False, colors=class_colors,
                                   prediction_width=output_width, prediction_height=output_height, read_image_type=1)

    # join
    output_image = np.zeros((h_new + padding, w_new + padding))

    for i, patch in enumerate(patches):
        _, patch_info = patch
        sx, sy, ex, ey = patch_info

        prediction = predictions[i]
        if (ey-sy, ex-sx) != prediction.shape[:2]:
            prediction = cv2.resize(prediction, (ey-sy, ex-sx), interpolation=cv2.INTER_NEAREST)
        output_image[sy:ey, sx:ex] = prediction

    # save results
    o_dir = Path(output_dir)
    o_dir.mkdir(parents=True, exist_ok=True)

    i_name = Path(input_file).stem
    output_image_rgb = labelmap_to_image(output_image[:h_new, :w_new])
    file_path = str(o_dir / i_name) + '_segmentation.png'
    cv2.imwrite(file_path, output_image_rgb)

    results = detect_cones_and_craters(labels=output_image[:h_new, :w_new], min_area=10, min_perimeter=5,
                                       min_solidity=0.5)

    log = print_detections(results)
    i_name = Path(input_file).stem
    file_path = str(o_dir / i_name) + '_regions.log'
    with open(file_path, 'w') as f:
        f.write(log)

    i_name = Path(input_file).stem
    image_reg = draw_regions2(image, results, thickness=1)
    file_path = str(o_dir / i_name) + '_img_regions.png'
    cv2.imwrite(file_path, image_reg)

    print('Results saved in %s' % output_dir)


if __name__ == '__main__':

    run()
