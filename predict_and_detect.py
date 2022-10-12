# process large input image and produce segmentation and list of detected objects

from __future__ import print_function, division, absolute_import, unicode_literals

from pathlib import Path

import click
import cv2
import numpy as np

from detect import detect_cones_and_craters, print_detections, draw_regions2
from keras_segmentation.data_utils.data_loader import class_colors, get_image_array
from keras_segmentation.predict import predict_multiple, model_from_checkpoint_path
from utils.image import labelmap_to_image, split_image
from utils.download import download_model
from keras_segmentation.models.config import IMAGE_ORDERING

@click.command()
@click.option('--input_file', default=None, help='Input image with Mars surface')
@click.option('--input_width', default=None, help='Model input width [in ptx]', type=int)
@click.option('--input_height', default=None, help='Model input height [in ptx]', type=int)
@click.option('--overlap', default=0, help='Patch overlaping size (in pixels or ratio)')
@click.option('--resize_ratio', default=1.0, help='Scaling ratio')
@click.option('--checkpoint_path', default=None, help='Path to model checkpoint')
@click.option('--output_dir', default='detection_output', help='Output directory')
def run(input_file, input_width=None, input_height=None, overlap=0, resize_ratio=0.1,
        output_dir='detection_output', checkpoint_path=None):

    heatmap, image = predict_large_image(input_file=input_file, input_width=input_width, input_height=input_width,
                                       overlap=overlap, resize_ratio=resize_ratio, checkpoint_path=checkpoint_path,
                                              output_type='heatmap')

    output_image = np.argmax(heatmap, axis=2)
    
    # save results
    o_dir = Path(output_dir)
    o_dir.mkdir(parents=True, exist_ok=True)

    i_name = Path(input_file).stem
    output_image_rgb = labelmap_to_image(output_image)
    file_path = str(o_dir / i_name) + '_segmentation.png'
    cv2.imwrite(file_path, output_image_rgb)

    results = detect_cones_and_craters(labels=output_image, min_area=10, min_perimeter=5,
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


def predict_large_image(input_file, input_width=None, input_height=None, overlap=0, resize_ratio=0.1,
                        checkpoint_path=None, output_type='labels'):
    # output_type: 'labels' - return segmentation as labels, shape [width, height]
    #              'heatmap' - return segmentation as heatmap, shape [width, heaight, n_classes]

    if checkpoint_path is None:
        checkpoint_path = download_model(target_dir='models')
    else:
        checkpoint_path = str(Path(checkpoint_path))

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

    # normalize data
    input_images = np.array([ get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING) for inp in input_images])

    # segmentation
    net_predictions = model.predict(input_images)

    output_height = model.output_height
    output_width = model.output_width
    n_img = input_images.shape[0]
    n_classes = net_predictions.shape[2]
    # predictions = np.argmax(net_predictions, axis=2).reshape((n_img, output_height, output_width))
    segmentation_heatmap = net_predictions.reshape((n_img, output_height, output_width, n_classes))

    # join
    # output_prediction = np.zeros((h_new + padding, w_new + padding))
    output_segmentation = np.zeros((h_new + padding, w_new + padding, n_classes))

    for i, patch in enumerate(patches):
        _, patch_info = patch
        sx, sy, ex, ey = patch_info

        # prediction = predictions[i]
        heatmap = segmentation_heatmap[i]
        if (ey-sy, ex-sx) != heatmap.shape[:2]:
            # prediction = cv2.resize(prediction, (ey-sy, ex-sx), interpolation=cv2.INTER_NEAREST)
            heatmap = cv2.resize(heatmap, (ey - sy, ex - sx), interpolation=cv2.INTER_NEAREST)

        # output_prediction[sy:ey, sx:ex] = prediction
        output_segmentation[sy:ey, sx:ex] = heatmap

    x = output_segmentation[:h_new, :w_new]
    if output_type == 'labels':
        x = np.argmax(x, axis=2)

    return x, image



if __name__ == '__main__':

    run()
