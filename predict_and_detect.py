# process large input image and produce segmentation and list of detected objects

from __future__ import print_function, division, absolute_import, unicode_literals

from pathlib import Path

import click
import cv2
import numpy as np

from detect import detect_cones_and_craters, print_detections, draw_regions2
from keras_segmentation.data_utils.data_loader import class_colors, get_image_array
from keras_segmentation.predict import predict_multiple, model_from_checkpoint_path, predict
from utils.image import labelmap_to_image, split_image, grayscale_to_rgb
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
@click.option('--norm', default='sub_and_divide', help='Imega normalization: sub_and_divide [-1, 1], sub_mean  [103.939, 116.779, 123.68], divide  [0,1]]')
def run(input_file, input_width=None, input_height=None, overlap=0, resize_ratio=0.1,
        output_dir='detection_output', checkpoint_path=None, norm='sub_and_divide'):

    heatmap, image = predict_large_image(input_file=input_file, input_width=input_width, input_height=input_height,
                                       overlap=overlap, resize_ratio=resize_ratio, checkpoint_path=checkpoint_path,
                                              output_type='heatmap', imgNorm=norm)

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
                        checkpoint_path=None, model=None, output_type='labels', imgNorm="sub_and_divide"):
    # output_type: 'labels' - return segmentation as labels, shape [width, height]
    #              'heatmap' - return segmentation as heatmap, shape [width, heaight, n_classes]

    if model is None:
        if checkpoint_path is None:
            checkpoint_path = download_model(target_dir='models')
        else:
            checkpoint_path = str(Path(checkpoint_path))

        model = model_from_checkpoint_path(checkpoint_path, input_width=input_width, input_height=input_height)

    print('Model input shape', model.input_shape)
    _, input_width, input_height, channels = model.input_shape

    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Exception('Cant open file %s' % input_file)

    h, w = image.shape[0], image.shape[1]
    print('Image original size: %dx%d' % (h, w))

    h_new, w_new = h, w
    if resize_ratio != 1.0:
        h_new = int(h * resize_ratio)
        w_new = int(w * resize_ratio)

        image = cv2.resize(image, (h_new, w_new), interpolation=cv2.INTER_AREA)

    if len(image.shape) == 2 and channels == 3:
        image_to_split = grayscale_to_rgb(image)
    else:
        image_to_split = image

    padding = max(input_height, input_width) - overlap
    patches = split_image(image_to_split, output_width=input_width, output_height=input_height, overlap=overlap,
                          padding=padding)

    input_images = []
    for i, patch in enumerate(patches):
        patch_x, patch_info = patch
        input_images.append(patch_x)

    # normalize data
    input_images = np.array([ get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING, imgNorm=imgNorm) for inp in input_images])

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

import matplotlib.pyplot as plt

def plot_predictions(images, targets, predictions):

    if isinstance(images, np.ndarray) and images.ndim == 2:
        # single grayscale image
        images, targets, predictions = [images], [targets], [predictions]

    fig, axs = plt.subplots(len(images), 5, figsize=(25, 5 * len(images)))

    if len(images) == 1:
        axs = [ axs ]

    for ax, image, label, prediction in zip(axs, images, targets, predictions):

        ax[0].matshow(image, cmap='Greys')
        ax[0].set_title('Input image')

        ax[1].matshow(label, vmin=0,  vmax=2)
        ax[1].set_title('Target')

        ax[2].matshow(prediction.argmax(axis=-1))
        ax[2].set_title('Segmentation')

        ax[3].matshow(prediction[:, :, 1], vmin=0.01, vmax=1.0, cmap='Reds')
        ax[3].set_title('Cone prediction')

        ax[4].matshow(prediction[:, :, 2], vmin=0.01, vmax=1.0, cmap='Reds')
        ax[4].set_title('Crater prediction')

    return fig

def predict_and_plot(input_file, target_file, resize_ratio=1.0, checkpoint_path=None, model=None, imgNorm="sub_and_divide"):

    heatmap, image = predict_large_image(input_file, resize_ratio=resize_ratio, checkpoint_path=checkpoint_path,
                                         model=model, output_type='heatmap', imgNorm=imgNorm)
    target_img = cv2.imread(target_file, 1)
    target_img = cv2.resize(target_img, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)
    fig = plot_predictions(image, target_img, heatmap)
    return fig

if __name__ == '__main__':

    run()
