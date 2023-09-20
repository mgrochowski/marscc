# process large input image and produce segmentation and list of detected objects

from __future__ import print_function, division, absolute_import, unicode_literals

from pathlib import Path
import matplotlib.pyplot as plt

import click
import cv2
import numpy as np
from matplotlib import patches

from detect import detect_cones_and_craters, print_detections, draw_regions2
from keras_segmentation.data_utils.data_loader import get_image_array
from keras_segmentation.predict import model_from_checkpoint_path
from utils.image import labelmap_to_image, split_image, grayscale_to_rgb
from utils.download import download_model
from keras_segmentation.models.config import IMAGE_ORDERING
try:
    from tqdm import tqdm
except ImportError:
    print("tqdm not found, disabling progress bars")

    def tqdm(iter):
        return iter

@click.command()
@click.option('--input_file', default=None, help='Input image with Mars surface')
@click.option('--input_width', default=None, help='Model input width [in ptx]', type=int)
@click.option('--input_height', default=None, help='Model input height [in ptx]', type=int)
@click.option('--overlap', default=0, help='Patch overlapping size (in pixels or ratio)')
@click.option('--resize_ratio', default=1.0, help='Scaling ratio')
@click.option('--checkpoint_path', default=None, help='Path to model checkpoint')
@click.option('--output_dir', default='detection_output', help='Output directory')
@click.option('--norm', default='sub_and_divide', help='Image normalization: sub_and_divide [-1, 1], sub_mean  [103.939, 116.779, 123.68], divide  [0,1]]')
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

    results = detect_cones_and_craters(label_image=output_image, min_area=10,
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
                        checkpoint_path=None, model=None, output_type='labels', imgNorm="sub_and_divide",
                        batch_size=32):
    # output_type: 'labels' - return segmentation as labels, shape [width, height]
    #              'heatmap' - return segmentation as heatmap, shape [width, height, n_classes]

    if model is None:
        if checkpoint_path is None:
            checkpoint_path = download_model(target_dir='models')
        else:
            checkpoint_path = str(Path(checkpoint_path))

        model = model_from_checkpoint_path(checkpoint_path, input_width=input_width, input_height=input_height)

    # print('Model input shape', model.input_shape)
    _, input_width, input_height, channels = model.input_shape

    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise Exception('Cant open file %s' % input_file)

    h, w = image.shape[0], image.shape[1]
    # print('Image original size: %dx%d' % (h, w))

    h_new, w_new = h, w
    if resize_ratio != 1.0:
        h_new = int(h * resize_ratio)
        w_new = int(w * resize_ratio)

        image = cv2.resize(image, (h_new, w_new), interpolation=cv2.INTER_AREA)

    if len(image.shape) == 2 and channels == 3:
        image_to_split = grayscale_to_rgb(image)
    else:
        image_to_split = image

    assert max(input_height, input_width) > overlap
    left_padding = max(input_height, input_width) // 2
    image_patches = split_image(image_to_split, output_width=input_width, output_height=input_height, overlap=overlap,
                                padding=left_padding)


    input_images = []
    for i, patch in enumerate(image_patches):
        patch_x, patch_info = patch
        input_images.append(patch_x)

    # normalize data
    input_images = np.array([get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING, imgNorm=imgNorm) for inp in input_images])

    n_images = input_images.shape[0]
    n_steps = n_images // batch_size
    if n_images % batch_size != 0:
        n_steps = n_images // batch_size + 1

    # segmentation
    predictions = []
    start_ind = 0
    for i in tqdm(range(n_steps)):
        p = model.predict(input_images[start_ind:start_ind+batch_size], verbose=0)
        predictions.append(p)
        start_ind = start_ind + batch_size

    net_predictions = np.concatenate(predictions, axis=0)

    output_height = model.output_height
    output_width = model.output_width
    n_classes = net_predictions.shape[2]
    # predictions = np.argmax(net_predictions, axis=2).reshape((n_img, output_height, output_width))
    segmentation_heatmap = net_predictions.reshape((n_images, output_height, output_width, n_classes))

    # join
    # output_prediction = np.zeros((h_new + padding, w_new + padding))
    right_padding = max(input_height, input_width)
    output_segmentation = np.zeros((left_padding + h_new + right_padding, left_padding + w_new + right_padding, n_classes))

    output_normalization = 1.0
    if overlap > 0:
        output_normalization = np.zeros(output_segmentation.shape[:2])

    # merge predictions into single big image
    for i, patch in enumerate(image_patches):
        _, patch_info = patch
        sx, sy, ex, ey = patch_info

        # prediction = predictions[i]
        heatmap = segmentation_heatmap[i]
        if (ey-sy, ex-sx) != heatmap.shape[:2]:
            # prediction = cv2.resize(prediction, (ey-sy, ex-sx), interpolation=cv2.INTER_NEAREST)
            heatmap = cv2.resize(heatmap, (ey - sy, ex - sx), interpolation=cv2.INTER_NEAREST)

        # output_prediction[sy:ey, sx:ex] = prediction
        if overlap > 0:
            output_segmentation[sy:ey, sx:ex] = output_segmentation[sy:ey, sx:ex] + heatmap
            output_normalization[sy:ey, sx:ex] = output_normalization[sy:ey, sx:ex] + 1.0
        else:
            output_segmentation[sy:ey, sx:ex] = heatmap

    segmentation = output_segmentation[left_padding:left_padding + h_new, left_padding:left_padding + w_new]
    if overlap > 0:
        output_normalization = output_normalization[left_padding:left_padding + h_new, left_padding:left_padding + w_new]
        assert (output_normalization < 1.0).sum() == 0
        segmentation = segmentation / output_normalization[:, :, np.newaxis]

    if output_type == 'labels':
        segmentation = np.argmax(output_segmentation, axis=2)

    return segmentation, image


def plot_predictions(images, targets=None, predictions=None, heatmaps=None, bbox=None, texts=None, offset=20,
                     pred_bbox=None, min_confidence=None):

    if isinstance(images, np.ndarray) and images.ndim == 2:
        # single grayscale image
        images, targets, predictions, heatmaps, bbox, texts, pred_bbox = [images], [targets], [predictions], [heatmaps], [bbox], [texts], [pred_bbox]

    n_rows = len(images)

    n_cols = 5
    if targets is None or targets[0] is None:
        n_cols = n_cols - 1
        targets = [None] * n_rows
    if heatmaps is None or heatmaps[0] is None:
        n_cols = n_cols - 2
        heatmaps = [None] * n_rows
    if bbox is None or bbox[0] is None:
        bbox = [None] * n_rows
    if predictions is None or predictions[0] is None:
        predictions = [None] * n_rows
        if heatmaps[0] is None:
            n_cols = n_cols - 1
    if texts is None or texts[0] is None:
        texts = [None] * n_rows
    if pred_bbox is None or pred_bbox[0] is None:
        pred_bbox = [None] * n_rows

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_rows == 1:
        axs = [axs]

    for ax, image, target, prediction, heatmap, bb, text, pred_bb in zip(axs, images, targets, predictions, heatmaps, bbox, texts, pred_bbox):

        k = 0
        ax[k].imshow(image, cmap='gray')
        ax[k].set_title('Input image')
        ax_image = ax[k]

        if target is not None:
            k = k + 1
            ax[k].imshow(target, vmin=0,  vmax=2)
            ax[k].set_title('Target')

        ax_pred = None
        if prediction is not None or heatmap is not None:
            if prediction is None:
                if min_confidence is None:
                    prediction = heatmap.argmax(axis=-1)
                else:
                    prediction = np.zeros(heatmap.shape[:2])
                    for i_label in range(1, heatmap.shape[2]):
                        prediction[heatmap[:, :, i_label] >= min_confidence] = i_label
            k = k + 1
            ax[k].imshow(prediction, vmin=0,  vmax=heatmap.shape[2]-1)
            ax[k].set_title('Segmentation')
            ax_pred = ax[k]

        if heatmap is not None:
            k = k + 1
            ax[k].imshow(heatmap[:, :, 1], vmin=0.0, vmax=1.0, cmap='Reds')
            ax[k].set_title('Cone prediction')

            k = k + 1
            ax[k].imshow(heatmap[:, :, 2], vmin=0.0, vmax=1.0, cmap='Reds')
            ax[k].set_title('Crater prediction')

        if bb is not None:
            y1, x1, y2, x2 = bb

            for i in range(n_cols):
                ax[i].set_xlim((x1-offset, x2+offset))
                ax[i].set_ylim((y2+offset, y1-offset))

            rect = patches.Rectangle(xy=(x1, y1), width=x2-x1, height=y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            ax_image.add_patch(rect)

        if pred_bb is not None and not np.isnan(pred_bb).any() and ax_pred is not None:
            y1, x1, y2, x2 = pred_bb
            rect = patches.Rectangle(xy=(x1, y1), width=x2-x1, height=y2-y1, linewidth=1, edgecolor='r', facecolor='none')
            ax_pred.add_patch(rect)

        if text is not None:
            yy1, yy2 = ax[0].get_ylim()
            xx1, xx2 = ax[0].get_xlim()
            ax[0].text(xx1, yy2-np.abs(yy2-yy1) * 0.1, text, fontsize=12)
            plt.subplots_adjust(hspace=0.4)
    return fig


def predict_and_plot(input_file, target_file, resize_ratio=1.0, checkpoint_path=None, model=None, imgNorm="sub_and_divide"):

    heatmap, image = predict_large_image(input_file, resize_ratio=resize_ratio, checkpoint_path=checkpoint_path,
                                         model=model, output_type='heatmap', imgNorm=imgNorm)
    target_img = cv2.imread(target_file, 1)
    target_img = cv2.resize(target_img, (image.shape[0], image.shape[1]), interpolation=cv2.INTER_NEAREST)
    fig = plot_predictions(image, targets=target_img, heatmaps=heatmap)
    return fig


if __name__ == '__main__':

    run()
