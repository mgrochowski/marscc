#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import cv2
import numpy as np

# change: before 221122
rgb_map_v1 = {
       'background': [255, 255, 255],
       'cone':  [190, 190, 255],
       'crater':  [115, 178, 115]
   }

# change: after 221122
rgb_map_v2 = {
       'background': [255, 255, 255],
       'cone':  [137, 133, 214],
       'crater':  [115, 178, 115]
   }

rgb_map = rgb_map_v2

label_map = {
      'background': 0,
      'cone': 1,
      'crater': 2
    }

label_list = [
    'background',
    'cone',
    'crater'
    ]


def recognize_rgb_map(x):

    if np.any(np.alltrue(x == rgb_map_v1['cone'], axis=2)):
        return rgb_map_v1
    else:
        return rgb_map_v2


def image_to_labelmap(x, rgb_map=rgb_map, label_map=label_map):
    
    h, w, c = x.shape
    y = np.zeros((h, w), dtype=np.uint8)
    for rgb_label in rgb_map:
        y[np.alltrue(x == rgb_map[rgb_label], axis=2)] = label_map[rgb_label]
      
    return y


def labelmap_to_image(labels, rgb_map=rgb_map, label_map=label_map):

    h, w = labels.shape
    y = np.zeros((h, w, 3), dtype=np.uint8)
    for rgb_label in rgb_map:
        y[label_map[rgb_label] == labels, : ] = rgb_map[rgb_label]

    return y


def sample_image(x, y, out_size=(300, 300), max_zoom=1.0):

    height, width = x.shape[0] , x.shape[1]
    print('Input size: %dx%d' % (height, width))

    # draw position
    dx, dy = np.random.randint(0, width), np.random.randint(0, height)
    
    # draw size
    if max_zoom != 1.0:
        zoom = 1.0 + np.random.rand() * (max_zoom - 1.0)
    else:
        zoom = max_zoom

    print('zoom', zoom)
    
    pw, ph = int(out_size[0] * zoom), int(out_size[1] * zoom)    
    
    print('patch: ', pw, ph)
    
    dx = dx - pw // 2
    dy = dy - pw // 2

    print('Check borders')
    print('Init', [dx, dy, dx+pw, dy + ph])

    if dx < 0: dx = 0
    if dy < 0: dy = 0
    if dx + pw > width: dx = width - pw
    if dy + ph > height: dy = height - ph
    print('Fin ', [dx, dy, dx+pw, dy + ph])
 
    # crop
    sx = x[dy:dy+ph, dx:dx+pw]
    sy = y[dy:dy+ph, dx:dx+pw]
    
    # scale
    sx = cv2.resize(sx, out_size, interpolation=cv2.INTER_NEAREST)
    sy = cv2.resize(sy, out_size, interpolation=cv2.INTER_NEAREST)
    return sx, sy


def save_samples(save_dir, images, labels):

    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)

    k = 0

    for image, label in zip(images, labels):
        cv2.imwrite(str(Path(save_dir).joinpath('%05d_image.png' % k)), image * 255.0)

        mask_img = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.int32)
        for i, l in enumerate(label_list):
            mask_img[label[:, :, i] == 1.0, :] = rgb_map[l]

        cv2.imwrite(str(Path(save_dir).joinpath('%05d_mask.png' % k)), mask_img)
        k = k + 1


def split_image(input_image, output_width=450, output_height=450, overlap=0.5, padding=None):

    assert overlap >= 0.0
    if overlap < 1.0:
        overlap = int(min(output_width, output_height) * overlap)

    if padding is None or padding == 0:
        left_padding = 0
        right_padding = max(output_width, output_height) - overlap
    else:
        left_padding = padding
        right_padding = padding

    patches = []

    x_start, y_start = 0, 0

    height, width = input_image.shape[0], input_image.shape[1]

    image = input_image

    # padding: always add right_padding at right and bottom edges of image
    # if padding arg. is provided then also add padding at left and top egge
    if image.ndim == 2:
        image = np.pad(input_image, [[left_padding, right_padding], [left_padding, right_padding]])
    elif image.ndim == 3:
        image = np.pad(input_image, [[left_padding, right_padding], [left_padding, right_padding], [0, 0]])

    while y_start + output_height <= height + left_padding + right_padding:

        y_end = y_start + output_height

        while x_start + output_width <= width + left_padding + right_padding:

            x_end = x_start + output_width

            patch = image[y_start:y_end, x_start:x_end]
            patches.append((patch, (x_start, y_start, x_end, y_end)))
            x_start = x_start + output_width - overlap

        x_start = 0
        y_start = y_start + output_height - overlap

    return patches


def grayscale_to_rgb(im):

    return np.concatenate([im[..., np.newaxis]]*3, axis=2)
