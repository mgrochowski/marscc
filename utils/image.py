#!/usr/bin/env python3
# -*- coding: utf-8 -*-

DATA_URL = 'https://www.is.umk.pl/~grochu/mars/mars_training_data.zip'
INPUT_SIZE = (256, 256)

import numpy as np
import cv2
from pathlib import Path
import urllib

rgb_map = {
       'background' : [255, 255, 255],
       'cone' :  [190, 190, 255],
       'crater' :  [115, 178, 115]
   }

label_map = {
      'background' : 0,
      'cone' : 1,
      'crater' : 2
    }

label_list = [
    'background',
    'cone',
    'crater'
    ]

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
    sx = cv2.resize(sx, out_size, interpolation = cv2.INTER_NEAREST)
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

