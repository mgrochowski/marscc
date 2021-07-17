#!/usr/bin/env python3
# -*- coding: utf-8 -*-

DATA_URL = 'https://www.is.umk.pl/~grochu/mars/mars_training_data.zip'
INPUT_SIZE = (256, 256)

import numpy as np
import cv2
import tensorflow as tf
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
    y = np.zeros((h,w), dtype=np.uint8)
    for rgb_label in rgb_map:
        y[np.alltrue(x == rgb_map[rgb_label], axis=2)] = label_map[rgb_label]
      
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

#
# channels = 1  # ilosc kanalow wejsciowych
# classes = 3  # ilosc klas
# # input_img_path = 'THEMIS.png'
# # target_img_path = 'masks.png'
# input_img_path = 'CTX_basemap_1.png'
# target_img_path = 'mask_1.png'


# funkcja tworzy zbiór treningowy obrazów składający się z 'n' obrazow o wymiarach 'nx' na 'ny'
def build_samples(sample_count, train_x, train_y, image_size=INPUT_SIZE, channels=1, classes=3, **kwargs):

    nx, ny = image_size
    images = np.zeros((sample_count, nx, ny, channels))
    labels = np.zeros((sample_count, nx, ny, classes))

    train_x_norm = train_x / 255.0

    for i in range(sample_count):
        image, mask = sample_image(train_x_norm, train_y, out_size=image_size, **kwargs)
        images[i, ..., 0] = image
        for c in range(classes):
            labels[i, mask == c, c] = 1

    return images, labels


def download_data(target_dir='./data'):

    import urllib, zipfile, os
    zip_path = Path('mars_training_data.zip')
    print('Downloading: ', DATA_URL)
    urllib.request.urlretrieve(DATA_URL, zip_path)
    if zip_path.exists():
        print('Exstracting: ', zip_path)
        zf = zipfile.ZipFile(zip_path, "r")
        zf.extractall()
        zf.close()

        os.remove(zip_path)
        os.rename(Path('mars_training_data'), Path(target_dir))
    else:
        print('Download ERROR')


def save_samples(save_dir, images, labels):

    if not Path(save_dir).exists():
        Path(save_dir).mkdir(parents=True)

    k = 0

    for image, label in zip(images, labels):
        cv2.imwrite(str(Path(save_dir).joinpath('%05d_image.png' % k)), image * 255.0)

        for i, l in enumerate(label_list):
            mask_img = label.copy()
            mask_img[label[:, :, i] == 1.0, :] = rgb_map[l]

        cv2.imwrite(str(Path(save_dir).joinpath('%05d_mask.png' % k)), mask_img)
        k = k + 1


def load_data(sample_per_image=100, validation_split=0.25, file_path='data/train_files.txt',
              save_dir = None, **kwargs):

    if not Path(file_path).exists():
        download_data()

    data_dir = Path(file_path).parent

    # read images and masks list
    files = []

    with open(file_path, 'r') as f:
        for line in f:
            # print(line)
            files.append( ( str(data_dir.joinpath(x)) for x in line.split() ) )

    train_images, train_masks, val_images, val_masks  = [], [], [], []

    for image_path, mask_path in files:

        input_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        h, w = input_img.shape

        target_img = cv2.imread(mask_path)
        labels = image_to_labelmap(target_img)

        split = h - int(h * validation_split)

        val_count = int(sample_per_image * validation_split)
        val_x = input_img[split:, :]
        val_y = labels[split:, :]

        train_count = sample_per_image - val_count
        train_x = input_img[:split, :]
        train_y = labels[:split, :]

        images, masks = build_samples(train_count, train_x, train_y, **kwargs)
        train_images.append(images)
        train_masks.append(masks)

        images, masks = build_samples(val_count, val_x, val_y, **kwargs)
        val_images.append(images)
        val_masks.append(masks)

        train_images = np.concatenate(train_images, axis=0)
        train_masks = np.concatenate(train_masks, axis=0)
        val_images = np.concatenate(val_images, axis=0)
        val_masks =  np.concatenate(val_masks, axis=0)


        if save_dir is not None:
            save_samples(Path(save_dir) / 'train' , train_images, train_masks)
            save_samples(Path(save_dir) / 'val', val_images, val_masks)

        return [train_images, train_masks], [val_images, val_masks]

    # return [tf.data.Dataset.from_tensor_slices(_build_samples(train_count, train_x, train_y, **kwargs)),
    #         tf.data.Dataset.from_tensor_slices(_build_samples(train_count, train_x, train_y, **kwargs))]
    #
