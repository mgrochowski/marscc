#!/usr/bin/env python3
# -*- coding: utf-8 -*-

DATA_URL = 'https://www.is.umk.pl/~grochu/mars/mars_training_data.zip'

import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import urllib

rgb_map = {
       'cone' :  [190, 190, 255],
       'crater' :  [115, 178, 115]
   }

label_map = {
      'background' : 0,
      'cone' : 1,
      'crater' : 2
    }

def image_to_labelmap(x, rgb_map=rgb_map, label_map=label_map):
    
    h, w, c = x.shape
    y = np.zeros((h,w), dtype=np.uint8)
    for rgb_label in rgb_map:
        y[np.alltrue(x == rgb_map[rgb_label], axis=2)] = label_map[rgb_label]
      
    return y

def sample_image(x, y, out_size=(300, 300), max_zoom=3.0):
    # size = ( width, height )
    
    
    height, width = x.shape[0] , x.shape[1]
    
    # draw position
    dx, dy = np.random.randint(0, width), np.random.randint(0, height)
    
    # draw size
    zoom = 1.0 + np.random.rand() * (max_zoom - 1.0)
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
    sx = cv2.resize(sx, out_size, interpolation = cv2.INTER_AREA)
    sy = cv2.resize(sy, out_size)
    return sx, sy


channels = 1  # ilosc kanalow wejsciowych
classes = 3  # ilosc klas
# input_img_path = 'THEMIS.png'
# target_img_path = 'masks.png'
input_img_path = 'CTX_basemap_1.png'
target_img_path = 'mask_1.png'


# funkcja tworzy zbiór treningowy obrazów składający się z 'n' obrazow o wymiarach 'nx' na 'ny'
def _build_samples(sample_count, train_x, train_y, nx, ny, **kwargs):
    images = np.zeros((sample_count, nx, ny, channels))
    labels = np.zeros((sample_count, nx, ny, classes))
    for i in range(sample_count):
        image, mask = sample_image(train_x, train_y, out_size=(nx, ny), **kwargs)
        images[i, ..., 0] = image / 255.0
        labels[i, mask == 0, 0] = 1
        labels[i, mask == 1, 1] = 1
        labels[i, mask == 2, 2] = 1

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

# funkcja przygotowuje zbiór treningowy i walidacyjny.
def load_data(count, validation_split=0.25, **kwargs):

    input_img = cv2.imread(input_img_path, cv2.IMREAD_GRAYSCALE)
    h, w = input_img.shape

    target_img = cv2.imread(target_img_path)
    labels = image_to_labelmap(target_img)

    split = h - int(h * validation_split)

    val_count = int(count * validation_split)
    val_x = input_img[split:, :]
    val_y = labels[split:, :]

    train_count = count - val_count
    train_x = input_img[:split, :]
    train_y = labels[:split, :]

    return [tf.data.Dataset.from_tensor_slices(_build_samples(train_count, train_x, train_y, **kwargs)),
            tf.data.Dataset.from_tensor_slices(_build_samples(train_count, train_x, train_y, **kwargs))]

