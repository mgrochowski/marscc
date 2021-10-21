# Detection of cones and craters on Mars

The project is built on top of the ```keras_segmentation``` package. See https://github.com/divamgupta/image-segmentation-keras/ repository for more information on its use.

## Prerequisites

* Keras ( recommended version : 2.4.3 )
* OpenCV for Python
* Tensorflow ( recommended  version : 2.4.1 )
* imgaug 0.4
* scikit-image  ( recommended version 0.18.1 )

```shell
apt-get install -y libsm6 libxext6 libxrender-dev
pip install opencv-python
pip install git+https://github.com/aleju/imgaug.git
```

## Working Google Colab Example

Training, segmentation and detection demo:
https://colab.research.google.com/drive/1DZkEFsjyv-4P4nDqW9ojRxVNKaperUft?usp=sharing

## Mars surface segmentation results

Segmentation results using varius deep learning models are reported in [segmentation_results.ipynb](segmentation_results.ipynb) notebook.

## Usage via command line
### Train UNet model on Mars data


To run default training procedure simply run 
```shell
python train.py
```
or use ``keras_segmentation`` CLI interface

```shell
python -m keras_segmentation train \
 --train_images "data/mars_data_20210923/train/images/" \
 --train_annotations "data/mars_data_20210923/train/annotations/" \
 --val_images "data/mars_data_20210923/val/images/" \
 --val_annotations "data/mars_data_20210923/val/annotations" \
 --n_classes=3 \
 --input_height 480 \
 --input_width 480 \
 --epochs 50 \
 --validate  \
 --checkpoints_path "logs/checkpoints/" \
 --model_name "unet" \
 --do_augment  \
 --read_image_type 0 \
 --steps_per_epoch 512
 --val_steps_per_epoch 512 \
 --batch_size 20 
 ```

The training data used in this project can be found here: [mars_data_20210923.zip](https:///www.fizyka.umk.pl/~grochu/mars/mars_data_20210923.zip)  (39MB)
### Getting the predictions

Working with small images (e.g. 480x480)

```shell
python -m keras_segmentation predict \
 --checkpoints_path "models/unet_mini_2021-09-27_003743.848447/unet_mini" \
 --input_path "data/sample/image/unnamed_testing_1_patch_005_00240_00240_r0.10.png" \
 --output_path 'out.png' \
 --read_image_type 0
 ```

### Cones and craters detection on large image

Using large image (size 11812x11812) of Mars surface in scale 1:500K with default pre-trained model (VGG_Unet)

```shell
python predict_and_detect.py --input_file "data/mars_test_500K/unnamed_testing_1.png" \
                             --resize_ratio 0.1 \
                             --output_dir "detection_output"

```

At first run, the default model will be downloaded automatically (123MB)

The ```--checkpoint_path``` option allows you to use your own model 
```shell
python predict_and_detect.py --input_file "data/mars_test_500K/unnamed_testing_1.png" \
                             --resize_ratio 0.1 \
                             --checkpoint_path "models/unet_mini_2021-09-27_003743.848447/unet_mini" \
                             --output_dir "detection_output"

```

### Model Evaluation 

Compute IoU (intersection over union) score

```shell
python -m keras_segmentation evaluate_model \
 --checkpoints_path "models/vgg_unet_2021-09-29_132526.566811/vgg_unet" \
 --images_path "data/mars_data_20210923/test_0.1/images/" \
 --segs_path 'data/mars_data_20210923/test_0.1/annotations/' \
 --read_image_type 0
```



### Detect cones and craters on segmented image

```shell 
python detect.py --input_file "data/test/scale_500K/unnamed_testing_1_mask.png" \
                       --input_image "data/test/scale_500K/unnamed_testing_1.png" \
                       --min_area 10000 \
                       --min_perimeter  600 \
                       --min_solidity  0.98 \
                       --output_dir  detection_output
```


### Create training data 

Transform the image to the format required by the ``keras_segmentation``. The image size is reduced 10 times (``--resize_size 0.1``) and then divided into 480x480 pixel chunks.

```shell
python generate_training_data.py --input_file "data/test/scale_500K/unnamed_testing_1.png" \
                                 --mask_file "data/test/scale_500K/unnamed_testing_1_mask.png" \
                                 --resize_ratio 0.1 
                                 --output_width 480 \ 
                                 --output_height 480 \ 
                                 --overlap 50 \
                                 --output_dir "data/training/" 
```
