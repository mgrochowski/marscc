# from keras_segmentation.models.unet import vgg_unet, unet, unet_mini, resnet50_unet
from keras_segmentation.models import pspnet
from keras_segmentation.models import unet
from keras_segmentation.models import segnet
# from keras_segmentation.models import fcn
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import datetime
from pathlib import Path
import tensorflow as tf

from keras.optimizers import Adam
from utils.image import grayscale_to_rgb

from utils.download import download_training_data
download_training_data()

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
#

# model = unet.unet_mini(n_classes=3, input_height=480, input_width=480, channels=1 )
# model = unet.unet(n_classes=3, input_height=480, input_width=480, channels=1 )
model = unet.vgg_unet(n_classes=3, input_height=480, input_width=480, channels=3 )
# model = unet.resnet50_unet(n_classes=3, input_height=480, input_width=480, channels=1 )
# model = unet.mobilenet_unet(n_classes=3, input_height=480, input_width=480, channels=1 )


# model = fcn.fcn_8( n_classes=3, input_height=480, input_width=480, channels=1 )
# model = fcn.fcn_8_vgg( n_classes=3, input_height=480, input_width=480, channels=1 )
# model = fcn.fcn_8_resnet50( n_classes=3, input_height=480, input_width=480, channels=1 )
# model = fcn.fcn_32( n_classes=3, input_height=480, input_width=480, channels=1 )
# model = fcn.fcn_8_vgg( n_classes=3, input_height=480, input_width=480, channels=1 )
# model = fcn.fcn_32_vgg( n_classes=3, input_height=480, input_width=480, channels=1 )
# model = fcn.fcn_32_resnet50( n_classes=3, input_height=480, input_width=480, channels=1 )
# model = fcn.fcn_32_mobilenet( n_classes=3, input_height=480, input_width=480, channels=1 )

# input requirement height % 192
# model = pspnet.pspnet(n_classes=3, input_height=384, input_width=384, channels=1 )
# model = pspnet.vgg_pspnet( n_classes=3, input_height=384, input_width=384, channels=1 )


# model = pspnet.pspnet_50( n_classes=3, input_height=473, input_width=473, channels=1 )
# input_shape == (473, 473) or  (713, 713):
# model = pspnet.pspnet_101( n_classes=3, input_height=473, input_width=473, channels=1 )

# model = segnet.segnet(n_classes=3, input_height=480, input_width=480, channels=1 )
# model = segnet.vgg_segnet(n_classes=3, input_height=480, input_width=480, channels=1 )
# model = segnet.resnet50_segnet(n_classes=3, input_height=480, input_width=480, channels=1 )

timestamp = str(datetime.now()).replace(' ', '_').replace(':','')
log_dir = 'logs/' + model.model_name + '_' + timestamp
checkpoint_path = log_dir +  '/checkpoints/' + model.model_name

callbacks = [
    ModelCheckpoint(
                filepath=checkpoint_path  + ".{epoch:05d}",
                save_weights_only=True,
                verbose=True
            ),
    # EarlyStopping(verbose=2, patience=3, monitor='val_accuracy', mode='max', restore_best_weights=True),  # cause error at early stopping event !
    TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False, update_freq=5)
]

optimizer_name = 'adam'
# optimizer_name = Adam(learning_rate=0.01)

preprocessing = None
if model.channels == 3:
    preprocessing = grayscale_to_rgb

model.train(
    # train_images =  "data/mars_data_20210923/train/images/",
    # train_annotations = "data/mars_data_20210923/train/annotations/",
    # train_images =  "data/mars_data_20220115/train/images/",
    # train_annotations = "data/mars_data_20220115/train/annotations/",
    # train_images =  "data/mars_data_20221207/train/images/",
    # train_annotations = "data/mars_data_20221207/train/annotations/",
    train_images="data/mars_data_20221207_x4_x8_x16/train/images/",                 # 7200 instances
    train_annotations="data/mars_data_20221207_x4_x8_x16/train/annotations/",
    optimizer_name=optimizer_name,
    batch_size = 20,
    steps_per_epoch = 1024,
    # steps_per_epoch = 2048,
    checkpoints_path =str(Path(checkpoint_path)),
    epochs=50,  # real epochs = epochs * (batch * steps) / instances = 50 * (20 * 1024) / 66984 = 15.3
                                               # np_one: real_epochs = 63,
                                               # x4_x8_x16 + mosaic = 140 epoch
    # val_images = "data/mars_data_20210923/val/images/",
    # val_annotations = "data/mars_data_20210923/val/annotations",
    # val_images = "data/mars_data_20221207/val/images/",
    # val_annotations = "data/mars_data_20221207/val/annotations",
    val_images="data/mars_data_20221207_x4_x8_x16/val/images/",         # 600 instances
    val_annotations="data/mars_data_20221207_x4_x8_x16/val/annotations",
    validate=True,
    # val_steps_per_epoch = 280,  # 5582 (instances) / 20 (batch_size)
    # val_steps_per_epoch=68,  # 1350 (instances) / 20 (batch_size) no_one
    val_steps_per_epoch=30,  # 600 (instances) / 20 (batch_size) no_one
    val_batch_size = 20,
    callbacks=callbacks,
    read_image_type=0,
    do_augment=True,
    ignore_zero_class=False,
    imgNorm = "sub_and_divide",    # x -> [-1, +1]
    # gen_use_multiprocessing=True,  # error on Windows
    verify_dataset=False,
    preprocessing=preprocessing
)