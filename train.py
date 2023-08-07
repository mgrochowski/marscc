from keras_segmentation.models.all_models import model_from_name
# from keras_segmentation.models.unet import vgg_unet, unet, unet_mini, resnet50_unet
# from keras_segmentation.models import unet
# from keras_segmentation.models import segnet
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
# model = unet.vgg_unet(n_classes=3, input_height=480, input_width=480, channels=3 )
# model = unet.resnet50_unet(n_classes=3, input_height=480, input_width=480, channels=3 )

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

# training configuration
configurations = {
   "unet_mini":        dict(n_classes=3, input_height=480, input_width=480, channels=1 ),
   "unet":            dict(n_classes=3, input_height=480, input_width=480, channels=1 ),
   "vgg_unet":         dict(n_classes=3, input_height=480, input_width=480, channels=3 ),
   "resnet50_unet":    dict(n_classes=3, input_height=480, input_width=480, channels=3 ),
   # "mobilenet_unet":   dict(n_classes=3, input_height=480, input_width=480, channels=3 ),

   "fcn_8":            dict( n_classes=3, input_height=480, input_width=480, channels=1 ),
   "fcn_8_vgg":        dict( n_classes=3, input_height=480, input_width=480, channels=3 ),
   "fcn_8_resnet50":   dict( n_classes=3, input_height=480, input_width=480, channels=3 ),
   "fcn_32":           dict( n_classes=3, input_height=480, input_width=480, channels=1 ),
   "fcn_32_vgg":       dict( n_classes=3, input_height=480, input_width=480, channels=3 ),
   "fcn_32_resnet50":  dict( n_classes=3, input_height=480, input_width=480, channels=3 ),
   # "fcn_32_mobilenet": dict( n_classes=3, input_height=480, input_width=480, channels=3 ),

   # input requirement height % 192
   "pspnet":           dict(n_classes=3, input_height=384, input_width=384, channels=1 ),
   "vgg_pspnet":       dict( n_classes=3, input_height=384, input_width=384, channels=3 ),
   "resnet50_pspnet":  dict( n_classes=3, input_height=384, input_width=384, channels=3 ),
   "pspnet_50":        dict( n_classes=3, input_height=473, input_width=473, channels=1 ),
   #t_shape ==  dict(473, 473) or  (713, 713):
   "pspnet_101":       dict( n_classes=3, input_height=473, input_width=473, channels=1 ),

   "segnet":           dict(n_classes=3, input_height=480, input_width=480, channels=1 ),
   "vgg_segnet":       dict(n_classes=3, input_height=480, input_width=480, channels=3 ),
   "resnet50_segnet":  dict(n_classes=3, input_height=480, input_width=480, channels=3 ),
}

selected_models = [
    #"unet_mini",
    #"unet",
    #"vgg_unet",
    # "resnet50_unet",
    # "mobilenet_unet",
    "fcn_8",
    "fcn_32",
    "pspnet",
    "segnet",
    "fcn_8_vgg",
    "fcn_8_resnet50",
    "fcn_32_vgg",
    "fcn_32_resnet50",
    # "fcn_32_mobilenet",
    "vgg_segnet",
    "resnet50_segnet"
    "vgg_pspnet",
    "pspnet_50",
    "pspnet_101",
    "resnet50_pspnet"
]

for key in selected_models:

    print('Start training: ', key)
    model = model_from_name[key](**configurations[key])

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
        # TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False, update_freq=5)
        TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=False, write_images=False, update_freq='epoch')
    ]

    initial_learning_rate = 0.001
    decay_rate = 0.5
    steps_per_epoch = 1024

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=(steps_per_epoch * 50),
        decay_rate=decay_rate,
        staircase=True)

    # optimizer_name = 'adam'
    optimizer_name = Adam(learning_rate=lr_schedule)

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
        steps_per_epoch = steps_per_epoch,
        # steps_per_epoch = 2048,
        checkpoints_path =str(Path(checkpoint_path)),
        epochs=300,  # real epochs = epochs * (batch * steps) / instances = 50 * (20 * 1024) / 66984 = 15.3
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

    print('End of training: ', key)
    tf.keras.backend.clear_session()

