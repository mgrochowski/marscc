from keras_segmentation_mod.models.unet import vgg_unet, unet, unet_mini
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import datetime
from pathlib import Path
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
#
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
#     # session = InteractiveSession(config=config)
#
# session = InteractiveSession(config=config)



# model = vgg_unet(n_classes=3, input_height=480, input_width=480, channels=1 )
# model = unet(n_classes=3, input_height=480, input_width=480, channels=1 )
model = unet_mini(n_classes=3, input_height=480, input_width=480, channels=1 )

# When using custom callbacks, the default checkpoint saver is removed

timestamp = str(datetime.now()).replace(' ', '_').replace(':','')

# log_dir = 'logs/' + model.model_name + '_' + timestamp + '/'
log_dir = 'logs/' + model.model_name + "_" + timestamp
print('log_dir', log_dir)

checkpoint_path = Path(log_dir) / 'checkpoints'

Path.mkdir(checkpoint_path, parents=True)

callbacks = [
    ModelCheckpoint(
                filepath=str(checkpoint_path /  model.model_name ) + ".{epoch:05d}",
                save_weights_only=True,
                verbose=True
            ),
    # EarlyStopping(verbose=2, patience=3, monitor='val_accuracy', mode='max', restore_best_weights=True),  # cause error at early stopping event !
    TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False, update_freq=5)
]

# callbacks = None
model.train(
    train_images =  "data/mars_data_20210923/train/images/",
    train_annotations = "data/mars_data_20210923/train/annotations/",
    batch_size = 20,
    steps_per_epoch = 512,
    checkpoints_path =str( checkpoint_path / model.model_name ) ,
    epochs=100,
    val_images = "data/mars_data_20210923/val/images/",
    val_annotations = "data/mars_data_20210923/val/annotations",
    validate=True,
    val_steps_per_epoch=512,
    callbacks=callbacks,
    read_image_type=0,
    do_augment=True,
    ignore_zero_class=False
)