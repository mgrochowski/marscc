from keras_segmentation_mod.models.unet import vgg_unet
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from datetime import datetime

model = vgg_unet(n_classes=3, input_height=480, input_width=480, channels=1 )

# When using custom callbacks, the default checkpoint saver is removed

timestamp = str(datetime.now()).replace(' ', '_')

log_dir = 'logs/' + model.model_name + '_' + timestamp + '/'
print('log_dir', log_dir)

callbacks = [
    ModelCheckpoint(
                filepath="checkpoints_" + timestamp + '/' + model.model_name + ".{epoch:05d}",
                save_weights_only=True,
                verbose=True
            ),
    EarlyStopping(),
    TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=False, update_freq=5)
]

model.train(
    train_images =  "../dane/coprates_1M_resize_0.2_480x480_overlap_120/images/",
    train_annotations = "../dane/coprates_1M_resize_0.2_480x480_overlap_120/annotations/",
    batch_size = 2,
    steps_per_epoch=50,
    checkpoints_path = "checkpoints/" + model.model_name  , epochs=5,
    val_images = "../dane/hydrotes_1M_resize_0.2_480x480_overlap_120/images/",
    val_annotations = "../dane/hydrotes_1M_resize_0.2_480x480_overlap_120/annotations/",
    validate=True,
    val_steps_per_epoch=50,
    callbacks=callbacks,
    read_image_type=0,
#     do_augment=True,
)