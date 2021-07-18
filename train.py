import unet
from utils import data
import tensorflow as tf

channels = 1
classes = 3
learning_rate = 1e-3
image_size = (256, 256)
epochs = 1000
n_samples = 1000

for layer_depth, filters_root in zip([3, 5], [32, 32]):
    print(layer_depth)

    tf.keras.backend.clear_session()

    unet_model = unet.build_model(*image_size, channels=channels,
                                  num_classes=classes, layer_depth=layer_depth,
                                  filters_root=filters_root, padding="same")

    unet.finalize_model(unet_model, learning_rate=learning_rate)

    train_dataset, validation_dataset = data.load_data(n_samples, validation_split=0.25, image_size=image_size,
                                                  max_zoom=4.0)

    trainer = unet.Trainer(name='mars/unet_%dx%d_ep_%d' % (layer_depth, filters_root, epochs), checkpoint_callback=True)
    trainer.fit(unet_model, train_dataset, validation_dataset, epochs=epochs, batch_size=32)