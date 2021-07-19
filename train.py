from __future__ import print_function, division, absolute_import, unicode_literals
import click

import unet
from utils import data

@click.command()
@click.option('--output_path', default=None, help='Checkpoint directory (default: ./mars/)')
@click.option('--epochs', default=100, help='Number of epoches (default: 100)')
@click.option('--layers', default=3, help='Number of layers (default: 3)')
@click.option('--filters_root', default=32, help='Number of kernels in first layer (default: 32)')
@click.option('--learning_rate', default=1e-3, help='Initial learning rate (default: 1e-3)')
@click.option('--dropout_rate', default=0.5, help='Dropout rate (default: 0.5)')
@click.option('--samples_per_image', default=1000, help='Number of training samples drawn for single input image (default: 1000)')
@click.option('--image_size', default=256, help='Size of input image (image_size x image_size) (default: 256)')
@click.option('--max_zoom', default=4.0, help='Range of image scaling for data augumentation (default: 4.0)')
def train(output_path, epochs, layers, filters_root, dropout_rate, learning_rate, samples_per_image,
          image_size, max_zoom):

    channels = 1
    classes = 3
    image_size = (image_size, image_size)
    n_samples = samples_per_image
    layer_depth = layers
    batch_size = 32
    if output_path is None:
        output_path = 'mars/unet_%dx%d_ep_%d' % (layer_depth, filters_root, epochs)

    unet_model = unet.build_model(*image_size, channels=channels,
                                      num_classes=classes, layer_depth=layer_depth,
                                      filters_root=filters_root, padding="same",
                                      dropout_rate=dropout_rate)

    unet.finalize_model(unet_model, learning_rate=learning_rate)

    train_dataset, validation_dataset = data.load_data(n_samples, validation_split=0.25, image_size=image_size,
                                                           max_zoom=max_zoom)

    trainer = unet.Trainer(name=output_path, checkpoint_callback=True)

    trainer.fit(unet_model, train_dataset, validation_dataset, epochs=epochs, batch_size=batch_size)


if __name__ == '__main__':
    train()
