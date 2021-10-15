from pathlib import Path
from urllib import request
from urllib.parse import urlparse
import urllib, zipfile

MARS_IMG_URL='https://www.fizyka.umk.pl/~grochu/mars/mars_images_scale_1_500K.zip'
MARS_TRAIN_URL='https://www.fizyka.umk.pl/~grochu/mars/mars_data_20210923.zip'
VGG_UNET_URL='https://www.fizyka.umk.pl/~grochu/mars/models/vgg_unet_2021-09-29_132526.566811.zip'

MODELS_DIR='models'
DATA_DIR='data'

MARS_IMG_TARGET='mars_images_scale_1_500K'
MARS_TRAIN_TARGET='mars_data_20210923'
VGG_UNET_TARGET='vgg_unet_2021-09-29_132526.566811'


from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_and_unzip(url, target_dir=DATA_DIR, delete_source=True):

    t_dir = Path(target_dir)
    t_dir.mkdir(exist_ok=True, parents=True)
    file_name = Path(urlparse(url).path).name
    zip_file = t_dir / file_name

    print('Downloading: ', url)
    print('Target file: ', zip_file)
    try:
        download_url(url, str(zip_file))
    except urllib.error.URLError as e:
        ResponseData = e.reason

    if zip_file.exists():
        print('Extracting:  ', zip_file)
        zf = zipfile.ZipFile(zip_file, "r")
        zf.extractall(path=target_dir)
        zf.close()
        if delete_source:
            print('Deleting:     %s' % str(zip_file))
            zip_file.unlink(missing_ok=False)
    else:
        print('Download ERROR')


def download_training_data(target_dir=DATA_DIR):

    training_data_dir = Path(target_dir + '/' + MARS_TRAIN_TARGET)
    if training_data_dir.exists():
        print('Target directory exists: %s' % str(training_data_dir))
    else:
        download_and_unzip(MARS_TRAIN_URL, target_dir)


def download_vgg_unet_checkpoint(target_dir=MODELS_DIR):
    model_dir = Path(target_dir + '/' + VGG_UNET_TARGET)
    if model_dir.exists():
        print('Model checkpoint directory exists: %s' % str(model_dir))
    else:
        download_and_unzip(VGG_UNET_URL, target_dir)


def download_original_images(target_dir=DATA_DIR):

    img_dir = Path(target_dir + '/' + MARS_IMG_TARGET)
    if img_dir.exists():
        print('Target directory exists: %s' % str(img_dir))
    else:
        download_and_unzip(MARS_IMG_URL, target_dir)


if __name__ == '__main__':

    # download_original_images()
    download_training_data()
    download_vgg_unet_checkpoint()
