from predict_and_detect import predict_large_image
from utils.image import image_to_labelmap
import matplotlib.pyplot as plt
import cv2

# Large MARS image in 1:500K scale
input_file = 'data/test/scale_500K/unnamed_testing_1.png'
resize_ratio = 0.1

checkpoint_path = 'models/unet_mini_2021-09-27_003743.848447/unet_mini'

prediction, image = predict_large_image(input_file, resize_ratio=resize_ratio, checkpoint_path=checkpoint_path)

mask_file = 'data/test/scale_500K/unnamed_testing_1_mask.png'
mask_img = cv2.imread(mask_file, 1)
mask_img = cv2.resize(mask_img, (image.shape), interpolation=cv2.INTER_NEAREST)

fig, ax = plt.subplots(1, 3, figsize=(20,7))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Input image')
ax[1].matshow(image_to_labelmap(mask_img), vmin=0)
ax[1].set_title('Target')
ax[2].matshow(prediction, vmin=0)
ax[2].set_title('Predicted')
plt.show()