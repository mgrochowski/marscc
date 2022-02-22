from predict_and_detect import predict_large_image
from utils.image import image_to_labelmap
from detect import detection_raport
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

# fig, ax = plt.subplots(1, 3, figsize=(20,7))
# ax[0].imshow(image, cmap='gray')
# ax[0].set_title('Input image')
# ax[1].matshow(image_to_labelmap(mask_img), vmin=0)
# ax[1].set_title('Target')
# ax[2].matshow(prediction, vmin=0)
# ax[2].set_title('Predicted')
# plt.show()

# detection
from detect import detect_cones_and_craters, print_detections, draw_regions2
min_perimeter=70

results = detect_cones_and_craters(labels=prediction, min_area=10, min_perimeter=min_perimeter, min_solidity=0.5)

print_detections(results)

image_reg = draw_regions2(image, results, thickness=2)
plt.figure(figsize=(18, 18))
plt.imshow(image_reg)


# ground truth detection
from utils.image import image_to_labelmap

mask_labels = image_to_labelmap(mask_img)

mask_results = detect_cones_and_craters(labels=mask_labels, min_area=0, min_perimeter=min_perimeter, min_solidity=0)

print_detections(mask_results)

image_reg = draw_regions2(mask_img, mask_results, thickness=2)
plt.figure(figsize=(18, 18))
plt.imshow(image_reg)


summary = detection_raport(mask_results, results)

# print(summary)
print('No errors ', len(summary['errors']))
print('No missig ', len(summary['missing']))
print('No added ', len(summary['added']))

for e in summary['errors']:
    print('true: %s, pred: %s' % (e[0][1], e[1][1]))
    bbox1, bbox2 = e[0][0].bbox, e[1][0].bbox
    bbox = (min(bbox1[0],bbox2[0]), min(bbox1[1],bbox2[1]), max(bbox1[2],bbox2[2]), max(bbox1[3],bbox2[3]))

    fig, ax = plt.subplots(1, 3, figsize=(20,7))
    ax[0].imshow(image[bbox[0]:bbox[2],bbox[1]:bbox[3]], cmap='gray', extent=bbox )
    ax[0].set_title('Input image')
    ax[1].matshow(image_to_labelmap(mask_img[bbox[0]:bbox[2],bbox[1]:bbox[3]]), vmin=0, extent=bbox )
    ax[1].set_title('Target')
    ax[2].matshow(prediction[bbox[0]:bbox[2],bbox[1]:bbox[3]], vmin=0, extent=bbox )
    ax[2].set_title('Predicted')
    plt.show()

