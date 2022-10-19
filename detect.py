# Detect cones and craters on (original) annotation image

from __future__ import print_function, division, absolute_import, unicode_literals

from pathlib import Path

import click
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from skimage.color import gray2rgb
from skimage.measure import regionprops
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.measure import label as label_region

from utils.image import image_to_labelmap, label_map


@click.command()
@click.option('--input_file', default=None, help='Input file with annotations (labels)')
@click.option('--input_image', default=None, help='Input image with Mars surface')
@click.option('--min_area', default=10, help='Minimum object area [in ptx]')
@click.option('--min_perimeter', default=5, help='Minimum object perimeter [in ptx]')
@click.option('--min_solidity', default=0.5, help='Minimum object solidity')
@click.option('--output_dir', default='detect_cones_output', help='Output directory')
def run(input_file, input_image=None, min_area=10, min_perimeter=5, min_solidity=0.5, output_dir='detect_cones_output'):

    image = cv2.imread(input_file)
    if image is None:
        raise Exception('Cant open file %s' % input_file)

    h, w, _ = image.shape
    # plt.imshow(image)
    # plt.show()

    labels = image_to_labelmap(image)
    # plt.imshow(label)

    results = detect_cones_and_craters(labels, min_area=min_area, min_perimeter=min_perimeter, min_solidity=min_solidity)
    log = print_detections(results)

    o_dir = Path(output_dir)
    o_dir.mkdir(parents=True, exist_ok=True)

    i_name = Path(input_file).stem
    file_path = str(o_dir / i_name) + '_regions.log'
    with open(file_path, 'w') as f:
        f.write(log)

    i_name = Path(input_file).stem
    labels_reg = draw_regions2(image, results)
    file_path = str(o_dir / i_name) + '_lab_regions.png'
    cv2.imwrite(file_path, labels_reg)

    if input_image is not None:
        inp_img = cv2.imread(input_image)

        if inp_img is None:
            print('Warrning: cant open file %s' % input_image)
        else:
            inp_img_reg = draw_regions2(inp_img, results)
            i_name = Path(input_image).stem
            file_path = str(o_dir / i_name) + '_img_regions.png'
            cv2.imwrite(file_path, inp_img_reg)

    print('Results saved in %s' % output_dir)


def detect_cones_and_craters(labels, min_area=10, min_perimeter=5, min_solidity=0.5):

    label_names = ['cone', 'crater']
    detected = {}

    # detect
    for label in label_names:
        label_image = detectoin(labels, label=label_map[label])
        res = regionprops(label_image)

        detected[label] = res
        print('%s detected %d objects' % (label, len(res)))

    # filter
    for label in label_names:

        res = detected[label]
        res_filtered = []
        for region in res:

            # take regions with large enough areas
            if region.area >= min_area and region.perimeter >= min_perimeter and region.solidity >= min_solidity:
                res_filtered.append(region)
            else:
                print('Removing region, area %d, perimeter %f, solidity %f' % (region.area, region.perimeter, region.solidity))

        detected[label] = res_filtered

    return detected


# obsolete
def draw_regions(image, detected, color=None):

    if color is None:
        color = ['red', 'green']

    # # plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)

    for i, label in enumerate(['cone', 'crater']):
        res = detected[label]
        for region in res:
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor=color[i], linewidth=2)
            ax.add_patch(rect)

    plt.savefig('output.png')


def draw_regions2(image, detected, color=None, thickness=3):

    if color is None:
        color = [[0, 0, 255], [0, 255, 0]]

    img_reg = image.copy()
    if len(img_reg.shape) == 2:
        img_reg = gray2rgb(img_reg)

    # alpha = 0.5

    for i, label in enumerate(['cone', 'crater']):
        res = detected[label]
        for region in res:
            # draw rectangle around segmented objects
            minr, minc, maxr, maxc = region.bbox
            # rr, cc = rectangle_perimeter(start=(minr, minc), end=(maxr, maxc), shape=img_reg.shape)
            # set_color(img_reg, (rr, cc), color[i], alpha=alpha)
            cv2.rectangle(img_reg, (minc, minr), (maxc, maxr), color=color[i], thickness=thickness)

    return img_reg


def print_detections(detected):

    text = ''
    # print (sorted by perimeter)
    for i, label in enumerate(['cone', 'crater']):
        res = detected[label]
        text += '\n%s N=%d\n\n' % (label, len(res))
        text += 'Region      area                   bbox          centroid        perimeter  solidity\n'

        for region in sorted(res, key=lambda x: getattr(x, 'perimeter')):
            text += '%5d  %9d  %25s  %8.1f %8.1f  %8.1f  %5.3f\n' % (
                region.label, region.area, str(region.bbox), region.centroid[0], region.centroid[1], region.perimeter,
                region.solidity)
    print(text)
    return text


def detectoin(x, label=1):
    label_img = (x == label).astype(np.int32)

    # apply threshold
    bw = closing(label_img, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label_region(cleared, background=0)
    return label_image


def class_report(cm, target_names=None):

    print("Confussion matrix")
    print("true \ predicted")

    print(*[ ("%10s  " % target_names[i]) + str(row)[1:-1] for i, row in enumerate(cm)], sep='\n')

    print("\n     label    precision  recall  f1-score   support")
    for i in range(cm.shape[0]):
        n = cm[i,:].sum()
        if n > 0:
            rec = cm[i,i] / n
        else:
            rec = 0.0
        prec = cm[i,i] / cm[:,i].sum()
        if prec + rec > 0:
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0.0
        print("%10s %10.2f %10.2f %10.2f %3d" % (target_names[i], prec, rec, f1, n ))

import numpy as np

def iou_bbox(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : coords [x1, y1, x2, y2]
    bb2 : coords [x1, y1, x2, y2]

    Returns
    -------
    float  in [0, 1]
    """
    assert bb1[0] < bb1[2]
    assert bb1[1] < bb1[3]
    assert bb2[0] < bb2[2]
    assert bb2[1] < bb2[3]

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1[0], bb2[0])
    y_top = max(bb1[1], bb2[1])
    x_right = min(bb1[2], bb2[2])
    y_bottom = min(bb1[3], bb2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
    bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def detection_report(true_regions, predicted_regions, iou_treshold=0.5):

    label_names = []
    label_idx = { }

    predicted_reg_list = []
    target_reg_list = []

    for i, label in enumerate(true_regions):

        assert label in predicted_regions

        label_names.append(label)
        label_idx[label] = i

        for r in predicted_regions[label]:
            predicted_reg_list.append((r, label))
        for r in true_regions[label]:
            target_reg_list.append((r, label))

    bcg_idx = len(label_names)   # background index
    label_names.append('backgroubd')

    # count mached and conf-mat
    mached_target = np.zeros((len(target_reg_list, )))
    mached_pred = np.zeros((len(predicted_reg_list, )))

    conff = np.zeros((len(label_names), len(label_names)), dtype=np.int)

    summary = { 'errors': [], 'missing': [], 'added': [] , 'confusion_matrix': None }
    # y_true, y_pred  = [], []
    for i, tr in enumerate(target_reg_list):
        for j, sr in enumerate(predicted_reg_list):
            # in_ptx = interesction_points(tr[0].coords, sr[0].coords)
            iou = iou_bbox(tr[0].bbox, sr[0].bbox)
            if iou > iou_treshold:
                mached_target[i] = mached_target[i] + 1
                mached_pred[j] = mached_pred[j] + 1
                # print(i, j, len(in_ptx), tr[1], sr[1])
                conff[label_idx[tr[1]], label_idx[sr[1]]] = conff[label_idx[tr[1]], label_idx[sr[1]]] + 1
                if tr[1] != sr[1]:
                    summary['errors'].append((tr, sr))
                # y_true.append(label_idx[tr[1]])
                # y_pred.append(label_idx[sr[1]])

    # print(mached_target)
    # print(mached_pred)

    # not detected target ROIs mark ad background predictions
    for tr, mt in zip(target_reg_list, mached_target):
        if mt == 0:
            # y_true.append(label_idx[tr[1]])
            # y_pred.append(bcg_idx)
            conff[label_idx[tr[1]], bcg_idx] = conff[label_idx[tr[1]], bcg_idx] + 1
            summary['missing'].append(tr)

    # prediction of regions not mached to target ROIs
    for pr, mp in zip(predicted_reg_list, mached_pred):
        if mp == 0:
            # y_true.append(bcg_idx)
            # y_pred.append(label_idx[pr[1]])
            conff[bcg_idx, label_idx[pr[1]]] = conff[bcg_idx, label_idx[pr[1]]] + 1
            summary['added'].append(pr)

    class_report(conff, label_names)
    summary['confusion_matrix'] = conff

    return summary

    from sklearn.metrics import  confusion_matrix, classification_report
    # cm = confusion_matrix(y_true, y_pred)
    # print(cm)
    # print(classification_report(y_true, y_pred, target_names=label_names))

def interesction_points(x, y, treshold=1):

    points = []

    for ex in x:
        if np.any(np.all(ex == y, axis=1)):
            points.append(ex)
        if len(points) >= treshold: break

    return points


if __name__ == '__main__':

    run()
