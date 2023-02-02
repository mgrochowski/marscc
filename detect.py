# Detect cones and craters on (original) annotation image

from __future__ import print_function, division, absolute_import, unicode_literals

from pathlib import Path

import click
import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from skimage.color import gray2rgb
from skimage.measure import regionprops
from skimage.morphology import closing, square
from skimage.segmentation import clear_border
from skimage.measure import label as label_region

from utils.image import image_to_labelmap, label_map, recognize_rgb_map

label_names = ('cone', 'crater')


@click.command()
@click.option('--input_file', default=None, help='Input file with annotations (labels)')
@click.option('--input_image', default=None, help='Input image with Mars surface')
@click.option('--min_area', default=10, help='Minimum object area [in ptx]')
@click.option('--min_solidity', default=0.5, help='Minimum object solidity')
@click.option('--output_dir', default='detect_cones_output', help='Output directory')
def run(input_file, input_image=None, min_area=10, min_solidity=0.5, output_dir='detect_cones_output'):

    image = cv2.imread(input_file)
    if image is None:
        raise Exception('Cant open file %s' % input_file)

    h, w, _ = image.shape
    # plt.imshow(image)
    # plt.show()

    rgb_map = recognize_rgb_map(image)
    labels = image_to_labelmap(image, rgb_map=rgb_map)
    # plt.imshow(label)

    detections = detect_cones_and_craters(labels, min_area=min_area, min_solidity=min_solidity)
    table = detections_to_datatable(detections)
    log = print_detections(table)

    o_dir = Path(output_dir)
    o_dir.mkdir(parents=True, exist_ok=True)

    i_name = Path(input_file).stem
    file_path = str(o_dir / i_name) + '_regions.log'
    with open(file_path, 'w') as f:
        f.write(log)

    table.to_csv(str(o_dir / i_name) + '_dt.csv', index=False)

    i_name = Path(input_file).stem
    labels_reg = draw_regions2(image, detections)
    file_path = str(o_dir / i_name) + '_lab_regions.png'
    cv2.imwrite(file_path, labels_reg)

    if input_image is not None:
        inp_img = cv2.imread(input_image)

        if inp_img is None:
            print('Warning: cant open file %s' % input_image)
        else:
            inp_img_reg = draw_regions2(inp_img, detections)
            i_name = Path(input_image).stem
            file_path = str(o_dir / i_name) + '_img_regions.png'
            cv2.imwrite(file_path, inp_img_reg)

    print('Results saved in %s' % output_dir)


def detect_cones_and_craters(label_image=None, heatmap=None, min_area=0, min_solidity=0.0, label_names=label_names,
                             min_significance=None):
    # if min_significance == None then
    #      label_image = heatmap.argmax(axis=2)
    # else
    #     label_image = heatmap[:,:, label] >= min_significance  * label_id

    if label_image is None:
        if heatmap is None:
            raise Exception('Provide label_image or heatmap argument')
        else:
            label_image = heatmap.argmax(axis=2)
    else:
        if heatmap is None:
            heatmap = np.ones((label_image.shape[0], label_image.shape[1], len(label_map)), dtype=np.float)
    detected = {}

    # detect
    for label in label_names:
        label_id = label_map[label]
        if min_significance is None:
            label_image_input = label_image
        else:
            label_image_input = np.zeros(heatmap[:, :, label_id].shape)
            label_image_input[heatmap[:, :, label_id] >= min_significance] = label_id
        label_image_det = detection(label_image_input, label=label_id)
        res = regionprops(label_image_det, intensity_image=heatmap[:, :, label_id])

        detected[label] = res
        # print('%s detected %d objects' % (label, len(res)))

    # filter
    if min_area > 0 or min_solidity > 0.0:
        for label in label_names:

            res = detected[label]
            res_filtered = []
            for region in res:

                # take regions with large enough areas
                if region.area >= min_area and region.solidity >= min_solidity:
                    res_filtered.append(region)
                else:
                    reasons = []
                    if region.area < min_area: reasons.append('area %.1f < %.1f' % (region.area, min_area))
                    if region.solidity < min_solidity: reasons.append('solidity %.1f < %.1f' % (region.solidity, min_solidity))

                    print('Ignoring region %d [%s] at %.0f,%.0f, approx. diameter %.1f, %s ' %
                          (region.label, label, region.centroid[0], region.centroid[1], 2.0 * np.sqrt(region.area / np.pi), ", ".join(reasons)))

            detected[label] = res_filtered

    return detected


# obsolete
def draw_regions(image, detected, color=None):

    if color is None:
        color = ['red', 'green']

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


def draw_regions3(image, detection_table, color=None, thickness=3):

    if color is None:
        color = [[255, 0, 0], [0, 0, 255], [0, 255, 0]]

    img_reg = image.copy()
    if len(img_reg.shape) == 2:
        img_reg = gray2rgb(img_reg)

    # alpha = 0.5

    for label, bbox in zip(detection_table.label, detection_table.bbox):
        # draw rectangle around segmented objects
        minr, minc, maxr, maxc = bbox
        # rr, cc = rectangle_perimeter(start=(minr, minc), end=(maxr, maxc), shape=img_reg.shape)
        # set_color(img_reg, (rr, cc), color[i], alpha=alpha)
        cv2.rectangle(img_reg, (minc, minr), (maxc, maxr), color=color[label_map[label]], thickness=thickness)

    return img_reg




def print_detections(detected):

    table = detected
    if isinstance(detected, dict):
        table = detections_to_datatable(detected, sort_by='area')

    if not table.empty:
        text = table.groupby('label').agg({'area': ['count', 'mean', 'std', 'min', 'max']}).to_string()
        text += '\n\n' + table.to_string()
    else:
        text = 'No objects detected'

    print(text)
    return text


def detections_to_datatable(detections, sort_by='area'):

    dt = pd.DataFrame(columns=['label', 'area', 'bbox', 'centroid', 'solidity', 'confidence', 'feret_diameter_max',
                               'equivalent_diameter'])
    i = 0
    for label in detections:
        for region in detections[label]:
            dt.loc[i] = [label, float(region.area), np.array(region.bbox), np.array(region.centroid),
                         float(region.solidity), float(region.max_intensity), float(region.feret_diameter_max),
                         float(region.equivalent_diameter)]
            i = i + 1

    # dt['diameter'] = 2.0 * np.sqrt(dt['area'] / np.pi)
    return dt.sort_values(by=[sort_by], ascending=False)


def detection(x, label=1):
    label_img = (x == label).astype(np.int32)

    # apply threshold
    bw = closing(label_img, square(3))

    # remove artifacts connected to image border
    cleared = clear_border(bw)

    # label image regions
    label_image = label_region(cleared, background=0)
    return label_image


def class_report(cm, target_names=None):

    print("Confusion matrix")
    print("true \\ predicted")

    print(*[("%10s  " % target_names[i]) + str(row)[1:-1] for i, row in enumerate(cm)], sep='\n')

    print("\n     label    precision  recall  f1-score   support")
    for i in range(cm.shape[0]):
        n = cm[i, :].sum()
        if n > 0:
            rec = cm[i, i] / n
        else:
            rec = 0.0
        prec = cm[i, i] / cm[:, i].sum()
        if prec + rec > 0:
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0.0
        print("%10s %10.2f %10.2f %10.2f %3d" % (target_names[i], prec, rec, f1, n))


def iou_bbox(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : cords [x1, y1, x2, y2]
    bb2 : cords [x1, y1, x2, y2]

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


def detection_report(true_regions, predicted_regions, iou_threshold=0.5):

    label_names = []
    label_idx = {}

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
    label_names.append('background')

    # count matched and conf-mat
    matched_target = np.zeros((len(target_reg_list, )))
    matched_pred = np.zeros((len(predicted_reg_list, )))

    conff = np.zeros((len(label_names), len(label_names)), dtype=np.int)

    summary = {'errors': [], 'missing': [], 'added': [], 'miss_iou': [], 'confusion_matrix': None}
    # y_true, y_pred  = [], []
    for i, tr in enumerate(target_reg_list):
        for j, sr in enumerate(predicted_reg_list):
            # in_ptx = intersection_points(tr[0].cords, sr[0].cords, threshold=len(tr[0].cords))
            # ptx_iou = float(len(in_ptx)) / (len(tr[0].cords) + len(sr[0].cords) - len(in_ptx))
            iou = iou_bbox(tr[0].bbox, sr[0].bbox)
            if iou > iou_threshold:
                matched_target[i] = matched_target[i] + 1
                matched_pred[j] = matched_pred[j] + 1
                # print(i, j, len(in_ptx), tr[1], sr[1])
                conff[label_idx[tr[1]], label_idx[sr[1]]] = conff[label_idx[tr[1]], label_idx[sr[1]]] + 1
                if tr[1] != sr[1]:
                    summary['errors'].append((tr, sr))
            if iou > 0 and iou <= iou_threshold:
                summary['miss_iou'].append((tr, sr, iou))

    # not detected target ROIs, mark as background predictions
    for tr, mt in zip(target_reg_list, matched_target):
        if mt == 0:
            conff[label_idx[tr[1]], bcg_idx] = conff[label_idx[tr[1]], bcg_idx] + 1
            summary['missing'].append(tr)

    # prediction of regions not matched to target ROIs
    for pr, mp in zip(predicted_reg_list, matched_pred):
        if mp == 0:
            conff[bcg_idx, label_idx[pr[1]]] = conff[bcg_idx, label_idx[pr[1]]] + 1
            summary['added'].append(pr)

    class_report(conff, label_names)
    summary['confusion_matrix'] = conff

    return summary


def intersection_points(x, y, threshold=1):

    points = []

    for ex in x:
        if np.any(np.all(ex == y, axis=1)):
            points.append(ex)
        if len(points) >= threshold:
            break

    return points


def match_detections(true_regions, predicted_regions):

    input_columns = [ 'label', 'bbox', 'centroid', 'confidence', 'file_name', 'diameter_km' ]
    n_columns = len(input_columns)
    t_regions = true_regions[input_columns]
    p_regions = predicted_regions[input_columns]
    columns = [ 'pred_' + c for c in input_columns] +  [ 'pred_id'] + [ 'true_' + c for c in input_columns]  + ['true_id', 'iou', 'message']
    matched = pd.DataFrame(columns=columns)

    matched_target = np.zeros((len(t_regions, )))
    matched_pred = np.zeros((len(p_regions, )))

    k = 0
    for i in range(len(t_regions)):
        true_id = t_regions.index[i]
        bbt  = t_regions.bbox[i]
        # find region with best match
        for j in range(len(p_regions)):
            pred_id = p_regions.index[j]
            bbp  = p_regions.bbox[j]
            iou = iou_bbox(bbt, bbp)
            if iou > 0.0:
                # print(i, j, bbt, bbp, iou)
                msg = ''
                if t_regions.label[i] == p_regions.label[j]:
                    msg = 'Correct: %s, ' % t_regions.label[i]
                else:
                    msg = 'Error, '
                matched_target[i] = matched_target[i] + 1
                matched_pred[j] = matched_pred[j] + 1
                matched.loc[k] = p_regions.loc[j].to_list()  + [pred_id] +  t_regions.loc[i].to_list() + [ true_id, iou, msg]
                k = k + 1

    missing_object = [ 'background' ] + [np.NAN] * (n_columns)
    for i in np.where(matched_target == 0)[0]:
        matched.loc[k] = missing_object  +  t_regions.loc[i].to_list() + [ t_regions.index[i], np.NaN, 'Missing detection, ' ]
        k = k + 1

    for j in np.where(matched_pred == 0)[0]:
        matched.loc[k] = p_regions.loc[j].to_list()  +  [p_regions.index[j]] + missing_object + [ np.NaN,  'False detection, ' ]
        k = k + 1

    return matched


def filter_matched_detections(matched_dt, min_diameter=0, iou_threshold=0.5):

    if isinstance(min_diameter, dict):
        min_dia = min_diameter
    else:
        min_dia ={ label:min_diameter for label in label_names}

    result = matched_dt.copy()

    # filter small objects
    for label in min_dia:
        too_small = (result['pred_equivalent_diameter'] < min_dia[label] ) & (result['pred_label'] == label)
        result.loc[too_small, 'pred_label']  = 'background'
        result.loc[too_small, 'message'] = result.loc[too_small, 'message'] + 'Predicted %s to small, ' % label

        too_small = (result['true_equivalent_diameter'] < min_dia[label]) & (result['true_label'] == label)
        result.loc[too_small, 'true_label']  = 'background'
        result.loc[too_small, 'message'] = result.loc[too_small, 'message'] + 'Target %s to small, ' % label

    # filter to small iou
    too_small = result['iou'] < iou_threshold
    result.loc[too_small, 'true_label']  = 'background'
    result.loc[too_small, 'message'] = result.loc[too_small, 'message'] + 'IOU < %.3f, ' % iou_threshold

    return result


if __name__ == '__main__':

    run()
