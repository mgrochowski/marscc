import ipywidgets as widgets
import math
import pandas as pd
from IPython.core.display import display, Javascript
from utils.download import download_model
from predict_and_detect import predict_large_image, plot_predictions
from detect import detect_cones_and_craters, draw_regions3, match_detections, filter_matched_detections
from detect import detections_to_datatable, class_report, merge_heatmaps
from utils.image import image_to_labelmap, recognize_rgb_map, label_map
from keras_segmentation.predict import model_from_checkpoint_path
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
# from tqdm.auto import tqdm
import cv2
from time import sleep
import os
import uuid
from sklearn.metrics import classification_report, confusion_matrix
from IPython.utils.capture import capture_output

checkpoint_path = None
model_name = 'pspnet_50'
model_name = 'unet'

input_files = [
    'data/test/testing1.png',
    'data/test/testing2.png',
    'data/test/testing3.png'
]
mask_files =  [
    'data/test/testing1_mask.png',
    'data/test/testing2_mask.png',
    'data/test/testing3_mask.png'
]


def select_model():
    choose  = widgets.Select(
        options=['unet', 'vgg_unet', 'vgg_unet2', 'pspnet_50'],
        value=model_name,
        description='Model:',
        disabled=False
    )
    return choose

def create_output_tabs(tab_names):

    children = [ widgets.Output() for i in range(len(tab_names)) ]

    tab = widgets.Tab(children = children)
    for i, name in enumerate(tab_names):
        tab.set_title(i, name)

    return tab



def get_notebook_name():
    display(Javascript('IPython.notebook.kernel.execute("NotebookName = " + "\'"+window.document.getElementById("notebook_name").innerHTML+"\'");'))
    try:
        _ = type(NotebookName)
        return NotebookName
    except:
        return 'mars_detection_2'

def export_to_html():
    display(Javascript("IPython.notebook.save_notebook()"), include=['application/javascript'] )

    notebook_name =  get_notebook_name()
    date_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    output_name = notebook_name + '-' + str(date_time) + '.html'
    cmd = 'jupyter nbconvert --to html_embed --no-input --no-prompt ' + notebook_name + '.ipynb --output ' + output_name
    sleep(3)
    os.system(cmd)
    print('Run: ', cmd)
    print('Notebook exported to: ', output_name)


class Message:
    def __init__(self, message) -> None:
        self._message = message

    def _repr_markdown_(self):
        return self._message

    def __repr__(self) -> str:
        return self._message

def display_message(message):
    display(Message(message))



class App():
    def __init__(self,
                 merge_scales = ( 1.0/4.0, 1.0/8.0, 1.0/16.0),   # merge scales,
                 # resize_ratio = 1.0/4.0,         # main scale, if None then first of scales will be used
                 meters_per_pixel = 4.2319,      # original input image scale
                 cone_min_diameter_km = 0.8,      # 0.8 km
                 crater_min_diameter_km = 0.150,  # 150 meters
                 batch_size = 4,
                 iou_threshold = 0.2,
                 min_confidence = 0.4,             # if None then argmax() or thresholding_fn is used
                 overlap = 0.5,                    # fraction (or number of pixels) of image width that will be overlaped with sliding window
                 closing_diameter = 25,
                 threshold_fn = None,      # use scikit-image thresholding functions, e.g. threshold_isodata
                 cl_border = True,
                 model_name='pspnet_50',
                 input_files = input_files,
                 mask_files = mask_files):

        self.scales =                     merge_scales
        self.resize_ratio =               merge_scales[0]
        self.meters_per_pixel =           meters_per_pixel
        self.cone_min_diameter_km =       cone_min_diameter_km
        self.crater_min_diameter_km =     crater_min_diameter_km
        self.batch_size =                 batch_size
        self.iou_threshold =              iou_threshold
        self.min_confidence =             min_confidence
        self.overlap =                    overlap
        self.closing_diameter =           closing_diameter
        self.threshold_fn =               threshold_fn
        self.cl_border =                  cl_border
        self.model_name                 = model_name
        self.input_files = input_files
        self.mask_files = mask_files

        self.cone_min_diameter_px = int(self.resize_ratio *  self.cone_min_diameter_km * 1000.0 / self.meters_per_pixel)
        self.crater_min_diameter_px = int(self.resize_ratio *  self.crater_min_diameter_km * 1000.0 / self.meters_per_pixel)
        self.cone_min_area = 0.25 * self.cone_min_diameter_px * self.cone_min_diameter_px * math.pi
        self.crater_min_area = 0.25 * self.crater_min_diameter_px * self.crater_min_diameter_px * math.pi
        self.meters_per_px_scaled = self.meters_per_pixel / self.resize_ratio

        pd.set_option("display.max_rows", None, "display.max_columns", None)

        self.imgNorm = "sub_mean"
        if model_name == 'pspnet_50' or model_name == 'vgg_unet3':
            self.imgNorm = "sub_and_divide"

        self.checkpoint_path = download_model(target_dir='models', name=model_name)
        self.model = None

    def print_configuration(self):
        print('Configuration:')
        print('Model name             : %s' % self.model_name)
        print('Checkpoint             : %s' % self.checkpoint_path)
        print('Resize ratio           : %.5f' % self.resize_ratio )
        print('Scales                 : %s' % str(self.scales) )
        print('Meters per px          : %.5f' % self.meters_per_pixel)
        print('Meters per px scaled   : %.5f' % self.meters_per_px_scaled)
        print('Cone min. diameter     : %.2f km' % self.cone_min_diameter_km)
        # print('Cone min. diameter : %d px' % cone_min_diameter_px)
        # print('Cone min. area     : %.0f px^2' %  cone_min_area)
        print('Crater min. diameter   : %.2f km' % self.crater_min_diameter_km)
        # print('Crater min. diameter : %d px' % crater_min_diameter_px)
        # print('Crater min. area     : %.0f px^2' %  crater_min_area)
        print('IOU threshold          : %f' % self.iou_threshold)
        print('Min. confidence        : %s' % str(self.min_confidence))
        print('Batch size             : %d' % self.batch_size)
        print('Normalization          : %s' % self.imgNorm)
        print('Overlap                : %f' % self.overlap)
        print('Closing dia.           : %d' % self.closing_diameter)
        print('Threshold fn.          : %s' % None if self.threshold_fn is None else self.threshold_fn.__name__)
        print('Clear border           : %s' % str(self.cl_border))
        print('Input images           : %s' % self.input_files)
        print('Annotations            : %s' % self.mask_files)


    def run_detection(self, plot_segmentation=False, plot_detections=False, print_detections=False, save=False, input_files = None, save_dir=None):

        current_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        if self.model is None:
            self.model = model_from_checkpoint_path(self.checkpoint_path, input_width=None, input_height=None)
            print('Model input shape', self.model.input_shape)
        _, input_width, input_height, channels = self.model.input_shape

        if save:
            if save_dir is None:
                save_dir = 'results/detection_' + current_time
            Path(save_dir).mkdir(exist_ok=True, parents=True)

        if input_files is None:
            input_files = self.input_files

        self.results_summary = []
        for i, input_file in enumerate(input_files):
            file_name = os.path.basename(input_file)
            output_file_path = str(Path(save_dir + '/' + file_name).with_suffix(''))

            display_message('## File %d/%d: %s' % (i+1, len(input_files), file_name))

            heatmaps_to_merge = []
            for scale in self.scales:
                heatmap1, image1 = predict_large_image(input_file, resize_ratio=scale,
                                                    model =self.model,
                                                    output_type='heatmap',
                                                    imgNorm=self.imgNorm, batch_size=self.batch_size,
                                                    overlap=self.overlap)

                heatmaps_to_merge.append(heatmap1)
                if scale == self.scales[0]:
                    image = image1

            heatmap = merge_heatmaps(heatmaps_to_merge, method='max')

            # plot segmentation results
            if plot_segmentation:
                fig = plot_predictions(image, heatmaps=heatmap, texts=input_file, min_confidence=self.min_confidence)
                if save:
                    fig.savefig(output_file_path + '_segmentation.png')
                fig.clear()
                plt.close(fig)

            # object detection
            pred_results = detect_cones_and_craters(heatmap=heatmap, min_confidence=self.min_confidence,
                                                        threshold_fn=self.threshold_fn, closing_diameter=self.closing_diameter,
                                                        cl_border=self.cl_border)


            pred_dt = detections_to_datatable(pred_results)
            pred_dt['file_name'] = file_name
            pred_dt['diameter_km'] = pred_dt['equivalent_diameter'] * self.meters_per_px_scaled / 1e3
            pred_dt['org_centroid'] = pred_dt['centroid'].apply(lambda x: (x/self.resize_ratio).astype(int))

            if print_detections:
                display_message('### [%s]: Detected objects' % (file_name))
                display(pred_dt.groupby('label').agg({'diameter_km' : ['count', 'mean', 'std', 'min', 'max']}))
                # display(pred_dt)
            if save:
                pred_dt.to_csv(output_file_path + '_dt.csv')
                with open(output_file_path + '_dt.txt', 'w') as text_file:
                    text_file.write(pred_dt.to_string())

            # show detection results
            if plot_detections:
                image_reg = draw_regions3(image, pred_dt, thickness=2)
                fig = plt.figure(figsize=(10, 10))
                plt.imshow(image_reg)
                plt.show()
                if save:
                     fig.savefig(output_file_path + '_detection.png')
                fig.clear()
                plt.close(fig)

            results = {
                'file_name':  file_name,
                'file_path':  input_file,
                'heatmap':  heatmap,
                'image':    image,
                'pred_dt':  pred_dt
            }
            self.results_summary.append(results)


    def run_evaluation(self, plot_segmentation=True, plot_detections=True, print_detections=True, save=False, input_files = None, mask_files=None, save_dir=None):
        # run detection and evaluation

        # show_columns = ['true_label', 'pred_label', 'iou', 'true_id', 'pred_id', 'pred_diameter_km', 'true_diameter_km', 'pred_centroid', 'pred_confidence', 'message']

        current_time = datetime.now().strftime("%Y-%m-%d_%H%M%S")

        if save:
            if save_dir is None:
                save_dir = 'results/evaluation_' + current_time
            Path(save_dir).mkdir(exist_ok=True, parents=True)
            with capture_output() as c:
                self.print_configuration()
            with open(save_dir + '/configuration.txt', 'w') as text_file:
                text_file.write(c.stdout)

        if input_files is None:
            input_files = self.input_files

        if mask_files is None:
            mask_files = self.mask_files

        self.run_detection(plot_segmentation=False, plot_detections=False, save=save, save_dir=save_dir, print_detections=False, input_files=input_files)

        i=0
        for mask_file, results in zip(mask_files, self.results_summary):

            mask_file_name = os.path.basename(mask_file)
            results['mask_file_name'] = mask_file_name
            file_name = results['file_name']

            output_file_path = str(Path(save_dir + '/' + file_name).with_suffix(''))
            output_mask_file_path = str(Path(save_dir + '/' + mask_file_name).with_suffix(''))

            image = results['image']
        #     prediction = results['prediction']
            heatmap = results['heatmap']
            pred_dt = results['pred_dt']

            display_message('### File %d/%d: %s' % (i+1, len(self.results_summary), file_name))

            mask_img = cv2.imread(mask_file, 1)
            mask_img = cv2.resize(mask_img, (image.shape), interpolation=cv2.INTER_NEAREST)
            rgb_map = recognize_rgb_map(mask_img)
            mask_labels = image_to_labelmap(mask_img, rgb_map=rgb_map)

            true_results = detect_cones_and_craters(label_image=mask_labels)
            true_dt = detections_to_datatable(true_results)
            true_dt['file_name'] = file_name
            true_dt['diameter_km'] = true_dt['equivalent_diameter'] * self.meters_per_px_scaled / 1e3
            true_dt['org_centroid'] = true_dt['centroid'].apply(lambda x: (x/self.resize_ratio).astype(int))

            if print_detections:
                display_message('#### [%s] Targets' % (mask_file_name))
                display(true_dt.groupby('label').agg({'diameter_km' : ['count', 'mean', 'std', 'min', 'max']}))
            #     display(true_dt)

                display_message('#### [%s]: Detected objects' % (file_name))
                display(pred_dt.groupby('label').agg({'diameter_km' : ['count', 'mean', 'std', 'min', 'max']}))
            #     display(pred_dt)

            if save:
                true_dt.to_csv(output_mask_file_path + '_dt.csv')
                with open(output_mask_file_path + '_dt.txt', 'w') as text_file:
                    text_file.write(true_dt.to_string())

            if plot_segmentation:
                display_message('#### [%s] Targets vs. predictions' % (file_name))
                fig = plot_predictions(image, heatmaps=heatmap, targets=mask_img, min_confidence=self.min_confidence)
                plt.show()

                if save:
                    fig.savefig(output_file_path + '_segmentation.png')
                fig.clear()
                plt.close(fig)

            if plot_detections:
                fig, ax = plt.subplots(1, 2, figsize=(15,10))
                image_reg = draw_regions3(image, true_dt, thickness=2)
                ax[0].imshow(image_reg)
                ax[0].set_title('Target')
                image_reg = draw_regions3(image, pred_dt, thickness=2)
                ax[1].imshow(image_reg)
                ax[1].set_title('Prediction')
                plt.show()

                if save:
                     fig.savefig(output_file_path + '_detection.png')
                fig.clear()
                plt.close(fig)


        #     mdisplay('### [%s] Matched IOU > %f' % (file_name, iou_threshold), raw=True)
            matched_dt = match_detections(true_dt, pred_dt, iou_threshold=self.iou_threshold)
        #     display(matched_dt[show_columns])

        #     mdisplay('### [%s] Matched and filtered' % (file_name), raw=True)
        #     min_diameter={'cone': cone_min_diameter_px, 'crater': crater_min_diameter_px}
            min_diameter_km = {'cone': self.cone_min_diameter_km, 'crater': self.crater_min_diameter_km}
            filtered_dt = filter_matched_detections(matched_dt, min_diameter_km=min_diameter_km)
        #     display(filtered_dt[show_columns])

            true_labels_id = [label_map[label] for label in filtered_dt['true_label']]
            pred_labels_id = [label_map[label] for label in filtered_dt['pred_label']]

            conf_mat = confusion_matrix(true_labels_id, pred_labels_id, labels=range(len(label_map)))

            display_message('#### [%s] Classification report' % (file_name))
            cr_str = classification_report(true_labels_id, pred_labels_id, labels=range(len(label_map)), target_names=label_map.keys())
            print(cr_str)

            with capture_output() as c:
                class_report(conf_mat[[1,2,0], :][:, [1,2,0]], ['cone', 'crater', 'background'])
            print(c.stdout)

            if save:
                with open(output_file_path + '_class_report.txt', 'w') as text_file:
                    text_file.write(cr_str)
                with open(output_file_path + '_conf_mat.txt', 'w') as text_file:
                    text_file.write(c.stdout)

            filtered_dt['image_id'] = i
            results['true_dt'] = true_dt
            results['matched_dt'] = filtered_dt
            results['conf_mat'] = conf_mat
            results['mask_img'] = mask_img
            i = i + 1

        self.results_all_dt = pd.concat([r['matched_dt']  for r in self.results_summary])
        self.results_all_dt.reset_index(drop=True, inplace=True)

        true_labels_id = [label_map[label] for label in self.results_all_dt['true_label']]
        pred_labels_id = [label_map[label] for label in self.results_all_dt['pred_label']]

        self.conf_mat = confusion_matrix(true_labels_id, pred_labels_id, labels=range(len(label_map)))

        display_message('### [%s] Classification report ALL' % (file_name))
        cr_str = classification_report(true_labels_id, pred_labels_id, labels=range(len(label_map)), target_names=label_map.keys())
        print(cr_str)

        with capture_output() as c:
            class_report(conf_mat[[1,2,0], :][:, [1,2,0]], ['cone', 'crater', 'background'])
        print(c.stdout)

        self.results_all_dt['TP'] = (self.results_all_dt.true_label != 'background') & (self.results_all_dt.pred_label ==  self.results_all_dt.true_label)

        if save:
            self.results_all_dt.to_csv(save_dir + '/results_all_dt.csv')
            with open(save_dir + '/results_all_dt.txt', 'w') as text_file:
                text_file.write(self.results_all_dt.to_string())
            with open(save_dir + '/results_all_class_report.txt', 'w') as text_file:
                text_file.write(cr_str)
            with open(save_dir + '/results_all_conf_mat.txt', 'w') as text_file:
                text_file.write(c.stdout)