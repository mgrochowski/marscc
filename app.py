import ipywidgets as widgets
import math
import pandas as pd
from IPython.core.display import display
from utils.download import download_model

checkpoint_path = None
model_name = 'pspnet_50'

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


from time import sleep
from IPython.display import display, Javascript
import os
import uuid
from datetime import datetime

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