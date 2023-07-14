import ipywidgets as widgets
from IPython.core.display import display

checkpoint_path = None
model_name = 'pspnet_50'

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
