import os, sys
from threading import Thread, Timer
from bokeh.layouts import column, row
from bokeh.models import Button
from bokeh.plotting import curdoc, figure
from bokeh.models.widgets import Div
from functools import partial
from PIL import Image
from pathlib import Path


try:
    import datashader
except ImportError:
    datashader = None
    print("\n\nThe datashader package needs to be installed from source to use the GUI:\n"
          "$ pip install git+ssh://git@github.com/bokeh/datashader.git@0.6.5#egg=datashader-0.6.5\n\n")
if datashader is None:
    sys.exit(1)


from cobras_ts.querier.visualquerier_images import VisualImageQuerier
from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.cli.cli import create_parser, prepare_data, prepare_clusterer
import cobras_ts.cli.image_to_feature_vec as image_to_feature_vec

import random
import numpy as np
import sys
import shutil


def communicate_query_result(query_result):
    global querier
    querier.query_result = query_result
    querier.query_answered = True

curdoc().title = "COBRAS"

loading = Div(text="""<h3>Loading...<h3>""", width=100, height=100)

def mustlink_callback():
    global query_answered
    global querier
    global layout
    global button_ml
    global button_cl

    #button_ml.disabled = True
    #button_cl.disabled = True

    layout.children[2] = loading

    t = Timer(0.1, partial(communicate_query_result, query_result=True))
    t.start()
    #querier.query_answered = True
    #querier.query_result = True

def cannotlink_callback():
    global query_answered
    global querier
    global layout



    layout.children[2] = loading

    #button_ml.disabled = True
    #button_cl.disabled = True

    t = Timer(0.1, partial(communicate_query_result, query_result=False))
    t.start()

    #querier.query_answered = True
    #querier.query_result = False

button_ml = Button(label="Yes (must-link)", button_type="success")
button_ml.on_click(mustlink_callback)

button_cl = Button(label="No (cannot-link)", button_type="warning")
button_cl.on_click(cannotlink_callback)


random.seed(123)
np.random.seed(123)

query_answered = False

sys.argv = sys.argv[1].split(' ')
parser = create_parser()
args = parser.parse_args(None)


fns, features = image_to_feature_vec.convert_img_to_feature_vec(vars(args)['inputs'][0])


if not os.path.exists(str(Path(__file__).parent) + '/static/to_cluster/'):
    os.makedirs(str(Path(__file__).parent) + '/static/to_cluster/')

for fn in fns:
    print("copying file " + fn)
    shutil.copyfile(fn, str(Path(__file__).parent) + '/static/to_cluster/' + fn.split('/')[-1])

doc = curdoc()


first_image = Image.open(fns[0]).convert('RGBA')
xdim, ydim = first_image.size

img = np.empty((ydim, xdim), dtype=np.uint32)
view = img.view(dtype=np.uint8).reshape((ydim, xdim, 4))
# Copy the RGBA image into view, flipping it so it comes right-side up
# with a lower-left origin
view[:,:,:] = np.flipud(np.asarray(first_image))



#all_data_plot = figure(plot_width=400, plot_height=180, title="A single sample",toolbar_location='above')
#all_data_plot.image_rgba(image=[view], x=0, y=0, dw=xdim, dh=ydim)




topdiv = Div(text="<h1> COBRAS <br>  iteration:  1 <br> # queries answered: 0 </h1>", css_classes=['top_title_div'],
width=500, height=100)
div = Div(text="<h2> The full dataset </h2>", css_classes=['title_div'],
width=200, height=100)
div2 = Div(text="<h2> Should these two instances be in the same cluster? </h2>", css_classes=['title_div'],
width=500, height=20, name='wopwopwop')

div3 = Div(text="<h2> The (intermediate) clustering </h2>", css_classes=['title_div'],
width=500, height=100)
div4 = Div(text="", width=500, height=100)


ts1 = figure(x_axis_type="datetime", plot_width=250, plot_height=120, toolbar_location=None) # placeholders
ts2 = figure(x_axis_type="datetime", plot_width=250, plot_height=120, toolbar_location=None)


q1 = Div(text="<img width=150 height=150  src='webapp_images/static/to_cluster/" + fns[2].split('/')[-1] + "'>")
q2 = Div(text="<img width=150 height=150 src='webapp_images/static/to_cluster/" + fns[3].split('/')[-1] + "'>")


empty_div = Div(text="")


#layout = column(row(topdiv), row(column(div2,row(q1,q2),column(button_ml, button_cl))), div3, div4, empty_div,sizing_mode="scale_width")

layout = column(topdiv,div2,row(q1,q2),column(button_ml,button_cl))

print("\n\n\n\n\n\n")
print(layout.children)




curdoc().add_root(layout)




querier = VisualImageQuerier(features, curdoc(), layout, fns, button_ml, button_cl)
clusterer = COBRAS_kmeans(features, querier, 100000)
#clusterer_args = vars(args)
#clusterer = prepare_clusterer(data=series, querier=querier, **clusterer_args)



def blocking_task():
    #time.sleep(100000)
    clusterer.cluster()

thread = Thread(target=blocking_task)
thread.start()


