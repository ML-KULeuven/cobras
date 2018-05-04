import os, sys
from threading import Thread, Timer
from bokeh.layouts import column, row
from bokeh.models import Button
from bokeh.plotting import curdoc, figure
from bokeh.models.widgets import Div
from functools import partial

try:
    import datashader
except ImportError:
    datashader = None
    print("\n\nThe datashader package needs to be installed from source to use the GUI:\n"
          "$ pip install git+ssh://git@github.com/bokeh/datashader.git@0.6.5#egg=datashader-0.6.5\n\n")
if datashader is None:
    sys.exit(1)

try:
    from cobras_ts.visualquerier import VisualQuerier
    from cobras_ts.cobras_kshape import COBRAS_kShape
except ImportError:
    sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir))
    from cobras_ts.visualquerier import VisualQuerier
    from cobras_ts.cobras_kshape import COBRAS_kShape

import random
import numpy as np
import pandas as pd
import sys


def communicate_query_result(query_result):
    global querier
    querier.query_result = query_result
    querier.query_answered = True

curdoc().title = "COBRAS-TS"

loading = Div(text="""<h3>Loading...<h3>""", width=100, height=100)

def mustlink_callback():
    global query_answered
    global querier
    global layout
    global button_ml
    global button_cl

    button_ml.disabled = True
    button_cl.disabled = True

    layout.children[1].children[1].children[1] = loading

    t = Timer(0.1, partial(communicate_query_result, query_result=True))
    t.start()
    #querier.query_answered = True
    #querier.query_result = True

def cannotlink_callback():
    global query_answered
    global querier
    global layout
    layout.children[1].children[1].children[1] = loading

    button_ml.disabled = True
    button_cl.disabled = True

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

fn = sys.argv[1]
doc = curdoc()

# TODO: We should reuse cli.prepare_data() here, it is now hardcoded for one case
df = pd.read_csv(fn,header=None)

labels = df.ix[:,0]
df = df.drop(0,axis=1)
ts = df.ix[0,:]

data = df.as_matrix()

# reformat the data into an appropriate DataFrame
dfs = []
split = pd.DataFrame({'x': [np.nan]})
for i in range(len(data)):
    x = list(range(len(ts)))
    y = data[i]
    df3 = pd.DataFrame({'x': x, 'y': y})
    dfs.append(df3)
    dfs.append(split)
df2 = pd.concat(dfs, ignore_index=True)

x_range = 0, data.shape[1]
y_range = data[1:].min(), data[1:].max()

all_data_plot = figure(plot_width=400, plot_height=180, x_range=x_range, y_range=y_range, title="Full dataset",toolbar_location='above')
p = figure(plot_width=400, plot_height=180, x_range=x_range, y_range=y_range, title="Full dataset",toolbar_location='above')
canvas = datashader.Canvas(x_range=x_range, y_range=y_range,
                           plot_height=180, plot_width=400)
agg = canvas.line(df2, 'x', 'y', datashader.count())
img = datashader.transfer_functions.shade(agg, how='eq_hist')
all_data_plot.image_rgba(image=[img.data], x=x_range[0], y=y_range[0], dw=x_range[1] - x_range[0], dh=y_range[1] - y_range[0])
p.image_rgba(image=[img.data], x=x_range[0], y=y_range[0], dw=x_range[1] - x_range[0], dh=y_range[1] - y_range[0])
initial_temp_clustering = row(p)


topdiv = Div(text="<h1> COBRAS<sup>TS</sup> <br>  iteration:  1 <br> # queries answered: 0 </h1>", css_classes=['top_title_div'],
width=500, height=100)
div = Div(text="<h2> The full dataset </h2>", css_classes=['title_div'],
width=200, height=100)
div2 = Div(text="<h2> Should these two instances be in the same cluster? </h2>", css_classes=['title_div'],
width=500, height=100)
div3 = Div(text="<h2> The (intermediate) clustering </h2>", css_classes=['title_div'],
width=400, height=100)
div4 = Div(text="", css_classes=['title_div'],width=400, height=100)


ts1 = figure(x_axis_type="datetime", plot_width=250, plot_height=120, toolbar_location=None) # placeholders
ts2 = figure(x_axis_type="datetime", plot_width=250, plot_height=120, toolbar_location=None)

layout = column(row(topdiv), row(column(div, all_data_plot), column(div2, row(ts1, ts2), column(button_ml, button_cl))), div3, initial_temp_clustering, div4)
curdoc().add_root(layout)



querier = VisualQuerier(data, curdoc(), layout)
clusterer = COBRAS_kShape(data, querier, 100000)

def blocking_task():
    clusterer.cluster()

thread = Thread(target=blocking_task)
thread.start()


