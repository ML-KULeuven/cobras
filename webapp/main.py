# myapp.py

import random
from threading import Thread

from bokeh.layouts import column, row
from bokeh.models import Button
from bokeh.palettes import RdYlBu3
from bokeh.plotting import figure, curdoc

from bokeh.client import push_session


import pandas as pd

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show, output_file

from cobras_ts.cobras_kshape import COBRAS_kShape

import matplotlib.pyplot as plt

import numpy as np
import datashader

import time

from tornado import gen
from functools import partial

from bokeh.models.widgets import (Div)


doc = curdoc()

print(type(doc))

## write a function that can be passed to the COBRAS algorithm, which queries the relation between two instances
## and returns must link or cannot link





'''
@gen.coroutine
def update(x, y, line1, line2):
    print("we do some updating here")

    line1.visible = False
    line2.visible = False

    line1 = ts1.line(list(range(len(ts))), y=list(df.ix[x, :]))
    line2 = ts2.line(list(range(len(ts))), y=list(df.ix[y, :]))

'''

def blocking_task():
    global query_answered

    print("!!!!!!!!!!!!!!!!!!!executing the clusterer here")
    clusterings, runtimes, ml, cl = clusterer.cluster()

    '''
    while True:
        # do some blocking computation
        x, y = random.randint(0,len(ts)-1), random.randint(0,len(ts)-1)


        # but update the document from callback
        #doc.add_next_tick_callback(partial(update, x=x, y=y, line1=line1, line2=line2))

        while not query_answered:
            pass

        query_answered = False

    '''




# create a plot and style its properties

'''
data = np.loadtxt('/home/toon/Downloads/UCR_TS_Archive_2015/CBF/CBF_TEST', delimiter=',')
series = data[:,1:]
labels = data[:,0]
budget = 10

clusterer = cobras_kshape.COBRAS_kShape(series, labels, budget)
clusterings, runtimes, ml, cl = clusterer.cluster()
print(clusterings)
'''

dataset = 'CBF'

df = pd.read_csv('/home/toon/Downloads/UCR_TS_Archive_2015/' + dataset + '/' + dataset + '_TEST_SAMPLE',header=None)

data = np.loadtxt('/home/toon/Downloads/UCR_TS_Archive_2015/' + dataset + '/' + dataset + '_TEST_SAMPLE', delimiter=',')
series = data[:,1:]
labels = data[:,0]


query_answered = False

labels = df.ix[:,0]
df = df.drop(0,axis=1)
ts = df.ix[0,:]


ts1 = figure(x_axis_type="datetime", plot_width=250, plot_height=120, toolbar_location=None)
#line1 = ts1.line(list(range(len(ts))),y=list(df.ix[0,:]))



ts2 = figure(x_axis_type="datetime", plot_width=250, plot_height=120, toolbar_location=None)

#line2 = ts2.line(list(range(len(ts))),y=list(df.ix[0,:]))


p = figure(x_axis_type="datetime", plot_width=800, plot_height=350, toolbar_location=None) # placeholder for clusters

bla = row(p)

print("this is bla")
print(bla)




i = 0

'''
def query_relation(idx1, idx2):
    global line1
    global line2

    line1.visible = False
    line2.visible = False

    line1 = ts1.line(list(range(len(ts))), y=list(df.ix[idx1, :]))
    line2 = ts2.line(list(range(len(ts))), y=list(df.ix[idx2, :]))
'''

# create a callback that will add a number in a random location
def mustlink_callback():
    print("mustlink callback")
    global i
    i = i + 1

    global query_answered
    global clusterer
    clusterer.query_answered = True
    clusterer.query_result = True

def cannotlink_callback():
    print("cannotlink callback")
    global i
    i = i + 1
    global clusterer
    clusterer.query_answered = True
    clusterer.query_result = False

# add a button widget and configure with the call back
button_ml = Button(label="Yes (must-link)", button_type="success")
button_ml.on_click(mustlink_callback)

button_cl = Button(label="No (cannot-link)", button_type="warning")
button_cl.on_click(cannotlink_callback)


data = df.as_matrix()

# reformat the data into an appropriate DataFrame
dfs = []
split = pd.DataFrame({'x': [np.nan]})
#for i in range(len(data)-1):
for i in range(len(data)):
    x = list(range(len(ts)))
    y = data[i]
    df3 = pd.DataFrame({'x': x, 'y': y})
    dfs.append(df3)
    dfs.append(split)
df2 = pd.concat(dfs, ignore_index=True)

x_range = 0, data.shape[1]
y_range = data[1:].min(), data[1:].max()

print("these are the ranges")
print(x_range)
print(y_range)

all_data_plot = figure(plot_width=400, plot_height=180, x_range=x_range, y_range=y_range, title="Full dataset",toolbar_location='above')
p = figure(plot_width=400, plot_height=180, x_range=x_range, y_range=y_range, title="Full dataset",toolbar_location='above')

# actually make the plot
canvas = datashader.Canvas(x_range=x_range, y_range=y_range, 
                           plot_height=180, plot_width=400)


agg = canvas.line(df2, 'x', 'y', datashader.count())
img = datashader.transfer_functions.shade(agg, how='eq_hist')


all_data_plot.image_rgba(image=[img.data], x=x_range[0], y=y_range[0], dw=x_range[1] - x_range[0], dh=y_range[1] - y_range[0])
p.image_rgba(image=[img.data], x=x_range[0], y=y_range[0], dw=x_range[1] - x_range[0], dh=y_range[1] - y_range[0])

bla = row(p)
#p2.image_rgba(image=[img.data], x=rg_range[0], y=x2_range[0],dw=[rg_range[1]-rg_range[0]], dh=[x2_range[1]-x2_range[0]])



topdiv = Div(text="""<h1> COBRAS<sup>TS</sup> demo</h1>""", css_classes=['title_div'],
width=500, height=100)

div = Div(text="""<h2> The full dataset </h2>""", css_classes=['title_div'],
width=200, height=100)

div2 = Div(text="""<h2> Should these two instances be in the same cluster? </h2>""", css_classes=['title_div'],
width=500, height=100)

div3 = Div(text="""<h2> The (intermediate) clustering </h2>""", css_classes=['title_div'],
width=400, height=100)

entire_thing = column(row(column(div,all_data_plot),column(div2,row(ts1,ts2),column(button_ml,button_cl))), div3, bla)

print("this is the entire thing")
print(entire_thing)
print("these are the children")
print(entire_thing.children)

print("this is the row")
print(row)

# put the button and plot in a layout and add to the document
curdoc().add_root(entire_thing)



clusterer = COBRAS_kShape(series, labels, 100, range(len(labels)), curdoc(), ts1, ts2, entire_thing)


print("this is now curdoc")
print(curdoc())

thread = Thread(target=blocking_task)
thread.start()


