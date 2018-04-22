from threading import Thread

from bokeh.layouts import column, row
from bokeh.models import Button
from bokeh.plotting import curdoc


import pandas as pd

from bokeh.plotting import figure
from cobras_ts.cobras_kshape import COBRAS_kShape


import numpy as np
import datashader
import time

from bokeh.models.widgets import (Div)

from cobras_ts.visualquerier import VisualQuerier

from bokeh.models import ColumnDataSource

doc = curdoc()

print(type(doc))


def blocking_task():
    global query_answered

    clusterings, runtimes, ml, cl = clusterer.cluster()


dataset = 'CBF'

df = pd.read_csv('/home/toon/Downloads/UCR_TS_Archive_2015/' + dataset + '/' + dataset + '_TEST_SAMPLE',header=None)

data = np.loadtxt('/home/toon/Downloads/UCR_TS_Archive_2015/' + dataset + '/' + dataset + '_TEST_SAMPLE', delimiter=',')
series = data[:,1:]
labels = data[:,0]

query_answered = False

labels = df.ix[:,0]
df = df.drop(0,axis=1)
ts = df.ix[0,:]






p = figure(x_axis_type="datetime", plot_width=800, plot_height=350, toolbar_location=None) # placeholder for clusters

bla = row(p)

i = 0


def mustlink_callback():
    global query_answered
    global querier
    querier.query_answered = True
    querier.query_result = True

def cannotlink_callback():
    global query_answered
    global querier
    querier.query_answered = True
    querier.query_result = False

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



topdiv = Div(text="""<h1> COBRAS<sup>TS</sup> demo</h1>""", css_classes=['title_div'],
width=500, height=100)

div = Div(text="""<h2> The full dataset </h2>""", css_classes=['title_div'],
width=200, height=100)

div2 = Div(text="""<h2> Should these two instances be in the same cluster? </h2>""", css_classes=['title_div'],
width=500, height=100)

div3 = Div(text="""<h2> The (intermediate) clustering </h2>""", css_classes=['title_div'],
width=400, height=100)

ts1 = figure(x_axis_type="datetime", plot_width=250, plot_height=120, toolbar_location=None)
#source1 = ColumnDataSource(data=dict(x=range(data.shape[1]), y=[0]*data.shape[1])) # for some mysterious reason we have to plot a real line here ?! not ideal
#line1 = ts1.line('x', 'y', source=source1, line_width=1)

ts2 = figure(x_axis_type="datetime", plot_width=250, plot_height=120, toolbar_location=None)
#source2 = ColumnDataSource(data=dict(x=range(data.shape[1]), y=[0]*data.shape[1])) # for some mysterious reason we have to plot a real line here ?! not ideal
#line2 = ts2.line('x', 'y', source=source2, line_width=1)


entire_thing = column(row(column(div,all_data_plot),column(div2,row(ts1,ts2),column(button_ml,button_cl))), div3, bla)

# put the button and plot in a layout and add to the document
curdoc().add_root(entire_thing)



querier = VisualQuerier(data, curdoc(), entire_thing)


clusterer = COBRAS_kShape(series, querier, 100, range(len(labels)))


print("this is now curdoc")
print(curdoc())

thread = Thread(target=blocking_task)
thread.start()


