from cobras_ts.querier import Querier

from tornado import gen
from bokeh.models.renderers import GlyphRenderer
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show
from bokeh.layouts import row, column
from functools import partial
import time
import numpy as np
import datashader
import pandas as pd
from bokeh.models import Button, CheckboxGroup, Toggle
import collections

@gen.coroutine
def update(bokeh_layout, xs, ys1, ys2):

    # This is the only thing that does not give glitches: making new figures each time.
    # Much better would probably be to put a figure once, and simply update the plot data.
    # This does not work, however, lines seem to frequently disappear for some unknown reason.

    ts1 = figure(x_axis_type="datetime", plot_width=250, plot_height=120, toolbar_location=None)
    ts2 = figure(x_axis_type="datetime", plot_width=250, plot_height=120, toolbar_location=None)

    ts1.line('x', 'y', source=dict(x=xs, y=ys1), line_width=1)
    ts2.line('x', 'y', source=dict(x=xs, y=ys2), line_width=1)

    bokeh_layout.children[0].children[1].children[1] = row(ts1,ts2)


def cluster_is_pure(metadata, attr, old_value, new_value):
    print("toggeling cluster purity")

    metadata["cluster"].is_pure = not metadata["cluster"].is_pure

@gen.coroutine
def update_clustering(bokeh_layout, data, clustering, cluster_indices, representatives):



    plot_width = int(800 / len(clustering))
    plot_height = int(plot_width / 2)

    plots = []
    buttons = []

    cols = []

    for c, c_idxs, cluster_representatives in zip(clustering, cluster_indices, representatives):

        cur_data = data[c_idxs, :]

        x_range = 0, cur_data.shape[1]
        y_range = cur_data[1:].min(), cur_data[1:].max()

        cluster_plot = figure(plot_width=plot_width, plot_height=plot_height, x_range=x_range, y_range=y_range,
                              toolbar_location=None)
        cluster_plot.toolbar.logo = None

        # actually make the plot
        canvas = datashader.Canvas(x_range=x_range, y_range=y_range,
                                   plot_height=plot_height, plot_width=plot_width)

        # reformat the data into an appropriate DataFrame
        dfs = []
        split = pd.DataFrame({'x': [np.nan]})
        # for i in range(len(data)-1):
        for i in range(len(cur_data)):
            x = list(range(cur_data.shape[1]))
            y = cur_data[i]
            df3 = pd.DataFrame({'x': x, 'y': y})
            dfs.append(df3)
            dfs.append(split)
        df2 = pd.concat(dfs, ignore_index=True)

        agg = canvas.line(df2, 'x', 'y', datashader.count())
        img = datashader.transfer_functions.shade(agg, how='eq_hist')

        cluster_plot.image_rgba(image=[img.data], x=x_range[0], y=y_range[0], dw=x_range[1] - x_range[0],
                                dh=y_range[1] - y_range[0])


        for repr_idx in cluster_representatives:
            cluster_plot.line('x', 'y', source=dict(x=range(data.shape[1]), y=data[repr_idx, :]), line_width=2, line_color='red')

        plots.append(cluster_plot)

        button = Toggle(label="This cluster is complete, stop querying.", active=c.is_pure)
        button.on_change("active", partial(cluster_is_pure, {"cluster" : c}))

        cols.append(column(cluster_plot,button))


    bokeh_layout.children[2] = row(cols)


class VisualQuerier(Querier):

    def __init__(self, data, bokeh_doc, bokeh_layout):
        super(VisualQuerier, self).__init__()

        self.data = data

        self.bokeh_doc = bokeh_doc
        self.bokeh_layout = bokeh_layout

        self.query_answered = False
        self.query_result = None


        self.cluster_marked_as_pure = collections.defaultdict(lambda: False)


    def query_points(self, idx1, idx2):

        print("querying points.. ")
        time.sleep(0.5)  # to fix (?) mysterious issue with bokeh..
        self.bokeh_doc.add_next_tick_callback(
            partial(update, bokeh_layout=self.bokeh_layout, xs=list(range(self.data.shape[1])), ys1=self.data[idx1, :], ys2=self.data[idx2, :]))

        while not self.query_answered:
            pass
        self.query_answered = False

        return self.query_result

    def update_clustering(self, clustering):

        # we basically have to cache everything here, as it all can be modified in the main cobras loop while
        # the plotting code is running
        clusters = []
        cluster_indices = []
        si_representatives = []
        for cluster in clustering.clusters:
            clusters.append(cluster)
            cluster_indices.append(cluster.get_all_points())
            si_representatives.append([si.representative_idx for si in cluster.super_instances])

        time.sleep(0.5)  # to fix (?) mysterious issue with bokeh..
        self.bokeh_doc.add_next_tick_callback(
            partial(update_clustering, bokeh_layout=self.bokeh_layout, data=self.data, clustering=clusters, cluster_indices=cluster_indices, representatives=si_representatives))


