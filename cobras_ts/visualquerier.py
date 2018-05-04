from cobras_ts.querier import Querier

from tornado import gen
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.models import Button, Toggle
from bokeh.models.widgets import Div

from functools import partial
import time
import numpy as np
import pandas as pd
import datashader


colors = ["#cc6600", "#a0a0a0", "#00cccc", "#0066cc", "#0000cc"]


@gen.coroutine
def update(bokeh_layout, xs, ys1, ys2, iteration, num_queries):

    topdiv = Div(text="<font size=\"15\"> <b>COBRAS<sup>TS</sup></b> </font>  <br><font size=\"2\">  # queries answered: " + str(num_queries) + "</font>", css_classes=['top_title_div'],
        width=500, height=100)
    bokeh_layout.children[0].children[0] = column(topdiv)

    # This is the only thing that does not give glitches: making new figures each time.
    # Much better would probably be to put a figure once, and simply update the plot data.
    # This does not work, however, lines seem to frequently disappear for some unknown reason.
    ts1 = figure(x_axis_type="datetime", plot_width=250, plot_height=120, toolbar_location=None)
    ts2 = figure(x_axis_type="datetime", plot_width=250, plot_height=120, toolbar_location=None)
    ts1.line('x', 'y', source=dict(x=xs, y=ys1), line_width=1)
    ts2.line('x', 'y', source=dict(x=xs, y=ys2), line_width=1)

    bokeh_layout.children[1].children[1].children[1] = row(ts1,ts2)

    button_ml = bokeh_layout.children[1].children[1].children[2].children[0].children[0]
    button_cl = bokeh_layout.children[1].children[1].children[2].children[1].children[0]
    button_ml.disabled = False
    button_cl.disabled = False


def cluster_is_pure(metadata, attr, old_value, new_value):
    metadata["cluster"].is_pure = not metadata["cluster"].is_pure


def cluster_is_finished(metadata, attr, old_value, new_value):
    metadata["cluster"].is_finished = not metadata["cluster"].is_finished


@gen.coroutine
def remove_cluster_indicators(querier, bokeh_layout):
    bokeh_layout.children[4] = row()
    for col in bokeh_layout.children[3].children:
        col.children[1] = row()
        col.children[2] = row()
    querier.finished_indicating = True


def remove_cluster_indicators_callback(querier, bokeh_layout,bokeh_doc):
    bokeh_doc.add_next_tick_callback(partial(remove_cluster_indicators, querier=querier, bokeh_layout=bokeh_layout))


@gen.coroutine
def update_clustering(querier, bokeh_layout, bokeh_doc, data, clustering, cluster_indices, representatives):

    plot_width = int(800 / len(clustering))
    plot_height = int(plot_width / 2)

    plots = []
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

        for i, repr_idx in enumerate(cluster_representatives):
            cluster_plot.line('x', 'y', source=dict(x=range(data.shape[1]), y=data[repr_idx, :]), line_width=2, line_color=colors[i % len(colors)])

        plots.append(cluster_plot)

        button = Toggle(label="This cluster is pure.", active=c.is_pure)
        button.on_change("active", partial(cluster_is_pure, {"cluster" : c}))

        button2 = Toggle(label="This cluster is pure and complete.", active=c.is_finished)
        button2.on_change("active", partial(cluster_is_finished, {"cluster" : c}))

        cols.append(column(cluster_plot,button,button2))

    topdiv = Div(
        text="<font size=\"15\"> <b>COBRAS<sup>TS</sup></b> </font>  <br><font size=\"2\">  # queries answered: " + str(
            querier.n_queries) + "</font>", css_classes=['top_title_div'],
        width=500, height=100)
    bokeh_layout.children[0].children[0] = column(topdiv)

    bokeh_layout.children[3] = row(cols)

    finished_indicating = Button(label="Show me more queries!",button_type="danger")
    finished_indicating.on_click(partial(remove_cluster_indicators_callback, querier=querier, bokeh_layout=bokeh_layout, bokeh_doc=bokeh_doc))

    bokeh_layout.children[4] = row(finished_indicating)

    bokeh_layout.children[1].children[1].children[1] = Div(text="""Click 'Show me more queries!' below to continue querying""", width=400, height=100)


class VisualQuerier(Querier):

    def __init__(self, data, bokeh_doc, bokeh_layout):
        super(VisualQuerier, self).__init__()

        self.data = data

        self.bokeh_doc = bokeh_doc
        self.bokeh_layout = bokeh_layout

        self.query_answered = False
        self.query_result = None

        self.iteration = 0
        self.n_queries = 0

        self.finished_indicating = False

    def query_points(self, idx1, idx2):

        time.sleep(0.8)  # to fix (?) mysterious issue with bokeh..
        self.bokeh_doc.add_next_tick_callback(
            partial(update, bokeh_layout=self.bokeh_layout, xs=list(range(self.data.shape[1])), ys1=self.data[idx1, :], ys2=self.data[idx2, :], iteration=self.iteration, num_queries=self.n_queries))

        while not self.query_answered:
            pass
        self.query_answered = False

        self.n_queries += 1

        return self.query_result

    def finished_indicating(self):
        return self.finished_indicating

    def update_clustering(self, clustering):

        self.finished_indicating = False

        # we basically have to cache everything here, as it all can be modified in the main cobras loop while
        # the plotting code is running
        clusters = []
        cluster_indices = []
        si_representatives = []
        for cluster in clustering.clusters:
            clusters.append(cluster)
            cluster_indices.append(cluster.get_all_points())
            si_representatives.append([si.representative_idx for si in cluster.super_instances])

        time.sleep(0.8)  # to fix (?) mysterious issue with bokeh..
        self.bokeh_doc.add_next_tick_callback(
            partial(update_clustering, querier=self, bokeh_layout=self.bokeh_layout, bokeh_doc=self.bokeh_doc, data=self.data, clustering=clusters, cluster_indices=cluster_indices, representatives=si_representatives))

        while not self.finished_indicating:
            # this is to prevent cobras from continuing
            pass

        self.iteration += 1

