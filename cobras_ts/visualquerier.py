from cobras_ts.querier import Querier

from tornado import gen
from bokeh.models.renderers import GlyphRenderer
from functools import partial
import time
import numpy as np
import datashader
import pandas as pd
from bokeh.plotting import figure

@gen.coroutine
def update(frame1, frame2, xs, ys1, ys2):

    to_delete = None
    for r in frame1.renderers:
        if isinstance(r, GlyphRenderer)   :
            to_delete = r
    if to_delete:
        frame1.renderers.remove(to_delete)

    to_delete = None
    for r in frame2.renderers:
        if isinstance(r, GlyphRenderer)   :
            to_delete = r
    if to_delete:
        frame2.renderers.remove(to_delete)

    line1 = frame1.line(xs, y=ys1)
    line2 = frame2.line(xs, y=ys2)

    return line1, line2

@gen.coroutine
def update_clustering(bokeh_layout, data, clustering):

    n_clusters = len(set(clustering))

    plot_width = int(800 / n_clusters)
    plot_height = int(plot_width / 2)

    plots = []
    for c in set(clustering):
        c_idxs = np.where(np.array(clustering) == c)[0]
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

        plots.append(cluster_plot)

    bokeh_layout.children[2] = row(plots)


class VisualQuerier(Querier):

    def __init__(self, data):
        super(VisualQuerier, self).__init__()

        self.data = data

    def query_points(self, idx1, idx2):
        time.sleep(0.2)  # to fix (?) mysterious issue with bokeh..
        self.bokeh_doc.add_next_tick_callback(
            partial(update, line1=self.line1, line2=self.line2, ts1=self.ts1, ts2=self.ts2,
                    xs=list(range(self.data.shape[1])), ys1=self.data[idx1, :], ys2=self.data[idx2, :]))

        while not self.query_answered:
            pass
        self.query_answered = False

        return self.query_result

