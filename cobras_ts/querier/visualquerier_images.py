from .querier import Querier

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
import random


colors = ["#cc6600", "#a0a0a0", "#00cccc", "#0066cc", "#0000cc"]


@gen.coroutine
def update(bokeh_layout, q1, q2, iteration, num_queries, fns, button_ml, button_cl):

    topdiv = Div(text="<img width=341 height=100 src=\'webapp_images/static/cobras_logo.png\'> <br><font size=\"2\"> <h3> # queries answered:  " + str(num_queries) + " </h3> </font>", css_classes=['top_title_div'],
        width=500, height=120)
    bokeh_layout.children[0] = topdiv

    # This is the only thing that does not give glitches: making new figures each time.
    # Much better would probably be to put a figure once, and simply update the plot data.
    # This does not work, however, lines seem to frequently disappear for some unknown reason.


    q1 = Div(text="<img width=150 height=150 src='webapp_images/static/to_cluster/" + fns[q1].split('/')[-1] + "'>")
    q2 = Div(text="<img width=150 height=150 src='webapp_images/static/to_cluster/" + fns[q2].split('/')[-1] + "'>")

    div2 = Div(text="<h2> Should these two instances be in the same cluster? </h2>", css_classes=['title_div'],
               width=500, height=20, name='wopwopwop')


    bokeh_layout.children[1] = div2

    bokeh_layout.children[2] = row(q1,q2)

    bokeh_layout.children[3] = column(button_ml,button_cl)

    #button_ml = bokeh_layout.children[3].children[0]
    #button_cl = bokeh_layout.children[3].children[1]

    #print("\n\n\n\n")
    #print(button_ml)
    #print(button_cl)
    #button_ml.disabled = False
    #button_cl.disabled = False



def cluster_is_pure(metadata, attr, old_value, new_value):
    metadata["cluster"].is_pure = not metadata["cluster"].is_pure


def cluster_is_finished(metadata, attr, old_value, new_value):
    metadata["cluster"].is_finished = not metadata["cluster"].is_finished


@gen.coroutine
def remove_cluster_indicators(querier, bokeh_layout):
    #print("\n\n\n\n\n")
    #print('are we in remove_cluster_indicators??')
    #bokeh_layout.children[4] = row()
    #for col in bokeh_layout.children[3].children:
    #    col.children[1] = row()
    #    col.children[2] = row()
    querier.finished_indicating = True


def remove_cluster_indicators_callback(querier, bokeh_layout,bokeh_doc):
    print("\n\n\n\n\n")
    print("do we get in this callback?????")
    bokeh_doc.add_next_tick_callback(partial(remove_cluster_indicators, querier=querier, bokeh_layout=bokeh_layout))


@gen.coroutine
def update_clustering(querier, bokeh_layout, bokeh_doc, data, clustering, cluster_indices, representatives, fns):

    plot_width = int(800 / len(clustering))
    plot_height = int(plot_width / 2)

    plots = []
    cols = []

    ctr = 0
    for c, c_idxs, cluster_representatives in zip(clustering, cluster_indices, representatives):

        print("\n\n\n\n\n\n\n\n\n")
        print(len(c_idxs))

        n_to_plot = min(25,len(c_idxs))

        random_selection = random.sample(c_idxs,n_to_plot)


        table_str = "<div class=\"results\"><h3>Cluster " + str(ctr+1) + "</h3><table><tr>"
        for i in range(n_to_plot):
            table_str += "<td><img width=65 height=65 src='webapp_images/static/to_cluster/" + fns[random_selection[i]].split('/')[-1] + "'></td>"
            if (i + 1) % 5 == 0:
               table_str += "</tr><tr>"

        table_str += "</tr></table></div>"

        if ctr % 2 == 0:
            table_str += "<br/><br/>"

        ctr += 1




        test_div = Div(text=table_str,width=500)
        cols.append(test_div)

    topdiv = Div(
        text="<img width=341 height=100 src=\'webapp_images/static/cobras_logo.png\'> <br><font size=\"2\"> </font>", css_classes=['top_title_div'],
        width=500, height=120)
    bokeh_layout.children[0] = topdiv

    bokeh_layout.children[1] = column(cols)

    finished_indicating = Button(label="Show me more queries!",button_type="danger")
    finished_indicating.on_click(partial(remove_cluster_indicators_callback, querier=querier, bokeh_layout=bokeh_layout, bokeh_doc=bokeh_doc))

    bokeh_layout.children[2]= row(finished_indicating)

    bokeh_layout.children[3] = Div(text="")


class VisualImageQuerier(Querier):

    def __init__(self, data, bokeh_doc, bokeh_layout, fns, button_ml, button_cl):
        super(VisualImageQuerier, self).__init__()

        self.data = data
        self.fns = fns

        self.bokeh_doc = bokeh_doc
        self.bokeh_layout = bokeh_layout

        self.query_answered = False
        self.query_result = None

        self.iteration = 0
        self.n_queries = 0

        self.finished_indicating = False

        self.button_ml = button_ml
        self.button_cl = button_cl

    def query_points(self, idx1, idx2):

        time.sleep(0.8)  # to fix (?) mysterious issue with bokeh..
        self.bokeh_doc.add_next_tick_callback(
            partial(update, bokeh_layout=self.bokeh_layout, q1=idx1, q2=idx2, iteration=self.iteration, num_queries=self.n_queries, fns=self.fns, button_ml=self.button_ml, button_cl=self.button_cl))

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
            partial(update_clustering, querier=self, bokeh_layout=self.bokeh_layout, bokeh_doc=self.bokeh_doc, data=self.data, clustering=clusters, cluster_indices=cluster_indices, representatives=si_representatives, fns=self.fns))

        while not self.finished_indicating:
            # this is to prevent cobras from continuing
            pass

        self.iteration += 1

