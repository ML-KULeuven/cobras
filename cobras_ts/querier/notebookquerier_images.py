import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

from .querier import Querier
from IPython import display


def _query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".

    Taken from: http://code.activestate.com/recipes/577058/
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

class NotebookQuerierImages(Querier):

    def __init__(self, fns):
        super(NotebookQuerierImages, self).__init__()
        self.fns = fns
        plt.figure(figsize=(20,20))

    def query_points(self, idx1, idx2):
        plt.clf()
        plt.subplot(1,2,1)
        print(idx1)
        img = mpimg.imread(self.fns[idx1])
        imgplot = plt.imshow(img)
        plt.subplot(1,2,2)
        img = mpimg.imread(self.fns[idx2])
        imgplot = plt.imshow(img)
        display.clear_output(wait=True)
        display.display(plt.gcf())

        return _query_yes_no(
            "Should the following instances be in the same cluster?  " + str(idx1) + " and " + str(idx2))

    def continue_cluster_process(self):
        return _query_yes_no("Continue querying?")

    def update_clustering(self, clustering):
        plt.clf()
        plt.subplots_adjust(wspace=0.2, hspace=0.0)
        n_clusters = len(clustering.clusters)
        for cluster_idx, cluster in enumerate(clustering.clusters):
            idxs = cluster.get_all_points()
            n_to_plot = min(5, len(idxs))
            random_selection = random.sample(idxs, n_to_plot)

            for idx, pt_idx in enumerate(random_selection):
                plt.subplot(len(clustering.clusters),5,cluster_idx * 5 + idx+1)
                img = mpimg.imread(self.fns[pt_idx])
                imgplot = plt.imshow(img)
                #plt.subplot(1,n_clusters,cluster_idx+1)
                #plt.plot(self.fns[clusterid, :], alpha=0.5)
        display.clear_output(wait=True)
        display.display(plt.gcf())
