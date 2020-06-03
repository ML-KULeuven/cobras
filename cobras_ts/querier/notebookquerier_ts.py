import sys
import matplotlib.pyplot as plt

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

class NotebookQuerierTS(Querier):

    def __init__(self, data):
        super(NotebookQuerierTS, self).__init__()
        self.data = data

    def query_points(self, idx1, idx2):
        plt.clf()
        plt.subplot(1,2,1)
        plt.plot(self.data[idx1,:])
        plt.subplot(1,2,2)
        plt.plot(self.data[idx2,:])
        display.clear_output(wait=True)
        display.display(plt.gcf())

        return _query_yes_no(
            "Should the following instances be in the same cluster?  " + str(idx1) + " and " + str(idx2))

    def continue_cluster_process(self):
        return _query_yes_no("Continue querying?")

    def update_clustering(self, clustering):
        plt.clf()
        n_clusters = len(clustering.clusters)
        for cluster_idx, cluster in enumerate(clustering.clusters):
            for clusterid in cluster.get_all_points():
                plt.subplot(1,n_clusters,cluster_idx+1)
                plt.plot(self.data[clusterid,:], alpha=0.5)
        display.clear_output(wait=True)
        display.display(plt.gcf())
