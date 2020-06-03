import abc


class Querier(abc.ABC):

    def __init__(self):
        pass

    @abc.abstractmethod
    def query_points(self, idx1, idx2):
        return

    def continue_cluster_process(self):
        """Returns whether or not the clustering process should continue"""
        return True

    def update_clustering(self, clustering):
        # not ideal? this has not too much to do with querying, it is only needed for the webapp
        pass  # do nothing

    def update_clustering_detailed(self, clustering):
        # not ideal? this has not too much to do with querying, it is only needed for the webapp
        pass  # do nothing
