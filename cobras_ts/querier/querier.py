import abc


class Querier:

    def __init__(self):
        pass

    @abc.abstractmethod
    def query_points(self, idx1, idx2):
        return

    @abc.abstractmethod
    def update_clustering(self, clustering):
        # not ideal? this has not too much to do with querying, it is only needed for the webapp
        return

    @abc.abstractmethod
    def update_clustering_detailed(self, clustering):
        # not ideal? this has not too much to do with querying, it is only needed for the webapp
        return
