from cobras_ts.querier import Querier

class LabelQuerier(Querier):

    def __init__(self, labels):
        super(LabelQuerier, self).__init__()
        self.labels = labels

    def query_points(self, idx1, idx2):
        return self.labels[idx1] == self.labels[idx2]

    def update_clustering(self, clustering):
        return
