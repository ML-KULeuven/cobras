import sys

from cobras_ts.querier import Querier


class VisualQuerier(Querier):

    def __init__(self, data):
        super(VisualQuerier, self).__init__()

    def query_points(self, idx1, idx2):
        return None
