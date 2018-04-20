import abc


class Querier:

    def __init__(self):
        pass

    @abc.abstractmethod
    def query_points(self, idx1, idx2):
        return