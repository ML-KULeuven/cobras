import abc


class SuperInstance:
    """
        A class representing a super-instance used in the COBRAS algorithm
    """

    def __init__(self, data, indices, train_indices, parent=None):
        """

        :param data: the full dataset
        :param indices: the indices of the instances that are in this super-instance
        :param train_indices: all training indices (i.e. training indices of the full dataset)
        :param parent: the parent super-instance (if any)
        """
        if not isinstance(indices, list):
            raise ValueError('Should give a list of indices as input to SuperInstance...')

        self.data = data
        #: The indices of the instances in this super-instance
        self.indices = indices
        #: The indices of the training instances in this super-instance
        self.train_indices = [x for x in indices if x in train_indices]
        #: Whether or not we have tried splitting this super-instance in the past and failed to do so
        self.tried_splitting = False
        #: The index of the super-instance representative instance
        self.representative_idx = None

        self.children = None
        self.parent = parent


    def get_representative_idx(self):
        """
        :return: the index of the super-instance representative
        """
        try:
            return self.representative_idx
        except:
            raise ValueError('Super instances without training instances')

    @abc.abstractmethod
    def distance_to(self, other_superinstance):
        """
            Calculates the distance to the given super-instance
            This is COBRAS variant specific

        :param other_superinstance: the super-instance to calculate the distance to
        :return: the distance between this super-instance and the given other_superinstance
        """
        return

    def get_leaves(self):
        if self.children is None:
            return [self]
        else:
            d = []
            for s in self.children:
                d.extend(s.get_leaves())
            return d

    def get_root(self):
        if self.parent is None:
            return self
        else:
            return self.parent.get_root()