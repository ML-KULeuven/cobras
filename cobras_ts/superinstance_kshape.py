import numpy as np
from kshape.core import _sbd
from cobras_ts.superinstance import SuperInstance


def get_prototype(A,indices, prototype):
    max_dist_to_others = np.inf
    prototype_idx = None

    for idx in indices:
        cur_dist, _ = _sbd(A[idx,:],prototype)
        if cur_dist < max_dist_to_others: 
            max_dist_to_others = cur_dist
            prototype_idx = idx
    return prototype_idx

class SuperInstance_kShape(SuperInstance):

    def __init__(self, data, indices, train_indices, sbd_centroid=None):
        super(SuperInstance_kShape, self).__init__(data, indices, train_indices)

        self.sbd_centroid = sbd_centroid

        if sbd_centroid is not None:
            self.representative_idx = get_prototype(data, self.train_indices, sbd_centroid)
        else:
            self.representative_idx = indices[0]

    def distance_to(self, other_superinstance):
        d, _ = _sbd(self.sbd_centroid, other_superinstance.sbd_centroid)
        return d
