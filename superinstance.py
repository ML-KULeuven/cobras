import numpy as np
from kshape.core import _sbd


def get_prototype(A,indices, prototype):
    max_dist_to_others = np.inf
    prototype_idx = None

    for idx in indices:
        cur_dist, _ = _sbd(A[idx,:],prototype)
        if cur_dist < max_dist_to_others: 
            max_dist_to_others = cur_dist
            prototype_idx = idx
    return prototype_idx

class SuperInstance:

    def __init__(self, A, indices, train_indices, representative=None):
        if not isinstance(indices, list):
            raise ValueError('A very specific bad thing happened.')
            exit()
        self.A = A
        self.indices = indices
        self.si_train_indices =  [x for x in indices if x in train_indices]

        self.train_indices = [x for x in indices if x in train_indices]

        self.representative = representative

        if representative is not None:
            self.representative_idx = get_prototype(A, self.train_indices, representative)
        else:
            self.representative_idx = indices[0]

        self.cov = None
        self.cov_inv = None
        self.cov_compute_indices = None
        self.do_not_split = False

        self.already_tried = False

    def get_medoid(self):
        try:
            return self.representative_idx
        except:
            raise ValueError('Super instances without training instances')

    def distance_to(self, other_cluster):
        d, _ = _sbd(self.representative, other_cluster.representative)
        return d
        #return self.A[self.prototype_idx, other_cluster.prototype_idx]

    '''
    def affinity_to_all_points(self, other_si):
        max_affinity = -np.inf
        for idx1 in self.indices:
            for idx2 in other_si.indices:
                cur_affinity = self.A[idx1, idx2]
                if cur_affinity > max_affinity:
                    max_affinity = cur_affinity
        return max_affinity
    '''
