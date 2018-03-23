import numpy as np
from sklearn.cluster import SpectralClustering
from cobras_ts.superinstance_dtw import SuperInstance_DTW, get_prototype

from cobras_ts.cobras import COBRAS


class COBRAS_DTW(COBRAS):

    def split_superinstance(self, si, k):
        data_to_cluster = self.data[np.ix_(si.indices, si.indices)]
        spec = SpectralClustering(k, affinity="precomputed")
        spec.fit(data_to_cluster)
        split_labels = spec.labels_.astype(np.int)

        labels_to_indices = []
        for label in set(split_labels):
            labels_to_indices.append(np.where(split_labels == label))

        training = []
        no_training = []

        for new_si_idx in range(len(set(split_labels))):
            # go from super instance indices to global ones
            cur_indices = [si.indices[idx] for idx, c in enumerate(split_labels) if c == new_si_idx]

            si_train_indices = [x for x in cur_indices if x in self.train_indices]
            if len(si_train_indices) != 0:
                training.append(SuperInstance_DTW(self.data, cur_indices, self.train_indices))
            else:
                no_training.append((cur_indices, get_prototype(self.data, cur_indices)))

        for indices, centroid in no_training:
            closest_train = max(training, key=lambda x: self.data[x.representative_idx, centroid])
            closest_train.indices.extend(indices)

        return training

    def create_superinstance(self, indices):
        return SuperInstance_DTW(self.data, indices, self.train_indices)