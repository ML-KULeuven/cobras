import numpy as np
from cobras_ts.superinstance_kmeans import SuperInstance_kmeans
from cobras_ts.cobras import COBRAS

from sklearn.cluster import KMeans

class COBRAS_kmeans(COBRAS):

    def split_superinstance(self, si, k):
        data_to_cluster = self.data[si.indices, :]
        km = KMeans(k)
        km.fit(data_to_cluster)

        split_labels = km.labels_.astype(np.int)

        training = []
        no_training = []

        for new_si_idx in set(split_labels):
            # go from super instance indices to global ones
            cur_indices = [si.indices[idx] for idx, c in enumerate(split_labels) if c == new_si_idx]

            si_train_indices = [x for x in cur_indices if x in self.train_indices]
            if len(si_train_indices) != 0:
                training.append(SuperInstance_kmeans(self.data, cur_indices, self.train_indices, si))
            else:
                no_training.append((cur_indices, np.mean(self.data[cur_indices,:],axis=0)))

        for indices, centroid in no_training:
            closest_train = min(training, key=lambda x: np.linalg.norm(self.data[x.representative_idx,:] - centroid))
            closest_train.indices.extend(indices)

        si.children = training

        return training

    def create_superinstance(self, indices, parent=None):
        return SuperInstance_kmeans(self.data, indices, self.train_indices, parent)