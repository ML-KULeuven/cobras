from cluster import Cluster
from clustering import Clustering
from superinstance import SuperInstance
from kshape.core import kshape, _sbd
import numpy as np
import random
import time
import itertools


class COBRAS:
    def __init__(self, A, labels, max_questions, train_indices):
        self.A = A
        self.labels = labels
        self.max_questions = max_questions
        self.train_indices = train_indices
        self.clustering = None
        self.split_cache = dict()

    def cluster(self):
        self.start = time.time()

        # 'results' will contain tuples (cluster labels, elapsed time, number of pairwise constraints)
        # we will add an entry for each constraint that is queried
        self.results = [([0] * len(self.labels),0,0)]
        self.ml = []
        self.cl = []

        # initially, there is only one super-instance that contains all data indices (i.e. list(range(len(self.labels))))
        initial_superinstance = SuperInstance(self.A, list(range(len(self.labels))), self.train_indices)

        # the split level for this initial super-instance is determined,
        # the super-instance is split, and a new cluster is created for each of the newly created superinstances
        initial_k = self.determine_split_level(initial_superinstance)
        superinstances = self.split_superinstance(initial_superinstance,initial_k)
        self.clustering = Clustering([])
        for si in superinstances:
            self.clustering.clusters.append(Cluster([si]))

        # in results we store a tuple for each constraint number
        # for the first couple of queries (those used to determine the initial splitting level)
        # we do not have a clustering, we just return 'all elements in one cluster'
        for i in range(len(self.ml) + len(self.cl)):
            self.results.append(([0] * len(self.labels), time.time() - self.start, len(self.ml) + len(self.cl)))

        # the first bottom up merging step
        self.merge_containing_clusters(starting_level=True)

        while len(self.ml) + len(self.cl) < self.max_questions:

            to_split, originating_cluster = self.identify_superinstance_to_split()
            if to_split is None:
                break

            originating_cluster.super_instances.remove(to_split)
            if len(originating_cluster.super_instances) == 0:
                self.clustering.clusters.remove(originating_cluster)


            split_level = self.determine_split_level(to_split)
            new_super_instances = self.split_superinstance(to_split, split_level)
            new_clusters = self.add_new_clusters_from_split(new_super_instances)

            if not new_clusters:
                # it is possible that splitting a super-instance does not lead to a new cluster:
                # e.g. a super-instance constains 2 points, of which one is in the test set
                # in this case, the super-instance can be split into two new ones, but these will be joined
                # again immediately, as we cannot have super-instances containing only test points (these cannot be
                # queried)
                # this case handles this, we simply add the super-instance back to its originating cluster,
                # and set the already_tried flag to make sure we do not keep trying to split this superinstance
                originating_cluster.super_instances.append(to_split)
                to_split.already_tried = True
                continue
            else:
                self.clustering.clusters.extend(new_clusters)

            self.merge_containing_clusters(starting_level=False)

        return [clust for clust, _, _ in self.results], [runtime for _, runtime, _ in self.results], self.ml, self.cl

    def split_superinstance(self, si, k):
        # the actual splitting
        pred = kshape(self.A[si.indices,:], k)

        # making sure that all of the super-instances contain at least one training instance
        # super-instances without training instance are merged with the closest one that does contain a
        # training instance
        training = []
        no_training = []

        for new_si_centroid, new_si_idx in pred:
            # go from super instance indices to global ones
            cur_indices = [si.indices[idx] for idx in new_si_idx]

            si_train_indices = [x for x in cur_indices if x in self.train_indices]
            if len(si_train_indices) != 0:
                training.append(SuperInstance(self.A, cur_indices, self.train_indices, new_si_centroid))
            else:
                no_training.append((cur_indices, new_si_centroid))

        for indices, centroid in no_training:
            # sets of indices without a training point are merged with their closest super-instance
            # closeness is based on the SBD centroid
            closest_train = None
            closest_train_dist = np.inf
            for training_si in training:
                cur_dist, _ = _sbd(training_si.representative, centroid)
                if cur_dist < closest_train_dist:
                    closest_train_dist = cur_dist
                    closest_train = training_si
            closest_train.indices.extend(indices)

        return training

    def determine_split_level(self, superinstance):
        # need to make a 'deep copy' here, we will split this one a few times just to determine an appropriate splitting
        # level
        si = SuperInstance(self.A, [idx for idx in superinstance.indices], self.train_indices, None)

        must_link_found = False

        split_level = 0
        while not must_link_found:

            if len(si.indices) == 2:
                new_si = [SuperInstance(self.A,[si.indices[0]],self.train_indices),SuperInstance(self.A,[si.indices[1]],self.train_indices)]
            else:
                new_si = self.split_superinstance(si,2)

            new_clusters = []
            for si in new_si:
                new_clusters.append(Cluster([si]))

            if len(new_clusters) == 1:
                # we cannot split any further along this branch, we reached the splitting level
                split_level = max([split_level, 1])
                split_n = 2 ** int(split_level)
                return min(len(si.indices),split_n)

            x = new_clusters[0]
            y = new_clusters[1]
            bc1, bc2 = x.get_comparison_points(y)
            pt1 = min([bc1.representative_idx, bc2.representative_idx])
            pt2 = max([bc1.representative_idx, bc2.representative_idx])

            if self.labels[pt1] == self.labels[pt2]:
                self.ml.append((pt1, pt2))
                self.results.append((self.results[-1][0], time.time() - self.start, len(self.ml) + len(self.cl)))
                must_link_found = True
            else:
                self.cl.append((pt1, pt2))
                self.results.append((self.results[-1][0], time.time() - self.start, len(self.ml) + len(self.cl)))
                split_level += 1

            si_to_choose = []
            if len(x.super_instances[0].si_train_indices) >= 2:
                si_to_choose.append(x.super_instances[0])
            if len(y.super_instances[0].si_train_indices) >= 2:
                si_to_choose.append(y.super_instances[0])

            if len(si_to_choose) == 0:
                split_level = max([split_level, 1])
                split_n = 2 ** int(split_level)
                return min(len(si.indices), split_n)

            si = random.choice(si_to_choose)

        split_level = max([split_level, 1])
        split_n = 2 ** int(split_level)
        return min(len(si.indices), split_n)

    def add_new_clusters_from_split(self, si):
        new_clusters = []
        for x in si:
            new_clusters.append(Cluster([x]))

        if len(new_clusters) == 1:
            return None
        else:
            return new_clusters

    def merge_containing_clusters(self, starting_level=False):

        start_clustering = self.results[-1][0]

        merged = True
        while merged and len(self.ml) + len(self.cl) < self.max_questions:

            cluster_pairs = itertools.combinations(self.clustering.clusters, 2)
            cluster_pairs = [x for x in cluster_pairs if
                             not x[0].cannot_link_to_other_cluster(x[1], self.cl)]
            cluster_pairs = sorted(cluster_pairs, key=lambda x: x[0].distance_to(x[1]))

            merged = False
            for x, y in cluster_pairs:

                if x.cannot_link_to_other_cluster(y, self.cl):
                    continue

                bc1, bc2 = x.get_comparison_points(y)
                pt1 = min([bc1.representative_idx, bc2.representative_idx])
                pt2 = max([bc1.representative_idx, bc2.representative_idx])


                if (pt1, pt2) in self.ml:
                    x.super_instances.extend(y.super_instances)
                    self.clustering.clusters.remove(y)
                    merged = True
                    break

                if len(self.ml) + len(self.cl) == self.max_questions:
                    break

                if self.labels[pt1] == self.labels[pt2]:
                    x.super_instances.extend(y.super_instances)
                    self.clustering.clusters.remove(y)
                    self.ml.append((pt1, pt2))
                    merged = True

                    if starting_level:
                        # if it is the first merging step there is no previous clustering that is being refined,
                        # so temporary results are the ones being constructed now
                        self.results.append(
                            (self.clustering.construct_cluster_labeling(), time.time() - self.start,
                             len(self.ml) + len(self.cl)))
                    else:
                        self.results.append((start_clustering, time.time() - self.start, len(self.ml) + len(self.cl)))

                    break
                else:
                    self.cl.append((pt1, pt2))

                    if starting_level:
                        self.results.append(
                            (self.clustering.construct_cluster_labeling(), time.time() - self.start,
                             len(self.ml) + len(self.cl)))
                    else:
                        self.results.append((start_clustering, time.time() - self.start, len(self.ml) + len(self.cl)))

            if not merged and not starting_level:
                self.results[-1] = (self.clustering.construct_cluster_labeling(), time.time() - self.start,
                                    len(self.ml) + len(self.cl))

    def identify_superinstance_to_split(self):
        superinstances = self.clustering.get_super_instances()

        if len(superinstances) == 1:
            return superinstances[0], self.clustering.clusters[0]

        superinstance_to_split = None
        max_heur = -np.inf

        for sis_id, superinstance in enumerate(superinstances):
            if superinstance.already_tried:
                continue

            if len(superinstance.indices) > max_heur:
                superinstance_to_split = superinstance
                max_heur = len(superinstance.indices)

        if superinstance_to_split is None:
            return None, None

        originating_cluster = None
        for cluster in self.clustering.clusters:
            if superinstance_to_split in cluster.super_instances:
                originating_cluster = cluster

        return superinstance_to_split, originating_cluster


