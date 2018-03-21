from cluster import Cluster
from clustering import Clustering
from superinstance import SuperInstance
from sklearn.cluster import KMeans, AffinityPropagation, SpectralClustering
from sklearn.preprocessing import MinMaxScaler
import abc
import numpy as np
from scipy import misc
import random
import time
import itertools
from scipy import stats
from sklearn import metrics

import matplotlib.pyplot as plt

'''
def cannot_link_between_clusters_all_points(c1,c2,cl):
    c1_pts = c1.get_all_points()
    c2_pts = c2.get_all_points()
    for x,y in cl:
        if x in c1_pts and y in c2_pts:
            return True
        if x in c2_pts and y in c1_pts:
            return True

    return False
'''


def cannot_link_between_clusters_only_centroids(c1, c2, cl):
    medoids_c1 = [si.prototype_idx for si in c1.super_instances]
    medoids_c2 = [si.prototype_idx for si in c2.super_instances]

    for x, y in itertools.product(medoids_c1, medoids_c2):
        if (x, y) in cl or (y, x) in cl:
            return True

    return False
    '''
    bc1, bc2 = c1.get_comparison_points(c2)
    pt1 = min([bc1.medoid, bc2.medoid])
    pt2 = max([bc1.medoid, bc2.medoid])

    if (pt1,pt2) in cl or (pt2,pt1) in cl:
        return True
    else:
        return False
    '''


def get_prototype(A,indices):
    max_affinity_to_others = -np.inf
    prototype_idx = None

    for idx in indices:
        affinity_to_others = 0.0
        for j in indices:
            if j == idx:
                continue
            affinity_to_others += A[idx,j]
        if affinity_to_others > max_affinity_to_others:
            prototype_idx = idx
            max_affinity_to_others = affinity_to_others

    return prototype_idx

class COBRAS:
    def __init__(self, A, labels, max_questions, train_indices, expected_change=False, alpha=1.0, recursive=False):
        self.A = A
        self.labels = labels
        self.max_questions = max_questions
        self.train_indices = train_indices
        self.clustering = None
        self.expected_change = expected_change

        self.split_cache = dict()

        self.alpha = alpha

        self.recursive = recursive

    def determine_starting_k(self):
        self.clustering = Clustering([Cluster([SuperInstance(self.A, range(len(self.labels)), self.train_indices)])])

        must_link_found = False
        superinstance = self.clustering.clusters[0].super_instances[0]

        init_depth = 0
        while not must_link_found:
            split_labels = self.split_superinstance(superinstance)
            new_clusters = self.add_new_clusters_from_split(superinstance, split_labels)
            if new_clusters is None:
                return init_depth
            x = new_clusters[0]
            y = new_clusters[1]

            bc1, bc2 = x.get_comparison_points(y)
            pt1 = min([bc1.prototype_idx, bc2.prototype_idx])
            pt2 = max([bc1.prototype_idx, bc2.prototype_idx])

            if self.labels[pt1] == self.labels[pt2]:
                self.ml.append((pt1, pt2))
                must_link_found = True
            else:
                self.cl.append((pt1, pt2))
                init_depth += 1

            si_to_choose = []
            if len(x.super_instances[0].si_train_indices) >= 2:
                si_to_choose.append(x.super_instances[0])
            if len(y.super_instances[0].si_train_indices) >= 2:
                si_to_choose.append(y.super_instances[0])

            if len(si_to_choose) == 0:
                return init_depth
            superinstance = random.choice(si_to_choose)

        return init_depth

    def determine_starting_k_for_superinstance(self, superinstance):
        # need to make a 'deep copy' here
        si = SuperInstance(self.A, [idx for idx in superinstance.indices], self.train_indices)

        must_link_found = False

        init_depth = 0
        while not must_link_found:
            data_to_cluster = self.A[np.ix_(si.indices, si.indices)]

            print("determining k for a super-instance")
            print(len(si.indices))
            if len(si.indices) == 2:
                split_labels = [0,1]
            else:
                km = SpectralClustering(n_clusters=2,affinity='precomputed')
                km.fit(data_to_cluster)
                split_labels = km.labels_.astype(np.int)

            si_containing_training = self.merge_superinstances_without_training(si, split_labels)

            new_clusters = []
            for si in si_containing_training:
                new_clusters.append(Cluster([si]))
            if len(new_clusters) == 1:
                new_clusters = None

            if new_clusters is None:
                return init_depth
            x = new_clusters[0]
            y = new_clusters[1]

            bc1, bc2 = x.get_comparison_points(y)
            pt1 = min([bc1.prototype_idx, bc2.prototype_idx])
            pt2 = max([bc1.prototype_idx, bc2.prototype_idx])

            if self.labels[pt1] == self.labels[pt2]:
                self.ml.append((pt1, pt2))
                self.results.append((self.results[-1][0], 0, len(self.ml) + len(self.cl)))
                must_link_found = True
            else:
                self.cl.append((pt1, pt2))
                self.results.append((self.results[-1][0], 0, len(self.ml) + len(self.cl)))

                init_depth += 1

            si_to_choose = []
            if len(x.super_instances[0].si_train_indices) >= 2:
                si_to_choose.append(x.super_instances[0])
            if len(y.super_instances[0].si_train_indices) >= 2:
                si_to_choose.append(y.super_instances[0])

            if len(si_to_choose) == 0:
                return init_depth

            si = random.choice(si_to_choose)

        return init_depth

    def merge_superinstances_without_training(self, to_split, split_labels):
        training = []
        no_training = []

        for new_si_idx in range(len(set(split_labels))):
            # go from super instance indices to global ones
            cur_indices = [to_split.indices[idx] for idx, c in enumerate(split_labels) if c == new_si_idx]

            si_train_indices = [x for x in cur_indices if x in self.train_indices]
            if len(si_train_indices) != 0:
                training.append(SuperInstance(self.A, cur_indices, self.train_indices))
            else:
                no_training.append((cur_indices, get_prototype(self.A,cur_indices)))

        for indices, centroid in no_training:
            closest_train = max(training, key=lambda x: self.A[x.prototype_idx,centroid])
            closest_train.indices.extend(indices)

        return training

    def add_new_clusters_from_split(self, to_split, split_labels):
        # print 'in add_new_clusters_from_split'
        # print to_split
        # print split_labels
        si_containing_training = self.merge_superinstances_without_training(to_split, split_labels)

        new_clusters = []
        for si in si_containing_training:
            new_clusters.append(Cluster([si]))

        if len(new_clusters) == 1:
            return None
        else:
            return new_clusters

    def query_relation_between_superinstances(self, x, y):
        x_pts = np.random.choice(x.si_train_indices, 3, replace=False)
        y_pts = np.random.choice(y.si_train_indices, 3, replace=False)

        n_cl = 0
        n_ml = 0

        idx = 0
        while n_cl < 2 and n_ml < 2:
            pt1 = min([x_pts[idx], y_pts[idx]])
            pt2 = max([x_pts[idx], y_pts[idx]])
            if self.labels[pt1] == self.labels[pt2]:
                n_ml += 1
                self.ml.append((pt1, pt2))
            else:
                n_cl += 1
                self.cl.append((pt1, pt2))

            if n_cl == 2:
                return False

            if n_ml == 2:
                return True

            idx += 1

        print
        "should never be here"
        exit()
        return None

    def merge_containing_clusters(self, starting_level=False):

        if len(self.results) > 0:
            start_clustering = self.results[-1][0]
        else:
            start_clustering = [0] * len(self.labels)

        merged = True

        while merged and len(self.ml) + len(self.cl) < self.max_questions:

            cluster_pairs = itertools.combinations(self.clustering.clusters, 2)
            cluster_pairs = [x for x in cluster_pairs if
                             not cannot_link_between_clusters_only_centroids(x[0], x[1], self.cl)]
            cluster_pairs = sorted(cluster_pairs, key=lambda x: -x[0].affinity_to(x[1]))

            merged = False
            for x, y in cluster_pairs:

                if cannot_link_between_clusters_only_centroids(x, y, self.cl):
                    continue

                bc1, bc2 = x.get_comparison_points(y)

                # must_link = self.query_relation_between_superinstances(bc1,bc2)



                pt1 = min([bc1.prototype_idx, bc2.prototype_idx])
                pt2 = max([bc1.prototype_idx, bc2.prototype_idx])

                '''
                if len(self.ml + self.cl) > 67:
                    plt.figure()
                    plt.scatter(self.data[:, 0], self.data[:, 1], s=100)
                    plt.scatter(self.data[bc1.indices,0],self.data[bc1.indices,1],s=100,color='black')
                    plt.scatter(self.data[bc2.indices,0],self.data[bc2.indices,1],s=100,color='black')

                    if self.labels[pt1] == self.labels[pt2]:
                        plt.plot([self.data[pt1, 0], self.data[pt2, 0]], [self.data[pt1, 1], self.data[pt2, 1]],
                                 color='green', linewidth=2, alpha=1.0)
                    else:
                        plt.plot([self.data[pt1, 0], self.data[pt2, 0]], [self.data[pt1, 1], self.data[pt2, 1]],
                                 color='red', linewidth=2, alpha=1.0)


                    plt.title(str(len(self.ml + self.cl)))


                    #plt.show()
                '''

                if (pt1, pt2) in self.ml:
                    # if must_link:

                    x.super_instances.extend(y.super_instances)
                    self.clustering.clusters.remove(y)
                    merged = True
                    break

                if len(self.ml) + len(self.cl) == self.max_questions:
                    break

                if self.labels[pt1] == self.labels[pt2]:
                    # if must_link:
                    x.super_instances.extend(y.super_instances)
                    self.clustering.clusters.remove(y)
                    self.ml.append((pt1, pt2))

                    merged = True

                    if starting_level:
                        self.results.append(
                            (construct_cluster_labeling(self.clustering.clusters), time.time() - self.start,
                             len(self.ml) + len(self.cl)))
                        self.superinstance_clust.append(
                            construct_cluster_labeling_from_superinstances(self.clustering.get_super_instances()))
                    else:
                        self.results.append((start_clustering, time.time() - self.start, len(self.ml) + len(self.cl)))
                        self.superinstance_clust.append(
                            construct_cluster_labeling_from_superinstances(self.clustering.get_super_instances()))

                    break
                else:
                    self.cl.append((pt1, pt2))

                    if starting_level:
                        self.results.append(
                            (construct_cluster_labeling(self.clustering.clusters), time.time() - self.start,
                             len(self.ml) + len(self.cl)))
                        self.superinstance_clust.append(
                            construct_cluster_labeling_from_superinstances(self.clustering.get_super_instances()))

                    else:
                        self.results.append((start_clustering, time.time() - self.start, len(self.ml) + len(self.cl)))
                        self.superinstance_clust.append(
                            construct_cluster_labeling_from_superinstances(self.clustering.get_super_instances()))

            if not merged and not starting_level:
                self.results[-1] = (construct_cluster_labeling(self.clustering.clusters), time.time() - self.start,
                                    len(self.ml) + len(self.cl))

    def cluster(self):

        self.start = time.time()

        self.superinstance_clust = []
        self.results = []
        self.ml = []
        self.cl = []

        depth = self.determine_starting_k()
        depth = max([1, depth])
        initial_k = min(2 ** int(depth), len(self.labels))

       

        for i in range(len(self.ml) + len(self.cl)):
            self.results.append(([0] * len(self.labels), time.time() - self.start, len(self.ml) + len(self.cl)))
            self.superinstance_clust.append([0] * len(self.labels))


        km = SpectralClustering(initial_k,affinity='precomputed')
        km.fit_predict(self.A)
        pred = km.labels_.astype(np.int)




        superinstances = self.merge_superinstances_without_training(
            SuperInstance(self.A, range(len(self.labels)), self.train_indices), pred)

        self.clustering = Clustering([])
        for si in superinstances:
            self.clustering.clusters.append(Cluster([si]))

        self.merge_containing_clusters(starting_level=True)

        while len(self.ml) + len(self.cl) < self.max_questions:
            # print "\n\n==========\nnew while iteration, current number of queries: " + str(len(self.ml+self.cl))
            # print "current number of super-instances: " + str(len(self.clustering.get_super_instances()))
            # print "current number of result clusterings: " + str(len(self.results))
            to_split, originating_cluster = self.identify_superinstance_to_split()

            if to_split is None:
                print("nothing left to split")
                break

            self.remove_superinstance(to_split)

            # if self.recursive:
            split_level = self.determine_starting_k_for_superinstance(to_split)
            split_level = max([split_level, 1])
            # else:
            #    split_level = 1
            # split_level = 1

            cur_n_k = 2 ** int(split_level)
            # cur_n_k = 10
            if cur_n_k > len(to_split.indices):
                cur_n_k = len(to_split.indices)

            split_labels = self.split_superinstance(to_split, cur_n_k)

            new_clusters = self.add_new_clusters_from_split(to_split, split_labels)

            if not new_clusters:
                print
                "there are no new clusters"
                originating_cluster.super_instances.append(to_split)
                to_split.already_tried = True
                continue
            else:
                self.clustering.clusters.extend(new_clusters)

            # print "current number of super-instances after adding new clusters: " + str(len(self.clustering.get_super_instances()))

            x = new_clusters[0]
            y = new_clusters[1]
            bc1, bc2 = x.get_comparison_points(y)
            pt1 = min([bc1.prototype_idx, bc2.prototype_idx])
            pt2 = max([bc1.prototype_idx, bc2.prototype_idx])

            '''
            plot_now = False
            if len(self.ml + self.cl) == 67:
                plt.figure()
                plt.subplot(1,3,1)
                plt.title("about to split this, nc = " + str(len(self.ml + self.cl)))

                plt.scatter(self.data[:,0],self.data[:,1],s=100,color='blue')
                plt.scatter(self.data[x.get_all_points(),0],self.data[x.get_all_points(),1],s=100,color='black')
                plt.scatter(self.data[y.get_all_points(),0],self.data[y.get_all_points(),1],s=100,color='black')

                plt.subplot(1,3,2)

                colors = [(166 / 255.0, 206 / 255.0, 227 / 255.0), (31 / 255.0, 120 / 255.0, 180 / 255.0),
                          (178 / 255.0, 223 / 255.0, 138 / 255.0), (51 / 255.0, 160 / 255.0, 44 / 255.0),
                          (251 / 255.0, 154 / 255.0, 153 / 255.0), (227 / 255.0, 26 / 255.0, 28 / 255.0),
                          (253 / 255.0, 191 / 255.0, 111 / 255.0), (255 / 255.0, 127 / 255.0, 0 / 255.0),
                          (202 / 255.0, 178 / 255.0, 214 / 255.0), (177 / 255.0, 89 / 255.0, 40 / 255.0),
                          (255 / 255.0, 255 / 255.0, 153 / 255.0), (106 / 255.0, 61 / 255.0, 154 / 255.0)]

                colors = colors * 10

                si_clust = construct_cluster_labeling_from_superinstances(self.clustering.get_super_instances())
                plt.scatter(self.data[:,0],self.data[:,1],s=100,c=[ colors[si_clust[i]] for i in range(len(self.labels))])

                plt.subplot(1,3,3)
                clust = construct_cluster_labeling(self.clustering.clusters)
                plt.scatter(self.data[:, 0], self.data[:, 1], s=100,
                            c=[colors[clust[i]] for i in range(len(self.labels))])


                plot_now = True
            '''

            self.merge_containing_clusters(starting_level=False)

            '''
            if plot_now:
                colors = [(166 / 255.0, 206 / 255.0, 227 / 255.0), (31 / 255.0, 120 / 255.0, 180 / 255.0),
                          (178 / 255.0, 223 / 255.0, 138 / 255.0), (51 / 255.0, 160 / 255.0, 44 / 255.0),
                          (251 / 255.0, 154 / 255.0, 153 / 255.0), (227 / 255.0, 26 / 255.0, 28 / 255.0),
                          (253 / 255.0, 191 / 255.0, 111 / 255.0), (255 / 255.0, 127 / 255.0, 0 / 255.0),
                          (202 / 255.0, 178 / 255.0, 214 / 255.0), (177 / 255.0, 89 / 255.0, 40 / 255.0),
                          (255 / 255.0, 255 / 255.0, 153 / 255.0), (106 / 255.0, 61 / 255.0, 154 / 255.0)]

                colors = colors * 10
                cur_clust = self.results[-1][0]
                plt.figure()
                plt.scatter(self.data[:,0],self.data[:,1],s=100,c=[ colors[cur_clust[i]] for i in range(len(self.labels))])
                plt.title(str(len(self.ml + self.cl)))
                plt.show()
            '''
            self.merge_containing_clusters(starting_level=False)
        return [clust for clust, _, _ in self.results], [runtime for _, runtime, _ in self.results], self.ml, self.cl

    def query_relations_of_new_clusters(self, new_clusters, originating_cluster, ml, cl):
        return self.clustering.merge_containing_clusters(self.labels, self.max_questions, ml, cl)

    def remove_superinstance(self, superinstance):
        for c in self.clustering.clusters:
            if superinstance in c.super_instances:
                c.super_instances.remove(superinstance)
                if len(c.super_instances) == 0:
                    self.clustering.clusters.remove(c)
                return

    def compute_split_value(self, superinstance):
        superinstance_data = self.A[superinstance.indices, :]

        p_prod = 1.0
        min_p = 1.0
        for dim in range(superinstance_data.shape[1]):

            _, cur_p = stats.shapiro(superinstance_data[:, dim])

            p_prod *= cur_p
            if cur_p < min_p:
                min_p = cur_p

        return 1.0 - p_prod

    def identify_superinstance_to_split(self):

        superinstances = self.clustering.get_super_instances()

        if len(superinstances) == 1:
            return superinstances[0], self.clustering.clusters[0]

        superinstance_to_split = None
        max_heur = -np.inf

        expected_changes = []
        split_values = []
        considered_superinstances = []
        for sis_id, superinstance in enumerate(superinstances):
            if superinstance.already_tried:
                continue

            if len(superinstance.indices) < 3:
                continue

            split = self.split_superinstance(superinstance)

            if not self.does_split_contain_train_points(superinstance, split):
                continue

            # average_changed_pairs = self.compute_expected_ri_change(superinstance, split)

            # average_ri_change = average_changed_pairs / misc.comb(self.data.shape[0], 2)
            # average_ri_change = average_changed_pairs
            # split_value = self.compute_split_value(superinstance)
            # split_value = 0
            # if self.expected_change:
            #    cur_heur = self.alpha * average_ri_change + (1.0 - self.alpha) * split_value
            #    #cur_heur = float(split_value) * len(superinstance.indices)
            # else:
            #    cur_heur = len(superinstance.indices)

            # expected_changes.append(average_changed_pairs)
            # expected_changes.append(average_ri_change)
            # split_values.append(split_value)
            # considered_superinstances.append(superinstance)

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

        '''
        #scaler = MinMaxScaler()
        #expected_changes = scaler.fit_transform(expected_changes)
        #
        # split_values = scaler.fit_transform(split_values)

        max_heur = -np.inf
        for ec, sv, si in zip(expected_changes, split_values, considered_superinstances):
            cur_heur = self.alpha * ec + (1.0 - self.alpha) * sv
            if cur_heur > max_heur:
                superinstance_to_split = si
                max_heur = cur_heur


        originating_cluster = None
        for cluster in self.clustering.clusters:
            if superinstance_to_split in cluster.super_instances:
                originating_cluster = cluster

        return superinstance_to_split, originating_cluster
        '''

    def compute_expected_ri_change(self, superinstance, split):
        counts = []  # contains the count of all clusters to which this superinstance does not belong
        original_cluster = None

        for c in self.clustering.clusters:
            if superinstance in c.super_instances:
                original_cluster = c
                continue
            counts.append(len(c.get_all_points()))

        n_s1 = np.sum(split == 0)  # number of instances in the newly created super-instance S_new1
        n_s2 = np.sum(split == 1)

        n_s = n_s1 + n_s2

        n_o = len(original_cluster.get_all_points()) - len(
            superinstance.indices)  # number of instances in original cluster, without the split super-instances

        nc = len(self.clustering.clusters)  # the number of clusters

        n_changed_pairs = 0  # will contain the sum of the number of changed pairs for each scenario

        # scenario 1: they both merge back into the original cluster, nothing changes

        # scenario 2: they both merge with a new cluster, but the same one
        for cidx1, count1 in enumerate(counts):
            n_changed_pairs += n_s * count1 + n_s * n_o

        # scenario 3: they both become new clusters
        n_changed_pairs += n_s * n_o + n_s1 * n_s2

        # scenario 4: they both are merged into new clusters, different from the original one
        for cidx1, count1 in enumerate(counts):
            for cidx2, count2 in enumerate(counts):
                if cidx1 == cidx2:
                    continue
                n_changed_pairs += n_s * n_o + n_s1 * n_s2 + n_s1 * count1 + n_s2 * count2

        # scenario 5: S_new1 merges back with C_k, S_new2 becomes a new cluster
        n_changed_pairs += n_s1 * n_s2 + n_o * n_s2

        # scenario 6: S_new1 merges back with C_k, S_new2 joins other cluster
        for cidx1, count1 in enumerate(counts):
            n_changed_pairs += n_s1 * n_s2 + n_o * n_s2 + count1 * n_s2

        # scenario 7: S_new2 merges back with C_k, S_new1 becomes a new cluster
        n_changed_pairs += n_s1 * n_s2 + n_o * n_s1

        # scenario 8: S_new2 merges back with C_k, S_new1 joins other cluster
        for cidx1, count1 in enumerate(counts):
            n_changed_pairs += n_s1 * n_s2 + n_o * n_s1 + count1 * n_s1

        return float(n_changed_pairs) / (4 + 2 * (nc - 1) + (nc - 1) * (nc - 2))

    def compute_expected_ri_change2(self, superinstance, split):

        originating_cluster_count = 0
        # now construct the cluster counts, without the points belonging to this super-instance
        cluster_counts = []
        is_originating_cluster = []
        for cur_cluster in self.clustering.clusters:
            cur_count = 0

            is_this_originating_cluster = False

            for super_instance in cur_cluster.super_instances:
                if super_instance != superinstance:
                    cur_count += len(super_instance.indices)
                else:
                    is_this_originating_cluster = True

            if cur_count != 0:
                cluster_counts.append(cur_count)
                is_originating_cluster.append(is_this_originating_cluster)

            if is_this_originating_cluster:
                originating_cluster_count = cur_count

        n_c1 = np.sum(split == 0)
        n_c2 = np.sum(split == 1)

        number_of_pairs_that_can_change = 0.0
        all_cases_to_consider = 0.0

        # all the cases in which both 'new clusters' are merged into an existing one
        for idx1, cluster_count1 in enumerate(cluster_counts):
            for idx2, cluster_count2 in enumerate(cluster_counts):
                if idx1 == idx2:
                    continue
                if is_originating_cluster[idx1]:
                    number_of_pairs_that_can_change += n_c1 * n_c2 + n_c1 * cluster_count1 + n_c2 * cluster_count2 + n_c2 * originating_cluster_count
                elif is_originating_cluster[idx2]:
                    number_of_pairs_that_can_change += n_c1 * n_c2 + n_c1 * cluster_count1 + n_c2 * cluster_count2 + n_c1 * originating_cluster_count
                else:
                    number_of_pairs_that_can_change += n_c1 * n_c2 + n_c1 * cluster_count1 + n_c2 * cluster_count2 + (
                                                                                                                     n_c1 + n_c2) * originating_cluster_count

                all_cases_to_consider += 1

        # the case in which one of them is assigned to an existing cluster, the other is not
        for cluster_count in cluster_counts:
            number_of_pairs_that_can_change += n_c1 * n_c2 + n_c1 * cluster_count + n_c1 * originating_cluster_count
            number_of_pairs_that_can_change += n_c1 * n_c2 + n_c2 * cluster_count + n_c2 * originating_cluster_count
            all_cases_to_consider += 2

        # the case in which neither of the new super-instances is merged with an existing cluster
        number_of_pairs_that_can_change += n_c1 * n_c2 + (n_c1 + n_c2) * originating_cluster_count
        # number_of_pairs_that_can_change += n_c1 * n_c2

        all_cases_to_consider += 1

        # now the average over all these possible cases
        return number_of_pairs_that_can_change / all_cases_to_consider

    def does_split_contain_train_points(self, superinstance, split):
        for clust_id in range(2):
            clust_indices = [superinstance.indices[idx] for idx, c in enumerate(split) if
                             c == clust_id and superinstance.indices[idx] in self.train_indices]
            if len(clust_indices) == 0:
                return False
        return True

    def split_superinstance(self, superinstance, split_level=2):
        # doing this with tuple(set()) because a change in indices due to reassigning points should trigger a re-clustering
        if tuple(set(superinstance.indices)) in self.split_cache and len(
                set(self.split_cache[tuple(set(superinstance.indices))])) == split_level:
            return self.split_cache[tuple(set(superinstance.indices))]
        else:
            if len(superinstance.indices) <= split_level:
                self.split_cache[tuple(set(superinstance.indices))] = range(len(superinstance.indices))
            else:
                data_to_cluster = self.A[np.ix_(superinstance.indices, superinstance.indices)]
                km = SpectralClustering(split_level,affinity="precomputed")
                km.fit(data_to_cluster)
                self.split_cache[tuple(set(superinstance.indices))] = km.labels_.astype(np.int)

            return self.split_cache[tuple(set(superinstance.indices))]


def construct_cluster_labeling(clusters):
    pts_per_cluster = [cluster.get_all_points() for cluster in clusters]
    pred = [-1] * sum([len(x) for x in pts_per_cluster])

    for i, pts in enumerate(pts_per_cluster):
        for pt in pts:
            pred[pt] = i
    return pred


def construct_cluster_labeling_from_superinstances(superinstances):
    pts_per_cluster = [si.indices for si in superinstances]
    pred = [-1] * sum([len(x) for x in pts_per_cluster])

    for i, pts in enumerate(pts_per_cluster):
        for pt in pts:
            pred[pt] = i
    return pred
