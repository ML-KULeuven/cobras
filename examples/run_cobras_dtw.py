import os

import numpy as np
from dtaidistance import dtw
from sklearn import metrics

from cobras_ts import cobras_dtw

ucr_path = '/home/toon/Downloads/UCR_TS_Archive_2015'
dataset = 'ECG200'
budget = 100
alpha = 0.5
window = 10

data = np.loadtxt(os.path.join(ucr_path,dataset,dataset + '_TEST'), delimiter=',')
series = data[:,1:]
labels = data[:,0]


dists = dtw.distance_matrix(series, window=int(0.01 * window * series.shape[1]))
dists[dists == np.inf] = 0
dists = dists + dists.T - np.diag(np.diag(dists))
affinities = np.exp(-dists * alpha)

clusterer = cobras_dtw.COBRAS_DTW(affinities, labels, budget)
clusterings, runtimes, ml, cl = clusterer.cluster()
print(clusterings)
print("done")
print(metrics.adjusted_rand_score(clusterings[-1],labels))