import numpy as np
from sklearn import metrics

from cobras_ts.cobras_kmeans import COBRAS_kmeans
from cobras_ts.labelquerier import LabelQuerier


budget = 100

data = np.loadtxt('/home/toon/data/iris.data', delimiter=',')
X = data[:,1:]
labels = data[:,0]

clusterer = COBRAS_kmeans(X, LabelQuerier(labels), budget)
clusterings, runtimes, ml, cl = clusterer.cluster()

print(clusterings)
print("done")
print(metrics.adjusted_rand_score(clusterings[-1],labels))
