import numpy as np
import os
import cobras_kshape
from sklearn import metrics

ucr_path = '/home/toon/Downloads/UCR_TS_Archive_2015'
dataset = 'ECG200'
budget = 100

data = np.loadtxt(os.path.join(ucr_path,dataset,dataset + '_TEST'), delimiter=',')
series = data[:,1:]
labels = data[:,0]

clusterer = cobras_kshape.COBRAS(series, labels, budget, range(len(labels)))
clusterings, runtimes, ml, cl = clusterer.cluster()
print(clusterings)
print("done")
print(metrics.adjusted_rand_score(clusterings[-1],labels))