=================================
COBRAS for time-series clustering
=================================

Library for semi-supervised time series clustering using pairwise constraints.


-----------------
Installation
-----------------

This package is available on PyPi::

    $ pip install cobras_ts


-----------------
Usage
-----------------

Examples can also be found in the examples subdirectory.

Running COBRAS_kShape:

    .. code-block:: python

        import os
        import numpy as np
        from cobras_ts import cobras_kshape

        ucr_path = '/path/to/UCR/archive'
        dataset = 'ECG200'
        budget = 100

        data = np.loadtxt(os.path.join(ucr_path,dataset,dataset + '_TEST'), delimiter=',')
        series = data[:,1:]
        labels = data[:,0]

        clusterer = cobras_kshape.COBRAS_kShape(series, labels, budget)
        clusterings, runtimes, ml, cl = clusterer.cluster()


Running COBRAS_DTW:

    .. code-block:: python

        import os
        import numpy as np
        from cobras_ts import cobras_dtw

        ucr_path = '/path/to/UCR/archive'
        dataset = 'ECG200'
        budget = 100

        import os

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

