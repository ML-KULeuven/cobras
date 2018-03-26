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

This uses the dtaidistance package to compute the DTW distance matrix.
Note that constructing this matrix is typically the most time consuming step, and significant speedups can be achieved
by using the C implementation in the dtaidistance package.

    .. code-block:: python

        import os
        import numpy as np
        from cobras_ts import cobras_dtw
        from dtaidistance import dtw

        ucr_path = '/path/to/UCR/archive'
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


-----------------
Dependencies
-----------------

This package uses Python3, numpy, scikit-learn, kshape and dtaidistance.

-----------------
Contact
-----------------
Toon Van Craenendonck at toon.vancraenendonck@cs.kuleuven.be

-----------------
License
-----------------

    COBRAS code for semi-supervised time series clustering.

    Copyright 2018 KU Leuven, DTAI Research Group

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.