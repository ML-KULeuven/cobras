=================================
COBRAS for time-series clustering
=================================

Library for semi-supervised time series clustering using pairwise constraints.

COBRAS_TS supports three modes for constraint elicitation:

1. With *labeled data*. in this case the pairwise relations are derived from the labels.
   This is mainly used to compare COBRAS_TS experimentally to competitors.

2. With *interaction through the commandline*.
   In this case the user is queried about the pairwise relations, and can answer with yes (y) and no (n)
   through the commandline. The indices that are shown in the queries are the row indices in the specified
   time series matrix (starting from zero).

3. With *interaction through a visual user interface*.
   We are currently also working on an interactive web application that visualizes the data, queries, and intermediate
   clustering results. The image below shows the prototype of this application, it will be available here soon!

.. class:: no-web

    .. image:: raw/master/images/cobras_ts_demo_resized.png
        :alt: COBRAS^TS for interactive time series clustering
        :width: 5%
        :align: center


-----------------
Installation
-----------------

This package is available on PyPi::

    $ pip install cobras_ts


-----------------
Usage
-----------------

COBRAS from the command line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The COBRAS-TS algorithm can easily be run from the command line.
A ``cobras_ts`` script will be installed by pip::

    $ cobras_ts --format=csv --labelcol=0 /path/to/UCR_TS_Archive_2015/ECG200/ECG200_TEST

This script is also available in the repository as ``cobras_ts_cli.py``.


COBRAS as a Python package
~~~~~~~~~~~~~~~~~~~~~~~~~~

Examples can also be found in the examples subdirectory.

Running COBRAS_kShape:

    .. code-block:: python

        import os

        import numpy as np
        from sklearn import metrics

        from cobras_ts.cobras_kshape import COBRAS_kShape
        from cobras_ts.labelquerier import LabelQuerier

        ucr_path = '/home/toon/Downloads/UCR_TS_Archive_2015'
        dataset = 'ECG200'
        budget = 100

        data = np.loadtxt(os.path.join(ucr_path,dataset,dataset + '_TEST'), delimiter=',')
        series = data[:,1:]
        labels = data[:,0]

        clusterer = COBRAS_kShape(series, LabelQuerier(labels), budget)
        clusterings, runtimes, ml, cl = clusterer.cluster()

        print(clusterings)
        print("done")
        print(metrics.adjusted_rand_score(clusterings[-1],labels))


Running COBRAS_DTW:

This uses the dtaidistance package to compute the DTW distance matrix.
Note that constructing this matrix is typically the most time consuming step, and significant speedups can be achieved
by using the C implementation in the dtaidistance package.

    .. code-block:: python

        import os

        import numpy as np
        from dtaidistance import dtw
        from sklearn import metrics

        from cobras_ts.cobras_dtw import COBRAS_DTW
        from cobras_ts.labelquerier import LabelQuerier

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

        clusterer = COBRAS_DTW(affinities, LabelQuerier(labels), budget)
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