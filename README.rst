======================================
Semi-supervised clustering with COBRAS
======================================

Library for semi-supervised clustering using pairwise constraints.

COBRAS supports three modes for constraint elicitation:

1. With *labeled data*. in this case the pairwise relations are derived from the labels.
   This is mainly used to compare COBRAS experimentally to competitors.

2. With *interaction through the commandline*.
   In this case the user is queried about the pairwise relations, and can answer with yes (y) and no (n)
   through the commandline. The indices that are shown in the queries are the row indices in the specified
   data matrix (starting from zero).

3. With *interaction through a visual user interface*.
   If you use COBRAS-TS, the instantiation of COBRAS that is tailored to time series clustering, you can use an
   interactive web application that visualizes the data, queries, and intermediate clustering results. A demo can be
   found at https://dtai.cs.kuleuven.be/software/cobras/

.. class:: no-web

    .. image:: ../../raw/master/images/cobras_ts_demo_resized.png
        :alt: COBRAS^TS for interactive time series clustering
        :width: 5%
        :align: center


-----------------
Installation
-----------------

This package is available on PyPi::

    $ pip install cobras_ts

The following dependencies are automatically installed: dtaidistance, kshape, numpy, scikit-learn.

In case you want to use the interactive GUI, install ``cobras_ts`` using the following command to
automatically install additional dependencies (bokeh, datashader, and cloudpickle)::

    $ pip install pip cobras_ts[gui]

If you want to additionally install tensorflow in order to cluster images::

    $ pip install pip cobras_ts[gui, images]

-----------------
Usage
-----------------

1. COBRAS from the command line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The COBRAS algorithm can easily be run from the command line.
A ``cobras_ts`` script will be installed by pip::

    $ cobras_ts --format=csv --labelcol=0 /path/to/UCR_TS_Archive_2015/ECG200/ECG200_TEST

This script is also available in the repository as ``cobras_ts_cli.py``.


2. COBRAS as a Python package
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Examples of using COBRAS as a Python package can be found in the `examples` subdirectory.


3. COBRAS in an jupyter notebook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An example on how to run COBRAS for time series clustering using a jupyter notebook can be found in examples/COBRAS_notebook_timeseries.ipynb.
An example for image clustering can be found in examples/COBRAS_notebook_images.ipynb.
In these examples, queries and results are plotted directly in the notebook and user feedback is given through the notebook prompt.


4. COBRAS with a GUI
~~~~~~~~~~~~~~~~~~~~

For instructions on using COBRAS with a GUI to cluster time series, see: https://dtai.cs.kuleuven.be/software/cobras/


To run COBRAS on image data, add the --images option followed by the directory containing the images to be clustered.
Note: this requires tensorflow to be installed. For example:

    $ cobras_ts --visual --images cobras_ts/webapp_images/data



.. class:: no-web

    .. image:: ../../raw/master/images/cobras_images_resized.png
        :alt: COBRAS^TS for interactive time series clustering
        :width: 5%
        :align: center


-------------
Documentation
-------------
Additional documentation can be found at https://ml-kuleuven.github.io/cobras/


-----------------
Dependencies
-----------------

This package uses Python3, numpy, scikit-learn, kshape, tensorflow and dtaidistance.

-----------------
Contact
-----------------
Toon Van Craenendonck at toonvancraenendonck@gmail.com

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
