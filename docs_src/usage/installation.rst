Installation
============

Pip
---

This package is available on PyPi::

    $ pip install cobras_ts

To automatically install the dependencies required to use the web application use::

    $ pip install cobras_ts[gui]

To cluster images use::

    $ pip install cobras_ts[images]

Or just install all dependencies::

    $ pip install cobras_ts[gui,images]


**Note:** If you get an error saying that the C compiled version of dtaidistance is not available we advise to manually reinstall dtaidistance using `these instructions <https://dtaidistance.readthedocs.io/en/latest/usage/installation.html>`_


GitHub
---------

The source is available on https://github.com/ML-KULeuven/cobras


Dependencies
------------

To use cobras_ts as a package

* Python3
* numpy
* scikit-learn
* kshape (for cobras_kshape)
* dtaidistance (for cobras_DTW)

To use the webapp

* datashader
* bokeh
* pygments

To cluster images

* tensorflow (only for images)

To use the notebooks

* jupyter

The newest version of the dependencies is automatically installed if you use the correct pip command from above.

Additionally, a full conda environment is provided under *environment.yml*.
This can be used to reproduce the exact python environment that this package was last tested with::

    $ conda env create -f environment.yml


