Web application
===============

If you supply the --visual flag to the ``cobras_ts`` command, a web application will be started.
The web application allows to visually answer queries.

Examples
--------
To cluster `a sample of the CBF dataset <https://dtai.cs.kuleuven.be/software/cobras/CBF_TEST_SAMPLE>`_ from the `UCR archive <http://timeseriesclassification.com/description.php?Dataset=CBF>`_::

    cobras_ts --visual --format=csv --dist=kshape --labelcol=0 https://bitbucket.org/toon_vc/cobras_ts/raw/master/cobras_ts/webapp/data/CBF_TEST_SAMPLE


To cluster images use::

    cobras_ts --visual --images path/to/image/directory

Where `path/to/image/directory/` is a path to a directory containing the .jpg images to cluster.

