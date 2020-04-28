Package
=======

These examples can also be found in the examples subdirectory.

Running COBRAS_kShape
---------------------

.. literalinclude:: ../../examples/run_cobras_kshape.py
  :language: python

Running COBRAS_dtw
------------------

This uses the dtaidistance package to compute the DTW distance matrix.
Note that constructing this matrix is typically the most time consuming step, and significant speedups can be achieved
by using the C implementation in the dtaidistance package.


.. literalinclude:: ../../examples/run_cobras_dtw.py
  :language: python
