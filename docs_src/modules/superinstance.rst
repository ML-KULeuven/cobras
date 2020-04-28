SuperInstance
=============

Each COBRAS variant has its own super-instance type.
The specific super-instance decides how the representative of a super-instance is chosen and provides a way to calculate the distance between two super-instances (used for querying the closest super-instances first while merging)

.. autoclass:: cobras_ts.superinstance.SuperInstance
    :members: __init__, distance_to
    :exclude-members: get_representative_idx

SuperInstance_DTW
-----------------

.. autoclass:: cobras_ts.superinstance_dtw.SuperInstance_DTW
    :members: __init__, distance_to

SuperInstance_kmeans
--------------------

.. autoclass:: cobras_ts.superinstance_kmeans.SuperInstance_kmeans
    :members: __init__, distance_to

SuperInstance_kShape
--------------------

.. autoclass:: cobras_ts.superinstance_kshape.SuperInstance_kShape
    :members: __init__, distance_to
