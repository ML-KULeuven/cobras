import argparse
import sys
from pathlib import Path
import logging
import subprocess
import time
import platform

import numpy as np
from sklearn import metrics


logger = logging.getLogger("cobra_ts")


description = """COBRAS-TS Time Series Active Semi-Supervised Clustering."""
epilog = """
Copyright 2018 KU Leuven, DTAI Research Group.
"""


def prepare_data(inputs, fileformat, labelcol, **_kwargs):
    """Read dataset based on the arguments given by the user.

    :param inputs: List of filepaths
    :param fileformat: Type of datafile (choices are "csv")
    :param labelcol: Integer or none
    :return: (series, labels)
    """
    data_fn = Path(inputs[0])
    data_format = None
    if fileformat is None:
        if data_fn.suffix == '.csv':
            data_format = 'csv'
    else:
        data_format = fileformat

    if data_format == 'csv':
        data = np.loadtxt(str(data_fn), delimiter=',')
    else:
        raise Exception("Unknown file format (use the --format argument)")

    if labelcol is None:
        series = data
        labels = None
    else:
        nonlabelcols = list(idx for idx in range(data.shape[1]) if idx != labelcol)
        series = data[:, nonlabelcols]
        labels = data[:, labelcol]
    return series, labels


def prepare_clusterer(dist, series, querier, budget, dtw_window=None, dtw_alpha=None, **_kwargs):
    """Create a clusterer based on the arguments given by the user.

    :param dist: Type of distance (options: "dtw", "kshape")
    :param series: List of sequences
    :param querier: Querier object
    :param budget: Max number of queries (passed to querier object)
    :param dtw_window: window parameter for DTW  (passed to querier object)
    :param dtw_alpha: Alpha paramter for DTW  (passed to querier object)
    :return: Clusterer object
    """
    if dist == 'dtw':
        from dtaidistance import dtw
        from cobras_ts.cobras_dtw import COBRAS_DTW
        window = dtw_window
        alpha = dtw_alpha
        dists = dtw.distance_matrix(series, window=int(0.01 * window * series.shape[1]))
        dists[dists == np.inf] = 0
        dists = dists + dists.T - np.diag(np.diag(dists))
        # noinspection PyUnresolvedReferences
        affinities = np.exp(-dists * alpha)
        clusterer = COBRAS_DTW(affinities, querier, budget)
    elif dist == 'kshape':
        from cobras_ts.cobras_kshape import COBRAS_kShape
        clusterer = COBRAS_kShape(series, querier, budget)
    else:
        raise Exception("Unknown distance type: {}".format(dist))
    return clusterer


def main(argv=None):
    parser = argparse.ArgumentParser(description=description,
                                     epilog=epilog)

    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')

    dist_group = parser.add_argument_group("distance arguments")
    dist_group.add_argument('--dist', choices=['dtw', 'kshape'], default='kshape',
                            help='Distance computation')
    dist_group.add_argument('--dtw-window', metavar='INT', dest='dtw_window', type=int, default=10,
                            help='Window size for DTW')
    dist_group.add_argument('--dtw-alpha', metavar='FLOAT', dest='dtw_alpha', type=float, default=0.5,
                            help='Compute affinity from distance: affinity = exp(-dist * alpha)')

    data_group = parser.add_argument_group("dataset arguments")
    data_group.add_argument('--format', dest='fileformat', choices=['csv'], help='Dataset format')
    data_group.add_argument('--labelcol', metavar='INT', type=int,
                            help='Column with labels')
    data_group.add_argument('--visual', action='store_true',
                            help='Use visual interface to query constraints if no labels are given')

    parser.add_argument('--budget', type=int, default=100,
                        help='Number of constraints to ask maximally')
    parser.add_argument('inputs', nargs=1, help='Dataset file')
    args = parser.parse_args(argv)

    logger.setLevel(max(logging.WARNING - 10 * args.verbose, logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))

    if args.visual:
        # TODO: check with Toon, this was in labelcol != none?
        # if platform.system() == "Darwin":
        #     # It's a Mac
        #     logger.info("Opening http://localhost:5006/webapp")
        #     subprocess.check_call(["open", "http://localhost:5006/webapp"])
        webapp_dir = Path(__file__).parent / "webapp"
        logger.debug(f"Opening bokeh webapp at {webapp_dir}")
        # TODO: All arguments should be passed to webapp
        subprocess.check_call(["bokeh", "serve", "--show", str(webapp_dir), "--args", args.inputs[0]])
        logger.info("Bokeh server closed")
        sys.exit(1)

    series, labels = prepare_data(**vars(args))

    if args.labelcol is None:
        from cobras_ts.commandlinequerier import CommandLineQuerier
        querier = CommandLineQuerier()
    else:
        from cobras_ts.labelquerier import LabelQuerier
        querier = LabelQuerier(labels)

    clusterer = prepare_clusterer(series=series, querier=querier, **vars(args))

    logger.info("Start clustering ...")
    start_time = time.time()
    clusterings, runtimes, ml, cl = clusterer.cluster()
    end_time = time.time()
    logger.info("... done clustering in {} seconds".format(end_time - start_time))
    print("Clustering:")
    print(clusterings)
    if args.labelcol is not None:
        print("ARI score = " + str(metrics.adjusted_rand_score(clusterings[-1], labels)))
