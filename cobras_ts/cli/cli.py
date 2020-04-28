import argparse
import sys
from pathlib import Path
import logging
import subprocess
import time
import tempfile
from urllib.request import urlopen
import shutil
# import platform

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
    data_fn = inputs[0]
    if data_fn[:7] == "http://" or data_fn[:8] == "https://":
        # File is a url, download first
        _, tfilefn = tempfile.mkstemp(prefix="cobras_")
        logger.debug("Writing to temporary file {}".format(tfilefn))
        try:
            with open(tfilefn, "wb") as tfile:
                with urlopen(data_fn) as response:
                    logger.info("Downloaded file from {} (return code {})".format(data_fn, response.getcode()))
                    # data = response.read()
                    shutil.copyfileobj(response, tfile)
        except Exception as exc:
            logger.error(exc)
            logger.error("Failed downloading file from {}".format(data_fn))
            sys.exit(1)
        data_fn = tfilefn

    data_fn = Path(data_fn)
    logger.info("Reading file {}".format(data_fn))
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


def prepare_clusterer(dist, data, querier, budget, dtw_window=None, dtw_alpha=None, dtw_psi=None,
                      store_intermediate_results=False, **_kwargs):
    """Create a clusterer based on the arguments given by the user.

    :param dist: Type of distance (options: "dtw", "kshape")
    :param data: List of sequences
    :param querier: Querier object
    :param budget: Max number of queries (passed to querier object)
    :param dtw_window: window parameter for DTW  (passed to querier object)
    :param dtw_alpha: Alpha paramter for DTW  (passed to querier object)
    :return: Clusterer object
    """
    if dist == 'dtw':
        logger.info("Distance function: DTW")
        from dtaidistance import dtw
        from cobras_ts.cobras_dtw import COBRAS_DTW
        alpha = dtw_alpha
        # TODO this only works if dtaidistance is installed with c support?
        dists = dtw.distance_matrix_fast(data, window=dtw_window, psi=dtw_psi)
        dists[dists == np.inf] = 0
        dists = dists + dists.T - np.diag(np.diag(dists))
        # noinspection PyUnresolvedReferences
        affinities = np.exp(-dists * alpha)
        clusterer = COBRAS_DTW(affinities, querier, budget,
                               store_intermediate_results=store_intermediate_results)
    elif dist == 'kshape':
        logger.info("Distance function: kShape")
        from cobras_ts.cobras_kshape import COBRAS_kShape
        clusterer = COBRAS_kShape(data, querier, budget,
                                  store_intermediate_results=store_intermediate_results)
    elif dist == 'euclidean':
        logger.info("Distance function: Euclidean")
        from cobras_ts.cobras_kmeans import COBRAS_kmeans
        clusterer = COBRAS_kmeans(data, querier, budget,
                                  store_intermediate_results=store_intermediate_results)
    else:
        raise Exception("Unknown distance type: {}".format(dist))
    return clusterer


def create_parser():
    parser = argparse.ArgumentParser(description=description,
                                     epilog=epilog)


    parser.add_argument('--verbose', '-v', action='count', default=0, help='Verbose output')

    dist_group = parser.add_argument_group("distance arguments")
    dist_group.add_argument('--dist', choices=['dtw', 'kshape', 'euclidean'], default='dtw',
                            help='Distance computation (default is dtw)')
    dist_group.add_argument('--dtw-window', metavar='INT', dest='dtw_window', type=float, default=0.1,
                            help='Window size for DTW (if <1.0, it is considered a percentage of the length). '
                                 'Default is 0.1.')
    dist_group.add_argument('--dtw-psi', metavar='INT', dest='dtw_psi', type=int, default=0,
                            help='Psi relaxation for DTW')
    dist_group.add_argument('--dtw-alpha', metavar='FLOAT', dest='dtw_alpha', type=float, default=0.5,
                            help='Compute affinity from distance: affinity = exp(-dist * alpha)')

    vis_group = parser.add_argument_group("visualization")
    vis_group.add_argument('--visclusters', metavar='DIR',
                           help='Visualize each cluster and store files in this directory')
    vis_group.add_argument('--vismargins', metavar='DIR',
                           help='Visualize margins for each cluster and store files in this directory')
    vis_group.add_argument('--vismargins-diffs', dest='vismargins_diffs', type=int,
                           help='Maximal number of sets of differences to indiciate most different zones')
    vis_group.add_argument('--vismargins-patternlen', dest='vismargins_patternlen', type=int,
                           help='Only learn patterns over indices maximally patternlen apart')

    data_group = parser.add_argument_group("dataset arguments")
    data_group.add_argument('--format', dest='fileformat', choices=['csv'],
                            help='Dataset format')
    data_group.add_argument('--labelcol', metavar='INT', type=int,
                            help='Column with labels')
    data_group.add_argument('--visual', action='store_true',
                            help='Use visual interface to query constraints if no labels are given')
    data_group.add_argument('--images', action='store_true',
                            help='The input is a folder containing images')

    parser.add_argument('--budget', type=int, default=100,
                        help='Number of constraints to ask maximally')

    parser.add_argument('--hide-intermediate', dest='store_intermediate_results', action='store_false',
                        help='Show intermediate clusterings')
    parser.add_argument('inputs', nargs=1, help='Dataset file')

    return parser

def main(argv=None):

    parser = create_parser()
    args = parser.parse_args(argv)
    print("printing stuff in the cli")
    print(sys.argv)

    logger.setLevel(max(logging.WARNING - 10 * args.verbose, logging.DEBUG))
    logger.addHandler(logging.StreamHandler(sys.stdout))
    # Include dtw logging
    logger_dtw = logging.getLogger("be.kuleuven.dtai.distance")
    logger_dtw.setLevel(max(logging.WARNING - 10 * args.verbose, logging.DEBUG))
    logger_dtw.addHandler(logging.StreamHandler(sys.stdout))

    #if args.images:
    #    fns, features = image_to_feature_vec.convert_img_to_feature_vec(vars(args)['inputs'][0])


    if args.visual:
        # if platform.system() == "Darwin":
        #     # It's a Mac
        #     logger.info("Opening http://localhost:5006/webapp")
        #     subprocess.check_call(["open", "http://localhost:5006/webapp"])

        if args.images:
            webapp_dir = Path(__file__).parent / "webapp_images"
        else:
            webapp_dir = Path(__file__).parent / "webapp"

        logger.debug(f"Opening bokeh webapp at {webapp_dir}")
        # TODO: All arguments should be passed to webapp
        subprocess.check_call(["bokeh", "serve", "--show", str(webapp_dir), "--args", " ".join(sys.argv)])
        logger.info("Bokeh server closed")
        sys.exit(1)

    series, labels = prepare_data(**vars(args))

    if args.labelcol is None:
        from cobras_ts.querier.commandlinequerier import CommandLineQuerier
        querier = CommandLineQuerier()
    else:
        from cobras_ts.querier.labelquerier import LabelQuerier
        querier = LabelQuerier(labels)


    if args.dtw_window <= 1.0:
        # Percentage
        args.dtw_window= int(args.dtw_window * series.shape[1])
        logger.info("DTW window {}".format(args.dtw_window))
    else:
        args.dtw_window = int(args.dtw_window)
    clusterer = prepare_clusterer(data=series, querier=querier, **vars(args))

    logger.info("Start clustering ...")
    start_time = time.time()

    if args.store_intermediate_results:
        clustering, intermediate_cluster_labelings, runtimes, ml, cl = clusterer.cluster()
    else:
        clustering = clusterer.cluster()


    end_time = time.time()
    logger.info("... done clustering in {} seconds".format(end_time - start_time))
    if args.store_intermediate_results:
        print("Intermediate clusterings:")
        for clustering_idx, cur_clustering in enumerate(intermediate_cluster_labelings):
            print("--- Intermediate clusters, iteration {} ---".format(clustering_idx + 1))
            for cluster_idx in set(cur_clustering):
                print(np.where(np.array(cur_clustering) == cluster_idx)[0])

    print("Clustering:")
    print("--- Final clusters ---")
    for cluster in clustering.clusters:
        print(cluster.get_all_points())
    if args.labelcol is not None:
        print("")
        print("ARI score = " + str(metrics.adjusted_rand_score(clustering.construct_cluster_labeling(), labels)))

    if args.vismargins:
        # TODO: It would be better to plot the superinstances because of separated clusters
        # TODO: Pass the actual medoids
        from ..visualization import plotsuperinstancemargins
        logger.info("Plotting cluster margins ...")
        plotsuperinstancemargins(clusterer.clustering, series, args.vismargins, window=args.dtw_window,
                                 psi=args.dtw_psi, clfs=args.vismargins_diffs,
                                 patternlen=args.vismargins_patternlen)
    if args.visclusters:
        from ..visualization import plotclusters
        logger.info("Plotting clusters ...")
        plotclusters(clustering, series, args.visclusters)
