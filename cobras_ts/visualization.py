import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw_weighted as dtww


logger = logging.getLogger("cobra_ts")

def plotsuperinstancemargins(clustering, series, directory, window=None, clfs=None):
    directory = Path(directory)
    for c_idx, cluster in enumerate(clustering.clusters):
        cluster_pts = set(cluster.get_all_points())
        for s_idx, super_instance in enumerate(cluster.super_instances):
            labels = np.zeros((series.shape[0],))
            labels[super_instance.indices] = 1
            ignore_idxs = cluster_pts - set(super_instance.indices)
            weights, importances = dtww.compute_weights_using_dt(series, labels, super_instance.representative_idx,
                                                             window=window, min_ig=0.1,
                                                             max_clfs=clfs,
                                                             only_max=False, strict_cl=True,
                                                             ignore_idxs=ignore_idxs)

            dtww.plot_margins(series[super_instance.representative_idx,:], weights, filename=str(directory / f"cluster_margins_{c_idx}_{s_idx}.png"))

def plotclustermargins(final_clustering, series, directory, window=None, clfs=None):
    # TODO: Can we get medoids from method?
    directory = Path(directory)
    final_clustering = np.array(final_clustering)
    for clusterid in np.unique(final_clustering):
        logger.info(f"Plotting cluster {clusterid}")
        # TODO: Should be medoids
        prototypeidx = np.where(final_clustering == clusterid)[0][0]
        # np.savetxt(str(directory / f"labels_{clusterid}.csv"), final_clustering, delimiter=',', fmt='%i')
        # np.savetxt(str(directory / f"series_{clusterid}.csv"), series, delimiter=',')
        labels = np.zeros(final_clustering.shape)
        labels[final_clustering == clusterid] = 1
        weights, importances = dtww.compute_weights_using_dt(series, labels, prototypeidx,
                                                                  window=window, min_ig=0.1,
                                                                  max_clfs=clfs,
                                                                  only_max=False, strict_cl=True)

        fig, ax = dtww.plot_margins(series[prototypeidx], weights,
                                    filename=str(directory / f"cluster_margins_{clusterid}.png"))
        plt.close(fig)


def plotclusters(clustering, series, directory):
    directory = Path(directory)
    fig, ax = plt.subplots(nrows=len(clustering.clusters), ncols=1, sharey=True)
    for cluster_idx, cluster in enumerate(clustering.clusters):
        for clusterid in cluster.get_all_points():
            ax[cluster_idx].plot(series[clusterid], alpha=0.5)
    logger.info("Saving plots to " + str(directory / f"clusters.png"))
    fig.savefig(str(directory / f"clusters.png"))
    plt.close(fig)
