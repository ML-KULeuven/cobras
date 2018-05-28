import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw_weighted as dtww
from dtaidistance import dtw


logger = logging.getLogger("cobra_ts")


def plotsuperinstancemargins(clustering, series, directory, window=None, psi=None, clfs=None, patternlen=None):
    directory = Path(directory)
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']

    cluster_to_generalized_super_instances = clustering.get_cluster_to_generalized_super_instance_map()
    for c_idx, cluster in enumerate(clustering.clusters):
        cluster_pts = set(cluster.get_all_points())
        for s_idx, generalized_superinstance in enumerate(cluster_to_generalized_super_instances[cluster]):
            labels = np.zeros((series.shape[0],), dtype=int)

            generalized_superinstance_indices = []
            for si in generalized_superinstance:
                generalized_superinstance_indices.extend(si.indices)

            labels[generalized_superinstance_indices] = 1
            ignore_idxs = cluster_pts - set(generalized_superinstance_indices)
            # TODO: plot all generalized superinstance representatives?
            weights, importances = dtww.compute_weights_using_dt(
                series, labels, generalized_superinstance[0].representative_idx,
                window=window, min_ig=0.01, min_purity=0.8, max_clfs=clfs,
                only_max=False, strict_cl=True, ignore_idxs=ignore_idxs,
                warping_paths_fnc=dtw.warping_paths, psi=psi, patternlen=patternlen)

            fig, ax = plt.subplots(nrows=2, ncols=1)
            dtww.plot_margins(series[generalized_superinstance[0].representative_idx,:], weights, ax=ax[0],
                              importances=importances)

            ml_cnt, cl_cnt, ig_cnt = 0, 0, 0
            for serie_idx, serie in enumerate(series):
                if serie_idx in ignore_idxs:
                    ig_cnt += 1
                    continue
                label = int(labels[serie_idx])
                if label == 0:
                    cl_cnt += 1
                elif label == 1:
                    ml_cnt += 1
                else:
                    raise Exception(f"Unknown label: {label}")
                color = colors[label]
                ax[1].plot(serie, '-', color=color, alpha=0.1 + label * 0.4)
            ax[1].set_title(f"This SI size: {ml_cnt}, Other SI size: {cl_cnt}, Ignored SI size: {ig_cnt}")

            plt.savefig(str(directory / f"cluster_margins_{c_idx}_{s_idx}.png"))
            logger.debug(f"Created cluster_margins_{c_idx}_{s_idx}.png")
            plt.close(fig)


def plotclustermargins(final_clustering, series, directory, window=None, clfs=None):
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
