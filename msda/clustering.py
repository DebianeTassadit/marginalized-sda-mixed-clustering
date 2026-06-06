"""Spectral clustering on the learned representation, and clustering metrics."""
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import (
    normalized_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from scipy.optimize import linear_sum_assignment


def spectral_cluster(Z, n_clusters, affinity="rbf", n_neighbors=10, gamma=1.0, seed=0):
    """Cluster the representation Z into `n_clusters` groups via spectral clustering.

    affinity : 'rbf' (globally smooth data) or 'nearest_neighbors' (locally clustered data).
    """
    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity=affinity,
        n_neighbors=n_neighbors,
        gamma=gamma,
        assign_labels="kmeans",
        random_state=seed,
    )
    return sc.fit_predict(Z)


def clustering_accuracy(y_true, y_pred):
    """Unsupervised clustering accuracy via optimal (Hungarian) label alignment."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    row, col = linear_sum_assignment(w.max() - w)
    return w[row, col].sum() / y_pred.size


def evaluate(Z, y_true, y_pred):
    """External (NMI, ARI, ACC) and internal (Silhouette, Calinski-Harabasz, Davies-Bouldin) metrics."""
    out = {
        "NMI": float(normalized_mutual_info_score(y_true, y_pred)),
        "ARI": float(adjusted_rand_score(y_true, y_pred)),
        "ACC": float(clustering_accuracy(y_true, y_pred)),
    }
    try:
        if len(np.unique(y_pred)) > 1:
            out["Silhouette"] = float(silhouette_score(Z, y_pred))
            out["CalinskiHarabasz"] = float(calinski_harabasz_score(Z, y_pred))
            out["DaviesBouldin"] = float(davies_bouldin_score(Z, y_pred))
    except Exception:
        pass
    return out
