"""Datasets and preprocessing for mixed-type clustering.

Provides:
  * Synthetic generators (Syn1 categorical-only, Syn2 numerical-only, Syn3 balanced mixed)
    that always work offline and reproduce the controlled-signal protocol of the paper.
  * preprocess(): StandardScaler for numerical + one-hot for categorical.
  * Optional UCI loaders (Heart, Adult, Dermatology, ...) via the `ucimlrepo` package.
"""
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# ----------------------------------------------------------------------------- preprocessing
def preprocess(X_num, X_cat):
    """Scale numerical features (StandardScaler) and one-hot encode categorical features.

    Returns
    -------
    X_enc        : (n, d_num + d_onehot) encoded feature matrix used as mSDA input
    X_num_scaled : (n, d_num) standardized numerical block (used for the RBF view)
    X_cat_onehot : (n, d_onehot) one-hot categorical block (used for the cosine view)
    """
    n = (X_num.shape[0] if X_num is not None and X_num.size else
         X_cat.shape[0])
    X_num_scaled = np.empty((n, 0))
    X_cat_onehot = np.empty((n, 0))
    if X_num is not None and X_num.shape[1] > 0:
        X_num_scaled = StandardScaler().fit_transform(np.asarray(X_num, dtype=float))
    if X_cat is not None and X_cat.shape[1] > 0:
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat_onehot = enc.fit_transform(np.asarray(X_cat))
    X_enc = np.hstack([X_num_scaled, X_cat_onehot])
    return X_enc, X_num_scaled, X_cat_onehot


# ----------------------------------------------------------------------------- synthetic data
def make_synthetic(kind="mixed", n_per_class=500, n_classes=3, seed=0):
    """Generate a controlled mixed-type dataset.

    kind : 'categorical' (Syn1), 'numerical' (Syn2), or 'mixed' (Syn3).
    """
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_classes), n_per_class)
    n = y.size

    def numerical_block(d):
        # one Gaussian blob per class, well separated
        centers = rng.normal(0, 6, size=(n_classes, d))
        X = np.vstack([centers[c] + rng.normal(0, 1.0, size=(n_per_class, d))
                       for c in range(n_classes)])
        return X

    def categorical_block(d, n_levels=4):
        # each class prefers a distinct level per feature, with noise
        X = np.empty((n, d), dtype=int)
        for j in range(d):
            preferred = rng.integers(0, n_levels, size=n_classes)
            for i in range(n):
                if rng.random() < 0.75:
                    X[i, j] = preferred[y[i]]
                else:
                    X[i, j] = rng.integers(0, n_levels)
        return X

    if kind == "numerical":      # Syn2
        X_num, X_cat = numerical_block(6), np.empty((n, 0), dtype=int)
    elif kind == "categorical":  # Syn1
        X_num, X_cat = np.empty((n, 0)), categorical_block(6)
    elif kind == "mixed":        # Syn3
        X_num, X_cat = numerical_block(3), categorical_block(3)
    else:
        raise ValueError(f"unknown kind: {kind}")

    # shuffle rows
    perm = rng.permutation(n)
    X_num = X_num[perm] if X_num.size else X_num
    X_cat = X_cat[perm] if X_cat.size else X_cat
    return X_num, X_cat, y[perm]


# ----------------------------------------------------------------------------- UCI (optional)
# id -> friendly name, from the UCI ML Repository (requires `pip install ucimlrepo`).
_UCI_IDS = {"heart": 45, "adult": 2, "dermatology": 33, "credit": 143, "bands": 32}


def load_uci(name):
    """Load a UCI dataset and split features into numerical / categorical blocks.

    Requires `ucimlrepo`. Categorical columns are detected as non-numeric dtypes.
    Note: light cleaning is applied (drop rows with missing values), so sample
    counts may differ slightly from the paper.
    """
    from ucimlrepo import fetch_ucirepo  # lazy import; optional dependency
    import pandas as pd

    if name not in _UCI_IDS:
        raise ValueError(f"unknown UCI dataset '{name}'. Known: {list(_UCI_IDS)}")
    ds = fetch_ucirepo(id=_UCI_IDS[name])
    X = ds.data.features.copy()
    y_raw = ds.data.targets.iloc[:, 0]

    mask = X.notna().all(axis=1)
    X, y_raw = X[mask], y_raw[mask]

    num_cols = X.select_dtypes(include="number").columns
    cat_cols = [c for c in X.columns if c not in num_cols]

    X_num = X[num_cols].to_numpy(dtype=float) if len(num_cols) else np.empty((len(X), 0))
    X_cat = X[cat_cols].to_numpy() if len(cat_cols) else np.empty((len(X), 0), dtype=int)
    y = pd.factorize(y_raw)[0]
    return X_num, X_cat, y
