"""Multi-view similarity construction and fusion.

Numerical features  -> Gaussian RBF kernel
Categorical features -> cosine similarity (on one-hot encodings)
Fusion              -> convex combination  S_fused = alpha * S_num + (1 - alpha) * S_cat
"""
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity


def row_normalize(S):
    """Row-normalize a (n, n) similarity matrix so each row sums to 1 (stochastic)."""
    S = np.asarray(S, dtype=float)
    r = S.sum(axis=1, keepdims=True)
    r[r == 0.0] = 1.0
    return S / r


def numerical_similarity(X_num, gamma=0.5):
    """RBF kernel over (standardized) numerical features. Returns None if no numerical features."""
    if X_num is None or X_num.shape[1] == 0:
        return None
    return rbf_kernel(X_num, gamma=gamma)


def categorical_similarity(X_cat_onehot):
    """Cosine similarity over one-hot encoded categorical features. None if no categorical features."""
    if X_cat_onehot is None or X_cat_onehot.shape[1] == 0:
        return None
    return cosine_similarity(X_cat_onehot)


def fuse(S_num, S_cat, alpha=0.5):
    """Convex combination of (row-normalized) numerical and categorical similarities.

    Falls back gracefully when one view is absent (purely numerical / purely categorical data).
    """
    if S_num is None and S_cat is None:
        raise ValueError("At least one similarity view is required.")
    if S_num is None:
        return row_normalize(S_cat)
    if S_cat is None:
        return row_normalize(S_num)
    fused = alpha * row_normalize(S_num) + (1.0 - alpha) * row_normalize(S_cat)
    return row_normalize(fused)


def build_fused_similarity(X_num, X_cat_onehot, gamma=0.5, alpha=0.5):
    """Convenience: build S_num, S_cat and fuse them in one call."""
    S_num = numerical_similarity(X_num, gamma=gamma)
    S_cat = categorical_similarity(X_cat_onehot)
    return fuse(S_num, S_cat, alpha=alpha)
