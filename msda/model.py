"""Graph-regularized marginalized stacked denoising autoencoder (mSDA).

Reconstruction of the method described in:
    Debiane, T. and Labiod, L.
    "Marginalized Stacked Denoising Auto-Encoders for Mixed-Type Data Clustering".

Each layer admits a closed-form solution (no iterative training): the dropout corruption
is analytically marginalized, following Chen et al. (2012). After the linear denoising map,
the representation is smoothed over the fused similarity graph (S_fused) and passed through
a tanh nonlinearity. Layer encodings are concatenated to form the final representation.
"""
import numpy as np


class MarginalizedSDA:
    def __init__(self, n_layers=3, noise=0.1, lam=1.0, nonlinearity=np.tanh):
        """
        Parameters
        ----------
        n_layers : int      number of stacked denoising layers
        noise    : float    feature corruption (dropout) probability p in [0, 1)
        lam      : float    ridge regularization on the closed-form solution
        nonlinearity : callable  element-wise activation applied after each layer (default tanh)
        """
        self.n_layers = n_layers
        self.noise = noise
        self.lam = lam
        self.act = nonlinearity
        self.weights_ = []

    def _layer_weights(self, X):
        """Closed-form marginalized-denoising weights for one layer.

        X : (n, d) -> W : (d + 1, d)   (last input row corresponds to the bias term)
        """
        n, d = X.shape
        Xb = np.hstack([X, np.ones((n, 1))])          # (n, d+1) append bias feature
        q = np.full(d + 1, 1.0 - self.noise)
        q[-1] = 1.0                                    # the bias is never corrupted

        S = Xb.T @ Xb                                  # (d+1, d+1) uncorrupted scatter
        # E[ X_tilde^T X_tilde ]: off-diagonal scaled by q_i q_j, diagonal by q_i
        Q = S * np.outer(q, q)
        np.fill_diagonal(Q, np.diag(S) * q)
        # E[ X_tilde^T X ]: corruption only on the input; rows scaled by q_i, target (d cols) clean
        P = (Xb.T @ X) * q[:, None]                    # (d+1, d)

        W = np.linalg.solve(Q + self.lam * np.eye(d + 1), P)   # (d+1, d)
        return W

    def fit_transform(self, X, S_fused=None):
        """Encode X (n, d) into the concatenated multi-layer representation (n, n_layers * d).

        S_fused : optional (n, n) row-normalized fused similarity used for graph smoothing.
        """
        X = np.asarray(X, dtype=float)
        reps, H = [], X
        self.weights_ = []
        for _ in range(self.n_layers):
            W = self._layer_weights(H)
            self.weights_.append(W)
            Hb = np.hstack([H, np.ones((H.shape[0], 1))])
            Z = Hb @ W                                 # (n, d) linear denoising map
            if S_fused is not None:
                Z = S_fused @ Z                        # smooth over the fused similarity graph
            Z = self.act(Z)
            reps.append(Z)
            H = Z
        return np.hstack(reps)
