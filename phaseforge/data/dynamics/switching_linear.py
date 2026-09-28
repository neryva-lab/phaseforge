"""Sticky Switching Linear Dynamical System (Sticky SLDS / AR-HMM) for dynamic phase discovery.

Models the multi-trajectory state-action transition dynamics:
    x_{t+1} = A_{z_t} x_t + B_{z_t} a_t + b_{z_t} + epsilon_{z_t},  epsilon_k ~ N(0, Sigma_k)
where z_t in {0, ..., K-1} is a discrete Markov state with sticky transition prior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from phaseforge.data.dynamics.features import extract_trajectory_transitions

logger = logging.getLogger(__name__)


@dataclass
class SLDSParameters:
    """Fitted parameters for a Sticky SLDS model."""

    num_regimes: int
    state_dim: int
    action_dim: int
    pi_0: np.ndarray  # (K,)
    transition_matrix: np.ndarray  # (K, K)
    A: np.ndarray  # (K, state_dim, state_dim)
    B: np.ndarray  # (K, state_dim, action_dim)
    b: np.ndarray  # (K, state_dim)
    covariances: np.ndarray  # (K, state_dim) diagonal variances
    log_likelihood_history: list[float]


def _log_gaussian_pdf_diag(
    y: np.ndarray,
    mean: np.ndarray,
    diag_var: np.ndarray,
) -> np.ndarray:
    """Compute log Gaussian PDF with diagonal covariance.

    Args:
        y: Target vectors, shape (N, D).
        mean: Mean vectors, shape (N, D) or (1, D).
        diag_var: Diagonal variances, shape (D,).

    Returns:
        Log probability density, shape (N,).
    """
    D = y.shape[-1]
    diff = y - mean
    var = np.maximum(diag_var, 1e-6)
    log_det = np.sum(np.log(var))
    quad = np.sum((diff**2) / var, axis=-1)
    return -0.5 * (D * np.log(2.0 * np.pi) + log_det + quad)


def _log_sum_exp(a: np.ndarray, axis: int = -1, keepdims: bool = False) -> np.ndarray:
    """Numerically stable log-sum-exp (handles -inf and keeps dtype)."""
    # Keepdims in the max/sum then squeeze at the end is numerically safest:
    # it avoids broadcasting mismatches when `keepdims` toggles and keeps the
    # reduction axis semantics identical for both terms.
    max_val = np.max(a, axis=axis, keepdims=True)
    # Guard against all -inf (e.g. impossible emissions) → max is -inf, exp(-inf)=0
    # then log(0) = -inf, result should be -inf, not nan.
    s = np.sum(np.exp(a - max_val), axis=axis, keepdims=True)
    # s can be 0 if max is -inf; log(0) → -inf handled by where
    with np.errstate(divide="ignore", invalid="ignore"):
        out = max_val + np.log(s)
    # When max was -inf, out is nan (-inf + -inf); correct to -inf
    out = np.where(np.isfinite(max_val), out, -np.inf)
    if not keepdims:
        out = np.squeeze(out, axis=axis)
    return out


class StickySLDS:
    """Sticky Switching Linear Dynamical System with EM estimation and Viterbi decoding.

    Args:
        num_regimes: Number of dynamic regimes K (default 6).
        sticky_kappa: Sticky prior weight favoring state self-transitions.
        dirichlet_alpha: Dirichlet prior concentration for transitions.
        ridge_lambda: L2 regularization for linear dynamics weights.
        min_variance: Minimum diagonal variance floor for numerical stability.
        max_em_iter: Maximum number of EM iterations.
        em_tol: Convergence tolerance on change in log-likelihood.
        min_duration: Minimum duration filter window for post-processing decoded regimes.
        seed: Random seed for initialization.
    """

    def __init__(
        self,
        num_regimes: int = 6,
        sticky_kappa: float = 50.0,
        dirichlet_alpha: float = 1.0,
        ridge_lambda: float = 1e-4,
        min_variance: float = 1e-4,
        max_em_iter: int = 40,
        em_tol: float = 1e-3,
        min_duration: int = 3,
        seed: int = 42,
    ) -> None:
        self.num_regimes = int(num_regimes)
        self.sticky_kappa = float(sticky_kappa)
        self.dirichlet_alpha = float(dirichlet_alpha)
        self.ridge_lambda = float(ridge_lambda)
        self.min_variance = float(min_variance)
        self.max_em_iter = int(max_em_iter)
        self.em_tol = float(em_tol)
        self.min_duration = int(min_duration)
        self.seed = int(seed)

        self.params: SLDSParameters | None = None

    def _init_params(
        self,
        trajs_x_t: list[np.ndarray],
        trajs_a_t: list[np.ndarray],
        trajs_x_next: list[np.ndarray],
    ) -> None:
        """Initialize parameters using K-means on transition residual features."""
        rng = np.random.default_rng(self.seed)
        K = self.num_regimes
        D_x = trajs_x_t[0].shape[1]
        D_a = trajs_a_t[0].shape[1]

        all_x_t = np.concatenate(trajs_x_t, axis=0)
        all_a_t = np.concatenate(trajs_a_t, axis=0)
        all_x_next = np.concatenate(trajs_x_next, axis=0)
        all_delta = all_x_next - all_x_t
        N = all_x_t.shape[0]

        # Initial clustering based on [a_t, delta_x]
        init_features = np.concatenate([all_a_t, all_delta], axis=-1)
        feat_std = np.std(init_features, axis=0) + 1e-6
        init_features_norm = init_features / feat_std

        # Simple random partition / deterministic kmeans init
        centers = init_features_norm[rng.choice(N, size=K, replace=False)]
        for _ in range(10):
            dists = np.linalg.norm(init_features_norm[:, None, :] - centers[None, :, :], axis=-1)
            assign = np.argmin(dists, axis=-1)
            for k in range(K):
                mask = assign == k
                if np.any(mask):
                    centers[k] = np.mean(init_features_norm[mask], axis=0)

        # Initial transition matrix with strong sticky diagonal
        trans = np.full((K, K), self.dirichlet_alpha)
        np.fill_diagonal(trans, self.dirichlet_alpha + self.sticky_kappa)
        trans = trans / np.sum(trans, axis=1, keepdims=True)

        pi_0 = np.full(K, 1.0 / K)

        A = np.zeros((K, D_x, D_x))
        B = np.zeros((K, D_x, D_a))
        b = np.zeros((K, D_x))
        covs = np.ones((K, D_x)) * 0.1

        # Fit initial linear dynamics per cluster
        u_all = np.concatenate([all_x_t, all_a_t, np.ones((N, 1))], axis=-1)  # (N, D_u)
        D_u = u_all.shape[-1]
        reg_eye = np.eye(D_u) * self.ridge_lambda
        reg_eye[-1, -1] = 0.0  # Do not regularize bias

        for k in range(K):
            mask = assign == k
            if np.sum(mask) < D_u:
                # Fallback to global model
                W = np.linalg.solve(u_all.T @ u_all + reg_eye, u_all.T @ all_x_next)
            else:
                u_k = u_all[mask]
                y_k = all_x_next[mask]
                W = np.linalg.solve(u_k.T @ u_k + reg_eye, u_k.T @ y_k)

            # W shape: (D_u, D_x) -> W.T shape: (D_x, D_u)
            W_T = W.T
            A[k] = W_T[:, :D_x]
            B[k] = W_T[:, D_x : D_x + D_a]
            b[k] = W_T[:, -1]

            # Estimate each regime's initial emission scale from the samples
            # assigned to that regime.  Using residuals from the full dataset
            # here gives every regime an unnecessarily broad covariance, which
            # makes the first E-step favor a collapsed assignment.  Keep the
            # global-model fallback only for the coefficient fit; the
            # covariance fallback remains numerically safe for tiny clusters.
            if np.any(mask):
                pred = u_all[mask] @ W
                res = all_x_next[mask] - pred
            else:  # pragma: no cover - k-means initializes every center
                pred = u_all @ W
                res = all_x_next - pred
            covs[k] = np.maximum(np.var(res, axis=0), self.min_variance)

        self.params = SLDSParameters(
            num_regimes=K,
            state_dim=D_x,
            action_dim=D_a,
            pi_0=pi_0,
            transition_matrix=trans,
            A=A,
            B=B,
            b=b,
            covariances=covs,
            log_likelihood_history=[],
        )

    def _compute_log_emissions(
        self,
        x_t: np.ndarray,
        a_t: np.ndarray,
        x_next: np.ndarray,
    ) -> np.ndarray:
        """Compute log P(x_{t+1} | x_t, a_t, z_t=k) for all regimes k.

        Returns:
            log_emissions, shape (T, K).
        """
        assert self.params is not None
        T = x_t.shape[0]
        K = self.num_regimes
        log_emis = np.zeros((T, K))

        for k in range(K):
            pred = x_t @ self.params.A[k].T + a_t @ self.params.B[k].T + self.params.b[k]
            log_emis[:, k] = _log_gaussian_pdf_diag(x_next, pred, self.params.covariances[k])

        return log_emis

    def _forward_backward(
        self,
        log_emis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Forward-backward algorithm for a single trajectory.

        Args:
            log_emis: Emission log-probabilities, shape (T, K).

        Returns:
            (gamma, xi, traj_ll) where gamma is posterior (T, K),
            xi is pairwise posterior (T-1, K, K), traj_ll is total log likelihood.
        """
        assert self.params is not None
        T, K = log_emis.shape
        log_pi_0 = np.log(np.maximum(self.params.pi_0, 1e-12))
        log_trans = np.log(np.maximum(self.params.transition_matrix, 1e-12))

        # Forward pass (in log space)
        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_pi_0 + log_emis[0]

        for t in range(1, T):
            # log_alpha[t, k] = log_emis[t, k] + log_sum_exp_j(log_alpha[t-1, j] + log_trans[j, k])
            trans_prop = log_alpha[t - 1, :, None] + log_trans  # (K, K)
            log_alpha[t] = log_emis[t] + _log_sum_exp(trans_prop, axis=0)

        traj_ll = float(_log_sum_exp(log_alpha[-1]))

        # Backward pass (in log space)
        log_beta = np.zeros((T, K))
        log_beta[-1] = 0.0  # log(1.0)

        for t in range(T - 2, -1, -1):
            # log_beta[t, j] = log_sum_exp_k(log_trans[j, k] + log_emis[t+1, k] + log_beta[t+1, k])
            back_prop = log_trans + log_emis[t + 1, None, :] + log_beta[t + 1, None, :]
            log_beta[t] = _log_sum_exp(back_prop, axis=1)

        # Compute posteriors (gamma)
        log_gamma = log_alpha + log_beta - traj_ll
        gamma = np.exp(np.clip(log_gamma, -700, 0))
        gamma = gamma / np.sum(gamma, axis=-1, keepdims=True)

        # Compute pairwise transitions (xi)
        xi = np.zeros((T - 1, K, K))
        for t in range(T - 1):
            log_xi_t = (
                log_alpha[t, :, None]
                + log_trans
                + log_emis[t + 1, None, :]
                + log_beta[t + 1, None, :]
                - traj_ll
            )
            xi[t] = np.exp(np.clip(log_xi_t, -700, 0))
            sum_xi = np.sum(xi[t])
            if sum_xi > 0:
                xi[t] /= sum_xi

        return gamma, xi, traj_ll

    def fit(self, trajectories: list[dict[str, Any]]) -> StickySLDS:
        """Fit Sticky SLDS using EM on training demonstrations.

        Args:
            trajectories: List of demonstration dicts with 'state' and 'action'.

        Returns:
            self
        """
        # Prepare trajectory arrays
        trajs_x_t: list[np.ndarray] = []
        trajs_a_t: list[np.ndarray] = []
        trajs_x_next: list[np.ndarray] = []

        for idx, traj in enumerate(trajectories):
            tb = extract_trajectory_transitions(traj, traj_idx=idx)
            trajs_x_t.append(tb.x_t.cpu().numpy())
            trajs_a_t.append(tb.a_t.cpu().numpy())
            trajs_x_next.append(tb.x_next.cpu().numpy())

        self._init_params(trajs_x_t, trajs_a_t, trajs_x_next)
        assert self.params is not None

        prev_ll = -np.inf
        K = self.num_regimes
        D_x = self.params.state_dim
        D_a = self.params.action_dim
        D_u = D_x + D_a + 1
        reg_eye = np.eye(D_u) * self.ridge_lambda
        reg_eye[-1, -1] = 0.0

        for iteration in range(self.max_em_iter):
            # E-Step: forward-backward on all trajectories
            total_ll = 0.0
            gammas: list[np.ndarray] = []
            xis: list[np.ndarray] = []

            for x_t, a_t, x_next in zip(trajs_x_t, trajs_a_t, trajs_x_next):
                log_emis = self._compute_log_emissions(x_t, a_t, x_next)
                gamma, xi, traj_ll = self._forward_backward(log_emis)
                gammas.append(gamma)
                xis.append(xi)
                total_ll += traj_ll

            self.params.log_likelihood_history.append(total_ll)
            ll_diff = total_ll - prev_ll
            logger.debug(
                f"SLDS EM iter {iteration}: Log-Likelihood = {total_ll:.4f}, diff = {ll_diff:.4f}"
            )

            if iteration > 0 and abs(ll_diff) < self.em_tol:
                logger.info(f"SLDS EM converged at iteration {iteration} with LL={total_ll:.4f}")
                break
            prev_ll = total_ll

            # M-Step: Parameter updates
            # 1. Initial state distribution
            pi_0_accum = np.sum([g[0] for g in gammas], axis=0) + 1e-4
            self.params.pi_0 = pi_0_accum / np.sum(pi_0_accum)

            # 2. Transition matrix with sticky Dirichlet prior
            xi_accum = np.sum([np.sum(xi, axis=0) for xi in xis], axis=0)  # (K, K)
            trans_prior = np.full((K, K), self.dirichlet_alpha)
            np.fill_diagonal(trans_prior, self.dirichlet_alpha + self.sticky_kappa)
            trans_updated = xi_accum + trans_prior
            self.params.transition_matrix = trans_updated / np.sum(
                trans_updated, axis=1, keepdims=True
            )

            # 3. Linear dynamics & variances
            all_u = np.concatenate(
                [
                    np.concatenate([x_t, a_t, np.ones((x_t.shape[0], 1))], axis=-1)
                    for x_t, a_t in zip(trajs_x_t, trajs_a_t)
                ],
                axis=0,
            )  # (N, D_u)
            all_y = np.concatenate(trajs_x_next, axis=0)  # (N, D_x)
            all_gamma = np.concatenate(gammas, axis=0)  # (N, K)

            for k in range(K):
                w_k = all_gamma[:, k : k + 1]  # (N, 1)
                eff_samples = float(np.sum(w_k))
                # Robustness: do not freeze low-occupancy regimes (which
                # self-reinforces collapse). With ridge prior the solve is
                # stable even for <2 effective samples; we only skip if
                # essentially zero responsibility.
                if eff_samples < 1e-6:
                    continue

                # Weighted least squares with ridge prior (stable for small eff_samples)
                u_w = all_u * np.sqrt(w_k)
                y_w = all_y * np.sqrt(w_k)
                try:
                    W_k = np.linalg.solve(u_w.T @ u_w + reg_eye, u_w.T @ y_w)
                except np.linalg.LinAlgError:
                    # Fallback: lstsq is more robust for near-singular systems
                    W_k, *_ = np.linalg.lstsq(u_w, y_w, rcond=None)

                W_k_T = W_k.T
                self.params.A[k] = W_k_T[:, :D_x]
                self.params.B[k] = W_k_T[:, D_x : D_x + D_a]
                self.params.b[k] = W_k_T[:, -1]

                # Weighted residual variance
                pred_k = all_u @ W_k
                res_sq = (all_y - pred_k) ** 2  # (N, D_x)
                weighted_var = np.sum(res_sq * w_k, axis=0) / eff_samples
                self.params.covariances[k] = np.maximum(weighted_var, self.min_variance)

        return self

    def decode_trajectory(self, traj: dict[str, Any]) -> np.ndarray:
        """Decode most likely dynamic regime sequence using Viterbi algorithm.

        Args:
            traj: Trajectory dict.

        Returns:
            Regime labels for all timesteps 0 ... T-1, shape (T,).
        """
        assert self.params is not None
        tb = extract_trajectory_transitions(traj)
        x_t = tb.x_t.cpu().numpy()
        a_t = tb.a_t.cpu().numpy()
        x_next = tb.x_next.cpu().numpy()

        T_minus_1 = x_t.shape[0]
        K = self.num_regimes

        log_emis = self._compute_log_emissions(x_t, a_t, x_next)
        log_pi_0 = np.log(np.maximum(self.params.pi_0, 1e-12))
        log_trans = np.log(np.maximum(self.params.transition_matrix, 1e-12))

        # Viterbi DP
        viterbi_log_prob = np.zeros((T_minus_1, K))
        backpointers = np.zeros((T_minus_1, K), dtype=int)

        viterbi_log_prob[0] = log_pi_0 + log_emis[0]

        for t in range(1, T_minus_1):
            for k in range(K):
                trans_probs = viterbi_log_prob[t - 1] + log_trans[:, k]
                best_prev = int(np.argmax(trans_probs))
                backpointers[t, k] = best_prev
                viterbi_log_prob[t, k] = trans_probs[best_prev] + log_emis[t, k]

        # Backtrack
        labels_trans = np.zeros(T_minus_1, dtype=int)
        best_last = int(np.argmax(viterbi_log_prob[-1]))
        labels_trans[-1] = best_last

        for t in range(T_minus_1 - 2, -1, -1):
            labels_trans[t] = backpointers[t + 1, labels_trans[t + 1]]

        # Extend to terminal step T-1 by repeating last decoded label
        T = (
            traj["state"].shape[0]
            if isinstance(traj["state"], np.ndarray)
            else traj["state"].size(0)
        )
        labels = np.zeros(T, dtype=int)
        labels[:T_minus_1] = labels_trans
        labels[T_minus_1:] = labels_trans[-1]

        # Apply minimum duration smoothing if configured
        if self.min_duration > 1:
            labels = self._apply_min_duration(labels, min_len=self.min_duration)

        return labels

    @staticmethod
    def _apply_min_duration(labels: np.ndarray, min_len: int) -> np.ndarray:
        """Enforce minimum duration on decoded regime segments."""
        n = len(labels)
        if n <= min_len:
            return labels

        smoothed = labels.copy()
        # Find runs
        runs: list[list[int]] = []  # list of [start, end, val]
        start = 0
        while start < n:
            end = start + 1
            while end < n and smoothed[end] == smoothed[start]:
                end += 1
            runs.append([start, end, smoothed[start]])
            start = end

        # Merge short runs into neighbors
        for i, (s, e, val) in enumerate(runs):
            length = e - s
            if length < min_len:
                # Prefer previous neighbor, else next neighbor
                if i > 0:
                    neighbor_val = runs[i - 1][2]
                elif i + 1 < len(runs):
                    neighbor_val = runs[i + 1][2]
                else:
                    neighbor_val = val
                smoothed[s:e] = neighbor_val

        return smoothed

    def score_trajectory(self, traj: dict[str, Any]) -> float:
        """Compute total log-likelihood for an out-of-sample trajectory."""
        assert self.params is not None
        tb = extract_trajectory_transitions(traj)
        x_t = tb.x_t.cpu().numpy()
        a_t = tb.a_t.cpu().numpy()
        x_next = tb.x_next.cpu().numpy()

        log_emis = self._compute_log_emissions(x_t, a_t, x_next)
        _, _, traj_ll = self._forward_backward(log_emis)
        return traj_ll


class SingleDynamicsModel:
    """Baseline single-regime linear dynamics model: x_{t+1} = A x_t + B a_t + b + epsilon."""

    def __init__(self, ridge_lambda: float = 1e-4) -> None:
        self.ridge_lambda = ridge_lambda
        self.W: np.ndarray | None = None
        self.cov: np.ndarray | None = None

    def fit(self, trajectories: list[dict[str, Any]]) -> SingleDynamicsModel:
        all_u_list: list[np.ndarray] = []
        all_y_list: list[np.ndarray] = []

        for idx, traj in enumerate(trajectories):
            tb = extract_trajectory_transitions(traj, traj_idx=idx)
            x_t = tb.x_t.cpu().numpy()
            a_t = tb.a_t.cpu().numpy()
            x_next = tb.x_next.cpu().numpy()
            u_t = np.concatenate([x_t, a_t, np.ones((x_t.shape[0], 1))], axis=-1)
            all_u_list.append(u_t)
            all_y_list.append(x_next)

        all_u = np.concatenate(all_u_list, axis=0)
        all_y = np.concatenate(all_y_list, axis=0)

        D_u = all_u.shape[-1]
        reg = np.eye(D_u) * self.ridge_lambda
        reg[-1, -1] = 0.0

        self.W = np.linalg.solve(all_u.T @ all_u + reg, all_u.T @ all_y)
        res = all_y - (all_u @ self.W)
        self.cov = np.maximum(np.var(res, axis=0), 1e-4)
        return self

    def score_trajectory(self, traj: dict[str, Any]) -> float:
        assert self.W is not None and self.cov is not None
        tb = extract_trajectory_transitions(traj)
        x_t = tb.x_t.cpu().numpy()
        a_t = tb.a_t.cpu().numpy()
        x_next = tb.x_next.cpu().numpy()

        u_t = np.concatenate([x_t, a_t, np.ones((x_t.shape[0], 1))], axis=-1)
        pred = u_t @ self.W
        log_pdf = _log_gaussian_pdf_diag(x_next, pred, self.cov)
        return float(np.sum(log_pdf))
