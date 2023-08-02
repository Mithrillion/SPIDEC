import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

import numpy as np

# from umap.spectral import spectral_layout
from scipy.sparse import coo_matrix

# TODO: allow more flexible implementation of alternative NN algorithms

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1
SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf

"""
Functions and classes concerning metric approximation and neighborhood map building
"""


class SmoothKnnDist(nn.Module):
    """
    Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.
    """

    def __init__(
        self, k: float, local_connectivity: float = 1.0, bandwidth: float = 1.0
    ) -> None:
        """
        :param k: The number of nearest neighbors to approximate for.
        :param local_connectivity: The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.
        :param bandwidth: The target bandwidth of the kernel, larger values will produce
        larger return values.
        """
        super().__init__()
        self.k = k
        self.bandwidth = bandwidth
        self.local_connectivity = local_connectivity
        self.rho = nn.UninitializedParameter(requires_grad=False, dtype=torch.float32)
        self.sigma = nn.UninitializedParameter(requires_grad=False, dtype=torch.float32)
        # TODO: must keep rho and sigma for each data sample... is there a batched solution?

    def initialize_parameters(self, distances: Tensor) -> None:
        """
        :param distances: SORTED kNN distances
        """
        # TODO: must run this before running forward() for the first time - maybe try making it a lazy module?
        with torch.no_grad():
            self.rho.materialize(distances.shape[0])
            self.sigma.materialize(distances.shape[0])
            self.rho[:] = 0
            self.sigma[:] = 1

            non_zero_mask = distances > 0
            non_zero_counts = torch.sum(non_zero_mask, dim=-1)
            zero_counts = distances.shape[1] - non_zero_counts
            is_non_zero_above_lc = non_zero_counts >= self.local_connectivity

            self.rho[(~is_non_zero_above_lc) & (non_zero_counts > 0)] = distances[
                (~is_non_zero_above_lc) & (non_zero_counts > 0), -1
            ]

            index = int(np.floor(self.local_connectivity))
            interpolation = self.local_connectivity - index
            if index > 0:
                # select (index-1)-th nonzero element (distance: columns) from each row (sample)
                near_sample_dists = torch.gather(
                    distances[is_non_zero_above_lc],
                    1,
                    (zero_counts[is_non_zero_above_lc] + index - 1).view(-1, 1),
                ).flatten()
                self.rho[is_non_zero_above_lc] = near_sample_dists
                # interpolate between (index-1)-th and index-th
                if interpolation > SMOOTH_K_TOLERANCE:
                    far_sample_dists = torch.gather(
                        distances[is_non_zero_above_lc],
                        1,
                        (zero_counts[is_non_zero_above_lc] + index).view(-1, 1),
                    ).flatten()
                    self.rho[is_non_zero_above_lc] += interpolation * (
                        far_sample_dists - near_sample_dists
                    )
            else:
                # select first nonzero element (distance: columns) from each row (sample)
                self.rho[is_non_zero_above_lc] = interpolation * torch.gather(
                    distances[is_non_zero_above_lc],
                    1,
                    zero_counts[is_non_zero_above_lc].view(-1, 1),
                )

    def binary_search(self, distances, n_iter=64) -> None:
        """
        Perform binary search to find optimal sigma values so that the transformed distances sum up to log_2(N)
        :param distances:
        :param n_iter:
        """
        with torch.no_grad():
            lows = torch.zeros(distances.shape[0], device=distances.device)
            highs = torch.zeros(distances.shape[0], device=distances.device) + torch.inf
            target = (
                torch.log2(torch.tensor([self.k], device=distances.device))
                * self.bandwidth
            )

            for i in range(n_iter):
                psums = torch.sum(self.evaluate(distances), axis=-1)
                errors = psums - target

                is_too_high = errors > 0

                if torch.linalg.norm(errors, torch.inf) < SMOOTH_K_TOLERANCE:
                    break

                # if psum > target
                highs[is_too_high] = self.sigma[is_too_high]
                self.sigma[is_too_high] = (lows[is_too_high] + highs[is_too_high]) / 2.0

                # if psum < target
                lows[~is_too_high] = self.sigma[~is_too_high]
                is_high_inf = highs == torch.inf
                self.sigma[(~is_too_high) & is_high_inf] *= 2
                self.sigma[(~is_too_high) & (~is_high_inf)] = (
                    lows[(~is_too_high) & (~is_high_inf)]
                    + highs[(~is_too_high) & (~is_high_inf)]
                ) / 2.0

            nan_distances = distances.clone()
            nan_distances[nan_distances == torch.inf] = torch.nan
            sample_mean_distances = torch.nanmean(nan_distances, dim=-1)
            mean_distance = torch.nanmean(nan_distances)
            is_rho_positive = self.rho > 0.0
            is_sigma_lt_sample_tol = (
                self.sigma < MIN_K_DIST_SCALE * sample_mean_distances
            )
            is_sigma_lt_overall_tol = self.sigma < MIN_K_DIST_SCALE * mean_distance
            self.sigma[is_rho_positive & is_sigma_lt_sample_tol] = (
                MIN_K_DIST_SCALE
                * sample_mean_distances[is_rho_positive & is_sigma_lt_sample_tol]
            )
            self.sigma[(~is_rho_positive) & is_sigma_lt_overall_tol] = (
                MIN_K_DIST_SCALE * mean_distance
            )

    def forward(self, distances: Tensor) -> Tensor:
        """
        Use the current rho values to evaluate corresponding smooth kNN distances
        :param distances: The kNN distances of data samples
        :return: The result of the smooth distance calculation
        """
        d = distances - self.rho.view(-1, 1)
        return torch.where(
            d > 0,
            torch.exp(-d / self.sigma.view(-1, 1)),
            torch.tensor([1.0], device=distances.device).float(),
        )

    def evaluate(self, distances: Tensor) -> Tensor:
        """
        Calculate smooth kNN distance with all distance entries, excluding self - for sigma value optimization
        :param distances:
        :return:
        """
        d = distances[:, 1:] - self.rho.view(-1, 1)
        return torch.where(
            d > 0,
            torch.exp(-d / self.sigma.view(-1, 1)),
            torch.tensor([1.0], device=distances.device).float(),
        )

    def flattened_forward(self, distances: Tensor, sample_idx: Tensor) -> Tensor:
        """
        Calculate smooth kNN distance from flattened distance values, using sample_idx to identify the sample each
        distance value was calculated from
        :param distances:
        :param sample_idx:
        :return:
        """
        d = distances - self.rho[sample_idx]
        return torch.where(
            d > 0,
            torch.exp(-d / self.sigma[sample_idx]),
            torch.tensor([1.0], device=distances.device).float(),
        )

    def test_sigma(self, distances: Tensor, external_sigma: Tensor) -> Tensor:
        d = distances[:, 1:] - self.rho.view(-1, 1)
        return torch.where(
            d > 0,
            torch.exp(-d / external_sigma.view(-1, 1)),
            torch.tensor([1.0], device=distances.device).float(),
        )


def train_smooth_knn_dist(
    distances: Tensor,
    k: float,
    max_iter: int = 64,
    local_connectivity: float = 1.0,
    bandwidth: float = 1.0,
) -> SmoothKnnDist:
    """

    :param distances:
    :param k:
    :param max_iter:
    :param local_connectivity:
    :param bandwidth:
    :return:
    """
    smooth_knn_dist = SmoothKnnDist(k, local_connectivity, bandwidth).to(
        distances.device
    )
    smooth_knn_dist.initialize_parameters(distances)
    smooth_knn_dist.binary_search(distances, max_iter)
    return smooth_knn_dist


def compute_membership_strengths(
    knn_indices: Tensor,
    knn_dists: Tensor,
    smooth_knn_dist: SmoothKnnDist,
    return_dists: bool = False,
    bipartite: bool = False,
) -> Tuple[torch.sparse.Tensor, torch.sparse.Tensor]:
    """
    Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.
    :param knn_indices: The SORTED BY DISTANCE indices on the ``n_neighbors`` closest points in the dataset.
    :param knn_dists: The SORTED distances to the ``n_neighbors`` closest points in the dataset.
    :param smooth_knn_dist: The distance metric derived from the metric tensor approximation
    :param return_dists: Whether to return the pairwise distance associated with each edge
    :param bipartite: Does the nearest neighbour set represent a bipartite graph?  That is are the
        nearest neighbour indices from the same point set as the row indices?
    :return:
    """
    n_samples = knn_indices.shape[0]
    if bipartite:
        idx, dists = knn_indices, knn_dists
    else:
        idx, dists = knn_indices[:, 1:], knn_dists[:, 1:]
    # calculate the coordinate pairs of the sparse matrix entries
    long_idx = torch.cat(
        [
            torch.cartesian_prod(
                torch.tensor([i], device=knn_indices.device).long(), row
            )
            for i, row in enumerate(idx.unbind())
        ]
    )
    sm_dists = smooth_knn_dist(dists)
    nn_map = torch.sparse_coo_tensor(
        long_idx.T, sm_dists.flatten(), size=(n_samples, n_samples)
    ).coalesce()
    dmap = None
    if return_dists:
        dmap = torch.sparse_coo_tensor(
            long_idx.T, dists.flatten(), size=(n_samples, n_samples)
        ).coalesce()
    return nn_map, dmap


def compute_membership_strengths_from_sparse_distance_matrix(
    distance_matrix: torch.sparse.Tensor,
    smooth_knn_dist: SmoothKnnDist,
    bipartite: bool = False,
) -> torch.sparse.Tensor:
    n_samples = distance_matrix.shape[0]
    # coalesced = distance_matrix.coalesce()
    idx, dists = distance_matrix.indices(), distance_matrix.values()
    if not bipartite:
        diag_ind = idx[0, :] == idx[1, :]
        idx, dists = idx[:, ~diag_ind], dists[~diag_ind]
    sm_dists = smooth_knn_dist.flattened_forward(dists, idx[0, :])
    return torch.sparse_coo_tensor(idx, sm_dists, size=(n_samples, n_samples))


def add_id_diag(G):
    diag_indices = torch.stack([torch.arange(len(G))] * 2)
    diag_values = torch.ones(len(G))
    return torch.sparse_coo_tensor(
        torch.cat([G.indices(), diag_indices], 1),
        torch.cat([G.values(), diag_values], 0),
        size=G.shape,
    ).coalesce()


def _sparse_row_wise_multiply(v, M):
    values = M.values()
    multiplier = v[M.indices()[0, :]]
    return torch.sparse_coo_tensor(
        M.indices(), values * multiplier, M.shape, device=M.device
    )


def purge_zeros_in_sparse_tensor(X, tolerance=1e-20):
    if not X.is_coalesced:
        X = X.coalesce()
    indices, values = X.indices(), X.values()
    zero_ind = values <= tolerance
    return torch.sparse_coo_tensor(
        indices[:, ~zero_ind], values[~zero_ind], size=X.shape, device=X.device
    ).coalesce()


def to_scipy_sparse(X: torch.tensor) -> coo_matrix:
    if not X.is_coalesced():
        X = X.coalesce()
    indices, values = X.indices().numpy(), X.values().numpy()
    return coo_matrix((values, (indices[0, :], indices[1, :])), shape=X.shape)
