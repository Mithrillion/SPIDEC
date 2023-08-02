import torch
import torch.nn as nn
import torch.distributions as distro
from torch import Tensor
from typing import Tuple

import numpy as np

# from umap.spectral import spectral_layout
from scipy.sparse import coo_matrix

from .layouts import optimize_layout_euclidean

try:
    import pykeops.torch as ktorch

    NO_KEOPS = False
except ImportError:
    NO_KEOPS = True

# TODO: allow more flexible implementation of alternative NN algorithms

INT32_MIN = np.iinfo(np.int32).min + 1
INT32_MAX = np.iinfo(np.int32).max - 1
SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf

DISCONNECTION_DISTANCES = {
    "correlation": 2,
    "cosine": 2,
    "hellinger": 1,
    "jaccard": 1,
    "dice": 1,
}

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


def prune_small_nns(
    SD: torch.Tensor, D: torch.Tensor, cutoff: float, includes_self: bool = False
):
    first_nns = D[:, 0] if includes_self else D[:, 1]
    indices, values = SD.indices(), SD.values()
    mask = (values > cutoff) & (values != 0) & (SD.values() > D[SD.indices()[0, :], 0])
    return torch.sparse_coo_tensor(
        indices[:, ~mask], values[~mask], SD.shape
    ).coalesce()


def add_id_diag(G: torch.Tensor):
    diag_indices = torch.stack([torch.arange(len(G))] * 2)
    diag_values = torch.ones(len(G))
    return torch.sparse_coo_tensor(
        torch.cat([G.indices(), diag_indices], 1),
        torch.cat([G.values(), diag_values], 0),
        size=G.shape,
    ).coalesce()


def add_zero_diag(G: torch.Tensor):
    diag_indices = torch.stack([torch.arange(len(G))] * 2)
    diag_values = torch.zeros(len(G))
    return torch.sparse_coo_tensor(
        torch.cat([G.indices(), diag_indices], 1),
        torch.cat([G.values(), diag_values], 0),
        size=G.shape,
    ).coalesce()


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


def fuzzy_simplicial_set(
    n_neighbors: int,
    knn_indices: Tensor = None,
    knn_dists: Tensor = None,
    distance_matrix: Tensor = None,
    set_op_mix_ratio: float = 1.0,
    local_connectivity: float = 1.0,
    apply_set_operations: bool = True,
    return_dists: bool = False,
    precomputed_smooth_knn_dist: SmoothKnnDist = None,
) -> Tuple[torch.sparse.Tensor, SmoothKnnDist, torch.sparse.Tensor]:
    """
    Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.
    :param n_neighbors:
    :param knn_indices:
    :param knn_dists:
    :param distance_matrix:
    :param set_op_mix_ratio:
    :param local_connectivity:
    :param apply_set_operations:
    :param return_dists:
    :param precomputed_smooth_knn_dist:
    :return:
    """
    if ((knn_indices is None) | (knn_dists is None)) & (distance_matrix is None):
        raise NotImplementedError()
    elif ((knn_indices is None) | (knn_dists is None)) & (distance_matrix is not None):
        # extract knn_indices and knn_dists from distance matrix and sort by distance
        raise NotImplementedError()

    if precomputed_smooth_knn_dist:
        smooth_knn_dist = precomputed_smooth_knn_dist
    else:
        smooth_knn_dist = train_smooth_knn_dist(
            knn_dists, n_neighbors, local_connectivity=local_connectivity, bandwidth=1
        )
    nn_map, dist_map = compute_membership_strengths(
        knn_indices, knn_dists, smooth_knn_dist, return_dists=True, bipartite=False
    )
    if apply_set_operations:
        transpose = nn_map.t()
        prod_mat = nn_map * transpose
        nn_map = (
            set_op_mix_ratio * (nn_map + transpose - prod_mat)
            + (1 - set_op_mix_ratio) * prod_mat
        ).coalesce()

    # TODO: eliminate zero?
    nn_map = purge_zeros_in_sparse_tensor(nn_map)

    if (not return_dists) | (return_dists is None):
        return nn_map, smooth_knn_dist
    else:
        dist_map = purge_zeros_in_sparse_tensor(dist_map)
        return nn_map, smooth_knn_dist, dist_map


"""
Calculating embeddings
"""


def simplicial_set_embedding(
    graph: Tensor,
    n_components: int,
    initial_alpha: float = 1.0,
    batch_size: int = 64,
    min_dist: float = 0.1,
    a: float = None,
    b: float = None,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    n_epochs: int = None,
    prune_graph: bool = False,
    init: str = "spectral",
    random_state: int = 1111,
    metric: str = "euclidean",
    metric_kwds: dict = {},
    densmap: bool = False,
    densmap_kwds: dict = {},
    output_dens: bool = False,
    output_metric="euclidean",
    output_metric_kwds={},
    euclidean_output=True,
    push_tail: bool = True,
    tqdm_kwds: dict = {},
    optim_kwds: dict = {},
    scheduler_kwds: dict = {},
) -> Tuple[Tensor, dict]:
    """
    Perform a fuzzy simplicial set embedding, using a specified
    initialisation method and then minimizing the fuzzy set cross entropy
    between the 1-skeletons of the high and low dimensional fuzzy simplicial
    sets.
    :param graph: The 1-skeleton of the high dimensional fuzzy simplicial set as
        represented by a graph for which we require a sparse matrix for the
        (weighted) adjacency matrix.
    :param n_components: The dimensionality of the euclidean space into which to embed the data.
    :param initial_alpha: Initial learning rate for the SGD.
    :param batch_size:
    :param min_dist:
    :param a: Parameter of differentiable approximation of right adjoint functor
    :param b: Parameter of differentiable approximation of right adjoint functor
    :param gamma: Weight to apply to negative samples.
    :param negative_sample_rate: The number of negative samples to select per positive sample
        in the optimization process. Increasing this value will result
        in greater repulsive force being applied, greater optimization
        cost, but slightly more accuracy.
    :param n_epochs: The number of training epochs to be used in optimizing the
        low dimensional embedding. Larger values result in more accurate
        embeddings. If 0 is specified a value will be selected based on
        the size of the input dataset (200 for large datasets, 500 for small).
    :param prune_graph:
    :param init: How to initialize the low dimensional embedding. Options are:
            * 'spectral': use a spectral embedding of the fuzzy 1-skeleton
            * 'random': assign initial embedding positions at random.
            * A numpy array of initial embedding positions.
    :param random_state: A state capable being used as a numpy random state.
    :param metric: The metric used to measure distance in high dimensional space; used if
        multiple connected components need to be layed out.
    :param metric_kwds: Key word arguments to be passed to the metric function; used if
        multiple connected components need to be layed out.
    :param densmap: Whether to use the density-augmented objective function to optimize
        the embedding according to the densMAP algorithm.
    :param densmap_kwds: Key word arguments to be used by the densMAP optimization.
    :param output_dens: Whether to output local radii in the original data and the embedding.
    :param output_metric: Function returning the distance between two points in embedding space and
        the gradient of the distance wrt the first argument.
    :param output_metric_kwds: Key word arguments to be passed to the output_metric function.
    :param euclidean_output: Whether to use the faster code specialised for euclidean output metrics
    :param tqdm_kwds:
    :return:
    """
    # TODO: Ignoring densmap for now. Implement later
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)
    if not graph.is_coalesced():
        graph = graph.coalesce()
    n_samples = graph.shape[0]

    # TODO: adapt number of epochs to PyTorch optimizers
    if graph.shape[0] <= 10000:
        default_epochs = 500
    else:
        default_epochs = 200

    if n_epochs is None:
        n_epochs = default_epochs

    # original code used some epoch-based pruning here... why?
    if (n_epochs > 10) & prune_graph:
        pruned_indicators = graph.values() < graph.values().max() / n_epochs
        pruned_indices, pruned_values = (
            graph.indices()[:, ~pruned_indicators],
            graph.values()[~pruned_indicators],
        )
        graph = torch.sparse_coo_tensor(
            pruned_indices, pruned_values, device=graph.device
        ).coalesce()

    # TODO: implement different types of initializations
    if isinstance(init, str) and init == "random":
        # random init
        raise NotImplementedError()
    elif isinstance(init, str) and init == "spectral":
        # assume fully connected, so skipping connected component analysis
        initialisation = _simplified_spectral_embedding(
            graph, n_components, random_state=random_state
        )
        # initialisation = torch.tensor(
        #     spectral_layout(
        #         None,
        #         to_scipy_sparse(graph.cpu()),
        #         n_components,
        #         np.random.RandomState(random_state),
        #     ),
        #     device=graph.device,
        # )
        expansion = 10.0 / torch.abs(initialisation).max()
        embedding = initialisation * expansion + distro.normal.Normal(0, 1e-4).sample(
            (n_samples, n_components)
        ).to(graph.device)
    else:
        embedding = init

    # epoch_budget_per_sample = _make_epochs_per_sample(graph.values(), n_epochs)

    # TODO: skipped densmap implementation
    aux_data = {}
    if densmap or output_dens:
        raise NotImplementedError()

    # converting all to positive - why?
    embedding = (
        10.0
        * (embedding - torch.amin(embedding, 0))
        / (torch.amax(embedding, 0) - torch.amin(embedding, 0))
    )

    # TODO: optimize output with other methods

    if euclidean_output:
        embedding = optimize_layout_euclidean(
            embedding,
            graph,
            n_epochs,
            batch_size=batch_size,
            initial_alpha=initial_alpha,
            negative_sample_rate=negative_sample_rate,
            tqdm_kwds=tqdm_kwds,
            push_tail=push_tail,
            layout_model_kwds={"min_dist": min_dist, "a": a, "b": b, "gamma": gamma},
            random_state=random_state,
            optim_kwds=optim_kwds,
            scheduler_kwds=scheduler_kwds,
        )
    else:
        raise NotImplementedError()
    if output_dens:
        raise NotImplementedError()

    return embedding, aux_data


def _simplified_spectral_embedding(
    graph: Tensor, n_components: int, random_state: int = 1111
) -> Tensor:
    torch.manual_seed(random_state)
    torch.cuda.manual_seed(random_state)
    n_samples = graph.shape[0]
    sample_similarity_sum = graph.mv(torch.ones(n_samples, device=graph.device))
    diag_idx = torch.stack(
        [
            torch.arange(n_samples, device=graph.device),
            torch.arange(n_samples, device=graph.device),
        ]
    )
    I = torch.sparse_coo_tensor(
        diag_idx, torch.ones(n_samples), graph.shape, device=graph.device
    )
    v = 1.0 / torch.sqrt(sample_similarity_sum)
    L = (
        -_sparse_row_wise_multiply(
            v, _sparse_row_wise_multiply(v, graph).t().coalesce()
        ).t()
        + I
    ).coalesce()
    k = n_components + 1
    eigs, eigvecs = torch.lobpcg(
        L,
        X=distro.normal.Normal(0, 1).sample((L.shape[0], k)).to(L.device),
        n=k,
        largest=False,
        tol=1e-8,
    )
    # eigs, eigvecs = torch.linalg.eigh(L)
    order = torch.argsort(eigs)[1:k]
    return eigvecs[:, order]


def _make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.
    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights ofhow much we wish to sample each 1-simplex.
    n_epochs: int
        The total number of epochs we want to train for.
    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * torch.ones(weights.shape[0], device=weights.device)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = n_epochs / n_samples[n_samples > 0]
    return result


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


"""
Set arithmetics
"""

# TODO: implement set operations

"""
Embedding new data
"""
# TODO: torch.sparse does not have cuda support for CSR format, so maybe convert everything to COO?
# TODO: implement embedding new data

"""
UMAP class
"""

# TODO: package everything into the new UMAP class

"""
End
"""
