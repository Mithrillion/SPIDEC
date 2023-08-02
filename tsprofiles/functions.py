import numpy as np
import torch
import torch.nn.functional as F
import typing
import pykeops.torch as ktorch
from pykeops.torch.cluster import from_matrix
from scipy.sparse import coo_matrix, eye, diags

from .umap_utils import (
    train_smooth_knn_dist,
    compute_membership_strengths_from_sparse_distance_matrix,
    _sparse_row_wise_multiply,
    SmoothKnnDist,
)

#############################################
# Data manipulation
#############################################


def extract_td_embeddings(
    arr: typing.Union[np.ndarray, torch.Tensor],
    delta_t: int,
    embedding_dim: int,
    skip_step: int,
    dim_order: str = "dpt",  # [(per-time) dim, (OG seq) position, (subseq) time]
) -> torch.tensor:
    source_tensor = torch.tensor(arr) if isinstance(arr, np.ndarray) else arr
    td_embedding = F.unfold(
        source_tensor.T.view(1, source_tensor.shape[1], 1, source_tensor.shape[0]),
        (1, embedding_dim),
        dilation=delta_t,
        stride=skip_step,
    ).view(source_tensor.shape[1], embedding_dim, -1)
    if dim_order == "dpt":
        td_embedding = td_embedding.permute(0, 2, 1)
    elif dim_order == "pdt":
        td_embedding = td_embedding.permute(2, 0, 1)
    elif dim_order == "dtp":
        pass
    elif dim_order == "ptd":
        td_embedding = td_embedding.permute(2, 1, 0)
    elif dim_order == "p_td":
        td_embedding = td_embedding.permute(2, 1, 0).flatten(-2, -1)
    else:
        raise ValueError("Invalid dim_order string!")
    return td_embedding


def purge_zeros_in_sparse_tensor(X: torch.Tensor, tolerance=1e-20) -> torch.Tensor:
    if not X.is_coalesced():
        X = X.coalesce()
    indices, values = X.indices(), X.values()
    zero_ind = values <= tolerance
    return torch.sparse_coo_tensor(
        indices[:, ~zero_ind], values[~zero_ind], size=X.shape, device=X.device
    ).coalesce()


def purge_large_in_sparse_tensor(X: torch.Tensor, tolerance=1) -> torch.Tensor:
    if not X.is_coalesced():
        X = X.coalesce()
    indices, values = X.indices(), X.values()
    large_ind = values >= tolerance
    return torch.sparse_coo_tensor(
        indices[:, ~large_ind], values[~large_ind], size=X.shape, device=X.device
    ).coalesce()


def to_scipy_sparse(X: torch.Tensor) -> coo_matrix:
    if not X.is_coalesced():
        X = X.coalesce()
    indices, values = X.indices().numpy(), X.values().numpy()
    return coo_matrix((values, (indices[0, :], indices[1, :])), shape=X.shape)


def zero_rows_and_cols(M, idx):
    diag = eye(M.shape[1]).tolil()
    for i in idx:
        diag[i, i] = 0
    res = diag.dot(M).dot(diag)
    res.eliminate_zeros()
    return res


def to_torch_sparse(X: coo_matrix, dtype=torch.float) -> torch.Tensor:
    return torch.sparse_coo_tensor(
        torch.stack([torch.tensor(X.row), torch.tensor(X.col)], dim=0),
        X.data,
        X.shape,
        dtype=dtype,
    ).coalesce()


def make_batches(data_size, batch_size, drop_last=False, stride=None):
    if stride is None:
        stride = batch_size
    s = np.arange(0, data_size - batch_size + stride, stride)
    e = s + batch_size
    if drop_last:
        s, e = s[e < data_size], e[e < data_size]
    else:
        s, e = s[s < data_size], e[s < data_size]
        e[-1] = data_size
    return list(zip(s, e))


#############################################
# kNN searches
#############################################


def knn_search(
    feats: torch.Tensor, k: int, dist: str = "ed"
) -> typing.Tuple[torch.tensor, torch.tensor]:
    X_i = ktorch.LazyTensor(feats[:, None, :])
    X_j = ktorch.LazyTensor(feats[None, :, :])
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        D, I = D_ij.Kmin_argKmin(K=k, dim=1)
        D = torch.sqrt(D)
    elif dist == "dot":
        D_ij = (X_i * X_j).sum(-1)
        D, I = (-D_ij).Kmin_argKmin(K=k, dim=1)
        D *= -1
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    return D, I


def exclusion_knn_search(
    feats: torch.Tensor, k: int, excl: int, dist: str = "ed"
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    X_i = ktorch.LazyTensor(feats[:, None, :])
    X_j = ktorch.LazyTensor(feats[None, :, :])
    indices = torch.arange(len(feats), device=feats.device).float()
    I_i = ktorch.LazyTensor(indices[:, None, None])
    I_j = ktorch.LazyTensor(indices[None, :, None])
    Diag_ij = float(excl) - (I_i - I_j).abs()
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
    elif dist == "dot":
        D_ij = 1 - (X_i * X_j).sum(-1)
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    D_ij = Diag_ij.ifelse(np.inf, D_ij)
    D, I = D_ij.Kmin_argKmin(K=k, dim=1)
    if dist == "ed":
        D = torch.sqrt(D)
    return D, I


def prune_by_exclusion(
    D: torch.Tensor, I: torch.Tensor, k: int, excl: int
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError()


def radius_exclusion_knn_search(
    feats: torch.Tensor, k: int, band: typing.Tuple[int, int], dist: str = "ed"
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    excl, radius = band
    X_i = ktorch.LazyTensor(feats[:, None, :])
    X_j = ktorch.LazyTensor(feats[None, :, :])
    indices = torch.arange(len(feats), device=feats.device).float()
    I_i = ktorch.LazyTensor(indices[:, None, None])
    I_j = ktorch.LazyTensor(indices[None, :, None])
    index_dists = (I_i - I_j).abs()
    Diag_ij_excl = float(excl) - index_dists
    Diag_ij_radius = index_dists - float(radius)
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
    elif dist == "dot":
        D_ij = 1 - (X_i * X_j).sum(-1)
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    D_ij = Diag_ij_excl.ifelse(np.inf, D_ij)
    D_ij = Diag_ij_radius.ifelse(np.inf, D_ij)
    D, I = D_ij.Kmin_argKmin(K=k, dim=1)
    if dist == "ed":
        D = torch.sqrt(D)
    return D, I


def _search_with_ranges(feats, k, ij_ranges, dist):
    X_i = ktorch.LazyTensor(feats[:, None, :])
    X_j = ktorch.LazyTensor(feats[None, :, :])
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        D, I = D_ij.Kmin_argKmin(k, 1, ranges=ij_ranges)
        D = torch.sqrt(D)
    elif dist == "ed_max":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        D, I = (-D_ij).Kmin_argKmin(k, 1, ranges=ij_ranges)
        D = torch.sqrt(torch.abs(D))
    elif dist == "dot":
        D_ij = (X_i * X_j).sum(-1)
        D, I = (-D_ij).Kmin_argKmin(k, 1, ranges=ij_ranges)
        D *= -1
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    return D, I


def band_knn_search(
    feats: torch.Tensor,
    k: int,
    band: typing.Tuple[int, int],
    block_size: int = 1,
    dist: str = "ed",
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    excl, radius = band
    x_ranges = torch.tensor(make_batches(len(feats), block_size)).int()
    if radius == np.inf:
        radius = len(x_ranges)
    pos_diags = np.arange(excl, min(excl + radius, len(x_ranges)))
    keep = torch.tensor(
        diags(
            [1] * len(pos_diags),
            pos_diags,
            (len(x_ranges), len(x_ranges)),
            dtype=bool,
        ).toarray()
    )
    keep = keep + keep.t()
    ij_ranges = from_matrix(x_ranges, x_ranges, keep)
    D, I = _search_with_ranges(feats, k, ij_ranges, dist)
    return D, I


# TODO: implement block-band knn search (more efficient)


def trivial_kfn_search(
    feats: torch.tensor, k: int, incl: int, dist: str = "ed"
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    """Inverted kNN search within the 'inclusion zone' to find maximum
    distance changes w.r.t. small phase drift

    Args:
        feats (torch.tensor): _description_
        k (int): _description_
        incl (int): _description_

    Returns:
        typing.Tuple[torch.Tensor, torch.Tensor]: _description_
    """
    X_i = ktorch.LazyTensor(feats[:, None, :])
    X_j = ktorch.LazyTensor(feats[None, :, :])
    indices = torch.arange(len(feats), device=feats.device).float()
    I_i = ktorch.LazyTensor(indices[:, None, None])
    I_j = ktorch.LazyTensor(indices[None, :, None])
    Diag_ij = (I_i - I_j).abs() - float(incl)
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
    elif dist == "dot":
        D_ij = 1 - (X_i * X_j).sum(-1)
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    D_ij = Diag_ij.ifelse(0, D_ij)
    D, I = (-D_ij).Kmin_argKmin(K=k, dim=1)
    D *= -1
    if dist == "ed":
        D = torch.sqrt(D)
    return D, I


def mean_dist_profile(
    feats: torch.tensor, k: int, incl: int, dist: str = "ed"
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    X_i = ktorch.LazyTensor(feats[:, None, :])
    X_j = ktorch.LazyTensor(feats[None, :, :])
    indices = torch.arange(len(feats), device=feats.device).float()
    I_i = ktorch.LazyTensor(indices[:, None, None])
    I_j = ktorch.LazyTensor(indices[None, :, None])
    Diag_ij = (I_i - I_j).abs() - float(incl)
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
    elif dist == "dot":
        D_ij = 1 - (X_i * X_j).sum(-1)
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    D_ij = Diag_ij.ifelse(0, D_ij)
    if dist == "ed":
        D = D_ij.sqrt().sum(dim=1) / len(feats)
    else:
        D = D_ij.sum(dim=1) / len(feats)
    return D


def cross_knn_search(
    A: torch.Tensor, B: torch.Tensor, k: int, dist: str = "ed"
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    X_i = ktorch.LazyTensor(A[:, None, :])
    X_j = ktorch.LazyTensor(B[None, :, :])
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        D, I = D_ij.Kmin_argKmin(K=k, dim=1)
        D = torch.sqrt(D)
    elif dist == "dot":
        D_ij = (X_i * X_j).sum(-1)
        D, I = (-D_ij).Kmin_argKmin(K=k, dim=1)
        D *= -1
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    return D, I


def cross_knn_search_index_only(
    A: torch.Tensor, B: torch.Tensor, k: int, dist: str = "ed"
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    X_i = ktorch.LazyTensor(A[:, None, :])
    X_j = ktorch.LazyTensor(B[None, :, :])
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        I = D_ij.argKmin(K=k, dim=1)
    elif dist == "dot":
        D_ij = (X_i * X_j).sum(-1)
        I = (-D_ij).argKmin(K=k, dim=1)
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    return I


#############################################
# triangular and banded-triangular kNN searches
#############################################


def tri_knn_search(
    TQ: torch.Tensor,
    T: torch.Tensor,
    k: int,
    dist: str = "ed",
    min_win=0,
    max_win=None,
    triangle="lower",
):
    X_i = ktorch.LazyTensor(TQ[:, None, :])
    X_j = ktorch.LazyTensor(T[None, :, :])
    indices = torch.arange(len(T), device=T.device).float()
    I_i = ktorch.LazyTensor(indices[:, None, None])
    I_j = ktorch.LazyTensor(indices[None, :, None])
    if max_win == np.inf:
        max_win = len(T)
    if triangle == "lower":
        diff = I_i - I_j
    elif triangle == "upper":
        diff = I_j - I_i
    else:
        raise ValueError(
            f"triangle value must be 'upper' or 'lower', but got {triangle}!"
        )
    win_LB = diff - min_win
    win_UB = max_win - diff
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        D_ij = win_LB.ifelse(D_ij, np.inf)
        D_ij = win_UB.ifelse(D_ij, np.inf)
        D, I = D_ij.Kmin_argKmin(K=k, dim=1)
        D = torch.sqrt(D)
    elif dist == "dot":
        D_ij = (X_i * X_j).sum(-1)
        D_ij = win_LB.ifelse(D_ij, -np.inf)
        D_ij = win_UB.ifelse(D_ij, -np.inf)
        D, I = (-D_ij).Kmin_argKmin(K=k, dim=1)
        D *= -1
    elif dist == "normal":
        NX_i = X_i.normalize()
        NX_j = X_j.normalize()
        S_ij = (NX_i * NX_j).sum(-1)
        D_ij = X_j.sqnorm2() * (1 - S_ij.square())
        D_ij = win_LB.ifelse(D_ij, np.inf)
        D_ij = win_UB.ifelse(D_ij, np.inf)
        D, I = D_ij.Kmin_argKmin(K=k, dim=1)
        D = torch.sqrt(D)
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    return D, I


def tri_knn_search_index_only(
    TQ: torch.Tensor,
    T: torch.Tensor,
    k: int,
    dist: str = "ed",
    min_win=0,
    max_win=None,
    triangle="lower",
):
    X_i = ktorch.LazyTensor(TQ[:, None, :])
    X_j = ktorch.LazyTensor(T[None, :, :])
    indices = torch.arange(len(T), device=T.device).float()
    I_i = ktorch.LazyTensor(indices[:, None, None])
    I_j = ktorch.LazyTensor(indices[None, :, None])
    if max_win == np.inf:
        max_win = len(T)
    if triangle == "lower":
        diff = I_i - I_j
    elif triangle == "upper":
        diff = I_j - I_i
    else:
        raise ValueError(
            f"triangle value must be 'upper' or 'lower', but got {triangle}!"
        )
    win_LB = diff - min_win
    win_UB = max_win - diff
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
        D_ij = win_LB.ifelse(D_ij, np.inf)
        D_ij = win_UB.ifelse(D_ij, np.inf)
        I = D_ij.argKmin(K=k, dim=1)
    elif dist == "dot":
        D_ij = (X_i * X_j).sum(-1)
        D_ij = win_LB.ifelse(D_ij, -np.inf)
        D_ij = win_UB.ifelse(D_ij, -np.inf)
        I = (-D_ij).argKmin(K=k, dim=1)
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    return I


#############################################
# Transitive Exclusion
#############################################


def transitive_exclusion_step(
    feats: torch.Tensor, excl: int, I_prev: torch.Tensor = None, dist: str = "ed"
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    X_i = ktorch.LazyTensor(feats[:, None, :])
    X_j = ktorch.LazyTensor(feats[None, :, :])
    indices = torch.arange(len(feats), device=feats.device).float()
    if I_prev is None:
        I_prev = indices[:, None]
    I_i = ktorch.LazyTensor(I_prev[:, None, :].float())
    I_j = ktorch.LazyTensor(indices[None, :, None])
    Diag_ij = (float(excl) - (I_i - I_j).abs()).ifelse(1, 0).sum(-1) - 0.5
    if dist == "ed":
        D_ij = ((X_i - X_j) ** 2).sum(-1)
    elif dist == "dot":
        D_ij = 1 - (X_i * X_j).sum(-1)
    else:
        raise ValueError(f"Invalid dist mode '{dist}'!")
    D_ij = Diag_ij.ifelse(np.inf, D_ij)
    D, I = D_ij.min_argmin(dim=1)
    if dist == "ed":
        D = torch.sqrt(D)
    return D, torch.cat((I_prev, I), dim=-1)


def transitive_exclusion_knn_search(
    feats: torch.Tensor,
    k: int,
    excl: int,
    dist: str = "ed",
    skip_n_excls: int = 0,
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    if excl == 0:
        return knn_search(feats, k, dist)
    if skip_n_excls > excl:
        raise ValueError(f"skip_n_excls must not be larger than k!")
    D_list = []
    for i in range(k):
        if skip_n_excls > 0 and i == 0:
            Dk, Ik = knn_search(feats, 1 + skip_n_excls, dist=dist)
            Dk = Dk[:, 1:]
        elif skip_n_excls > 0 and i < skip_n_excls:
            continue
        else:
            Dk, Ik = transitive_exclusion_step(
                feats, excl, I_prev=Ik if i != 0 else None, dist=dist
            )
        D_list += [Dk]
    return torch.cat(D_list, dim=-1), Ik[:, 1:].long()


def transitive_exclusion_kde(
    feats: torch.Tensor, I_knn: torch.Tensor, excl: int, smoothing: float
):
    X_i = ktorch.LazyTensor(feats[:, None, :])
    X_j = ktorch.LazyTensor(feats[None, :, :])
    indices = torch.arange(len(feats), device=feats.device).float()
    I_i = ktorch.LazyTensor(I_knn[:, None, :].float())
    I_j = ktorch.LazyTensor(indices[None, :, None])
    Diag_ij = ((I_i - I_j).abs() - 0.5).ifelse(1, 0).sum(-1)  # check if is a k-th NN
    # ^^ positive if not matching any NN's index, otherwise 0
    Excl_ij = (float(excl) - (I_i - I_j).abs()).ifelse(1, 0).sum(-1)
    # ^^ positive if within exlcusion radius of any NN, otherwise 0
    D_ij = ((X_i - X_j) ** 2).sum(-1)
    D_ij = (Diag_ij * Excl_ij - 0.5).ifelse(np.inf, D_ij)
    D = (-D_ij / smoothing).exp().sum(1)
    return D


def mpdist_exclusion_knn_search(
    feats: torch.Tensor,
    k: int,
    bag_size: int,
    tau: int = 1,
    skip_step: int = 1,
    quantile: float = 0,
    skip_n_excls: int = 0,
    dist: str = "ed",
):
    pth_smallest = int(bag_size * quantile)
    # first compute subseq knn dists
    D_s, I_s = transitive_exclusion_knn_search(
        feats, k, bag_size, skip_n_excls=skip_n_excls, dist=dist
    )
    # break knn dist array into sliding windows
    D_s_sets, I_s_sets = extract_td_embeddings(
        D_s, tau, bag_size, skip_step, "ptd"
    ), extract_td_embeddings(I_s.float(), tau, bag_size, skip_step, "ptd")
    # now the format is (metaseq position, subseq position, knn list)
    # now pre-allocate mpdist and index arrays
    D_m, I_m = (
        torch.zeros((len(D_s_sets), k)).float().to(feats.device),
        torch.zeros((len(D_s_sets), k)).int().to(feats.device)
        - 2
        * bag_size,  # ensure no match on initial run (can be arbitrary "impossible" value)
    )
    mask = torch.zeros_like(I_s_sets).bool().to(feats.device)
    for i in range(k):
        diff = I_m[:, None, None, :] - I_s_sets[:, :, :, None]
        mask = mask | torch.any(
            # (diff < bag_size // skip_step) & (diff >= -bag_size // skip_step // 4),
            torch.abs(diff) < bag_size // skip_step,
            -1,
        )
        # mask = mask | torch.any(
        #     torch.abs(I_m[:, None, None, :] - I_s_sets[:, :, :, None])
        #     < bag_size // skip_step,
        #     -1,
        # )
        # ^^ check for matching destination overlap

        masked_D_s_sets = torch.where(
            ~mask, D_s_sets, torch.inf
        )  # (metaseq, subseq, knn)
        match_choice_set, knn_indices = torch.min(
            masked_D_s_sets, dim=2
        )  # (metaseq, subseq)
        # knn_indices records which knn is the current unmasked smallest per metaseq per subseq

        if quantile == 0:
            D_m[:, i], min_subseq_pos = torch.min(match_choice_set, dim=1)
            # min_subseq_pos records which subseq produces the smallest dist per metaseq
            # mask = mask | (D_s_sets <= D_m.amax(-1)[:, None, None])
        else:
            top_d, top_i = torch.topk(
                match_choice_set, pth_smallest, dim=1, largest=False
            )  # (metaseq)
            D_m[:, i], min_subseq_pos = top_d[..., -1], top_i[..., -1]

        # we still need the kth NN indices
        min_subseq_knn_indices = knn_indices.gather(
            1, min_subseq_pos[:, None]
        )  # (metaseq)
        I_m[:, i] = (
            I_s_sets.gather(1, min_subseq_pos[:, None, None].expand(-1, -1, k))
            .squeeze()
            .gather(1, min_subseq_knn_indices)
            .squeeze()
        )  # (metaseq)

    return D_m, I_m.long()


#############################################
# kNN graph processing
#############################################


def remove_temporal_drift_from_knn(
    D: torch.tensor,
    I: torch.tensor,
    exclusion_radius: int,
    max_neighbours: int = None,
    device: torch.device = None,
    return_best_match_indicator: bool = False,
    is_affinity: bool = False,
) -> torch.tensor:
    """
    function for removing temporal drift matches (generalised trival matches)

    Parameters
    ----------
    D : torch.tensor
        _description_
    I : torch.tensor
        _description_
    exclusion_radius : int
        _description_
    normalise : bool, optional
        _description_, by default False
    self_transition : bool, optional
        _description_, by default True
    max_neighbours : int, optional
        _description_, by default None
    device : torch.device, optional
        _description_, by default None
    return_best_match_indicator: bool, optional
        _description_, by default False
    is_affinity: bool, optional
        _description_, by default False
    Returns
    -------
    torch.tensor
        _description_
    """
    dev = D.device if device is None else device
    nn_indices, nn_distances = I.to(dev), D.to(dev)
    temporal_drift_mask = torch.zeros_like(nn_indices, dtype=torch.bool, device=dev)
    best_match_indicator = torch.zeros_like(nn_indices, dtype=torch.bool, device=dev)
    current_best_match_indices = torch.arange(
        nn_indices.shape[0], dtype=torch.long, device=dev
    )  # start with the current positions

    def is_in_exclusion_radius(index, current):
        return (index > (current - exclusion_radius).unsqueeze(1)) & (
            index < (current + exclusion_radius).unsqueeze(1)
        )

    for i in range(nn_indices.shape[1] + 1):
        # flag all index positions inside the current exclusion zone (including best matches)
        temporal_drift_mask[
            is_in_exclusion_radius(nn_indices, current_best_match_indices)
        ] = 1
        # mark the best match positions to differentiate from temporal drift points
        best_match_indicator[nn_indices == current_best_match_indices.unsqueeze(1)] = 1
        # count the number of masked points per row
        num_masked_per_position = torch.sum(temporal_drift_mask, dim=-1)
        # check if an entire row is masked
        is_fully_masked = num_masked_per_position >= nn_indices.shape[1]
        # if all fully masked, break
        if torch.all(is_fully_masked):
            break
        # find the smallest knn distance among unmasked
        if is_affinity:
            current_best_match_pos = torch.argmax(
                nn_distances - torch.inf * temporal_drift_mask,
                dim=-1,
                keepdim=True,
            )
        else:
            current_best_match_pos = torch.argmin(
                nn_distances + torch.inf * temporal_drift_mask,
                dim=-1,
                keepdim=True,
            )
        current_best_match_indices = torch.gather(
            nn_indices, 1, current_best_match_pos
        ).squeeze()
        if max_neighbours is not None and i >= max_neighbours:
            break
    # construct a sparse pairwise distance matrix to represent the kNN graph
    best_matches_per_position = torch.sum(best_match_indicator, dim=-1)
    position_from = torch.repeat_interleave(
        torch.arange(nn_indices.shape[0], device=dev),
        best_matches_per_position,
    )
    sparse_dist_mat = torch.sparse_coo_tensor(
        torch.stack([position_from, nn_indices[best_match_indicator]], dim=0),
        nn_distances[best_match_indicator],
        size=(len(nn_indices), len(nn_indices)),
        device=dev,
    ).coalesce()
    sparse_dist_mat = purge_zeros_in_sparse_tensor(sparse_dist_mat)
    if return_best_match_indicator:
        return sparse_dist_mat, best_match_indicator
    return sparse_dist_mat


def find_local_density(
    D: torch.tensor,
    k: int = None,
    local_connectivity: float = 1.0,
    bandwidth: float = 1.0,
    n_iter: int = 64,
    device: torch.device = None,
):
    """
    Find local data density to be used in UMAP-related calculations

    Parameters
    ----------
    D : torch.tensor
        _description_
    local_connectivity : float, optional
        _description_, by default 1.0
    bandwidth : float, optional
        _description_, by default 1.0
    k : int, optional
        _description_, by default None
    n_iter : int, optional
        _description_, by default 64
    device : torch.device, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    dev = D.device if device is None else device
    nn_distances = D.to(dev)
    if k is None:
        k = D.shape[1]
    smooth_knn_dists = train_smooth_knn_dist(
        nn_distances.to(dev),
        k,
        n_iter,
        local_connectivity,
        bandwidth,
    )
    return smooth_knn_dists


def knn_entries_to_sparse_dists(D: torch.tensor, I: torch.tensor) -> torch.tensor:
    dev = D.device
    position_from = torch.repeat_interleave(
        torch.arange(I.shape[0], device=dev),
        I.shape[1],
    )
    sparse_dist_mat = torch.sparse_coo_tensor(
        torch.stack([position_from, I.flatten()], dim=0),
        D.flatten(),
        device=dev,
    ).coalesce()
    return sparse_dist_mat


def knn_entries_to_scipy_sparse(D: torch.tensor, I: torch.tensor) -> torch.tensor:
    position_from = torch.repeat_interleave(
        torch.arange(I.shape[0]),
        I.shape[1],
    )
    sparse_dist_mat = coo_matrix(
        (D.flatten(), (position_from, I.flatten())), (I.shape[0], I.shape[0])
    )
    return sparse_dist_mat


def sparse_distance_to_membership(
    sparse_dist_mat: torch.tensor,
    smooth_knn_dists: torch.nn.Module,
    bipartite: bool = True,
    normalise: bool = False,
) -> torch.tensor:
    """
    Calculate a sparse membership matrix from sparse distance matrix
    according to the UMAP algorithm's formulation

    Parameters
    ----------
    sparse_dist_mat : torch.tensor
        _description_
    smooth_knn_dists : torch.Module
        _description_
    bipartite : bool, optional
        _description_, by default True
    normalise : bool, optional
        _description_, by default False

    Returns
    -------
    torch.tensor
        _description_
    """
    sparse_membership_mat = compute_membership_strengths_from_sparse_distance_matrix(
        sparse_dist_mat, smooth_knn_dists, bipartite=bipartite
    ).coalesce()
    sparse_membership_mat = purge_zeros_in_sparse_tensor(sparse_membership_mat)
    if normalise:
        normaliser = torch.sparse.sum(sparse_membership_mat, dim=-1).to_dense()
        normaliser[normaliser < 1] = 1
        return (
            _sparse_row_wise_multiply(1 / normaliser, sparse_membership_mat)
            .t()
            .coalesce()
        )
    else:
        return sparse_membership_mat.t().coalesce()


def temporal_link(
    nn_graph: torch.tensor,
    diag: list = None,
    link_strengths: list = None,
    zero_idx: list = None,
) -> torch.tensor:
    if diag is None:
        diag = [-1, 1]
    scipy_mat = to_scipy_sparse(nn_graph)
    if link_strengths is None:
        temporal_diags = diags([1] * len(diag), diag, shape=nn_graph.shape)
    else:
        temporal_diags = diags(link_strengths, diag, shape=nn_graph.shape)
    if zero_idx is not None:
        temporal_diags = zero_rows_and_cols(temporal_diags, zero_idx)
    sym_graph = scipy_mat + temporal_diags
    return to_torch_sparse(sym_graph.tocoo())


def sparse_make_symmetric(X: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
    return (X + X.t() - X * X.t()).coalesce()


def invert_affinity(G: torch.sparse_coo_tensor) -> torch.sparse_coo_tensor:
    vals = 1 - G.values()
    return torch.sparse_coo_tensor(G.indices(), vals, G.shape)
