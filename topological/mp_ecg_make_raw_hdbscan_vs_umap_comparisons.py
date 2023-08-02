# %%
import os
import sys
import numpy as np
import pandas as pd
import torch
from functools import reduce
import operator

import holoviews as hv

from scipy.io import arff, loadmat

# from pyts.transformation import ROCKET

hv.extension("bokeh")

sys.path.append("..")
sys.path.append("../..")

from tsprofiles.functions import *
from sigtools.transforms import *
from rocket.rocket import Rocket
import hdbscan

# from ripser import Rips
# import persim

# import matplotlib.pyplot as plt

from umap_torch.nonparametric_umap import (
    simplicial_set_embedding,
    compute_membership_strengths,
    add_id_diag,
)
from topological.utils.find_exemplar_ids import find_exemplar_ids

hv.opts.defaults(hv.opts.Curve(width=700, height=200))


def get_closest(array, values):
    # make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = (idxs == len(array)) | (
        np.fabs(values - array[np.maximum(idxs - 1, 0)])
        < np.fabs(values - array[np.minimum(idxs, len(array) - 1)])
    )
    idxs[prev_idx_is_less] -= 1

    return array[idxs], idxs


# %%
DATASET_ROOT = "../data/mpsubseq/"
cls_filename_list = os.listdir(DATASET_ROOT)
# %%
dat = loadmat(
    f"{DATASET_ROOT}/106.mat",
    squeeze_me=True,
    chars_as_strings=True,
    struct_as_record=True,
    simplify_cells=True,
)
# %%
arr = dat["data"]
coord = dat["coord"]
coordIdx = dat["coordIdx"]
coordLab = dat["coordLab"]
# %%
closest_idx, closest_pos = get_closest(coordIdx, np.arange(len(arr)))
closest_lab = coordLab[closest_pos]
is_closest_abnormal = closest_lab != 3
# %%
is_not_N = coordLab != 3
g = hv.Curve(arr) + hv.Spikes({"x": coordIdx[is_not_N], "c": coordLab[is_not_N]}).opts(
    width=700, height=100, color="c"
)
# g.cols(1)
# %%
# l = 178
# ds_rate = l
# excl = max(1, l // 4 // ds_rate)

l = 1024
ds_rate = l // 2
excl = max(1, l // 4 // ds_rate)

soft_interval_labels = (
    pd.Series(is_closest_abnormal[::-1])
    .rolling(l, l, False)
    .mean()[::-1]
    .dropna()
    .values[::ds_rate]
)
# %%
trace_tensor = torch.tensor(arr.astype(np.float32))[:, None]
trace_subseqs = extract_td_embeddings(trace_tensor, 1, l, ds_rate, "p_td")
trace_subseqs = znorm(trace_subseqs, -1)
# %%
# trace_feats = trace_subseqs.contiguous()

n_kernels = 100
rocket = Rocket(num_kernels=n_kernels, normalise=False, n_jobs=-1, random_state=7777)
trace_feats = torch.tensor(
    rocket.fit_transform(trace_subseqs.numpy()[:, None, :])
).float()
trace_feats = trace_feats[:, :].contiguous()  # second feature per kernel is the max
# %%
# try HDBSCAN now
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=0)
clusterer.fit(trace_feats)
# %%
pd.Series(clusterer.labels_).value_counts()
# %%
selection = (clusterer.labels_ != -1) & (clusterer.probabilities_ > 0.95)
(
    g
    + hv.Spikes(
        {
            "x": np.arange(ds_rate * len(trace_feats), step=ds_rate)[selection],
            "c": clusterer.labels_[selection],
        },
        kdims="x",
        vdims="c",
    ).opts(width=700, height=100, color="c", spike_length=1, cmap="Set2")
).cols(1)
# %%
# find exemplars
exemplars = find_exemplar_ids(clusterer)
# %%
for i, ex in enumerate(exemplars):
    hv.output(
        reduce(
            operator.mul,
            [hv.Curve(trace_subseqs[c, :], label=f"cluster = {i}") for c in ex],
        )
    )
# %%
k = 20
D, I = transitive_exclusion_knn_search(trace_feats, k, excl)
I = I.long()
skd = find_local_density(D, k, local_connectivity=1, bandwidth=1)
G, SD = compute_membership_strengths(I, D, skd, True, True)
G = add_id_diag(G)
G = sparse_make_symmetric(G)
# G = temporal_link(G, [-1, 1], [1 / k, 1 / k])
emb, _ = simplicial_set_embedding(
    G.cuda(),
    2,
    batch_size=64,
    initial_alpha=1,
    random_state=7777,
    min_dist=0.1,
    n_epochs=500,
    gamma=0.1,
)
# %%
umap_clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
umap_clusterer.fit(emb.cpu())
# %%
pd.Series(umap_clusterer.labels_).value_counts()
# %%
(
    hv.Scatter(
        {
            "emb1": emb[:, 0].cpu(),
            "emb2": emb[:, 1].cpu(),
            # "label": soft_interval_labels[density > cutoff][: len(emb)],
            "cluster": clusterer.labels_,
            "label": soft_interval_labels[: len(emb)],
        },
        "emb1",
        ["emb2", "label", "cluster"],
    ).opts(width=500, height=500, color="label", alpha=0.5, cmap="bkr", colorbar=True)
    + hv.Scatter(
        {
            "emb1": emb[:, 0].cpu(),
            "emb2": emb[:, 1].cpu(),
            # "label": soft_interval_labels[density > cutoff][: len(emb)],
            "cluster": clusterer.labels_,
            "label": soft_interval_labels[: len(emb)],
        },
        "emb1",
        ["emb2", "label", "cluster"],
    ).opts(
        width=500,
        height=500,
        color="cluster",
        alpha=0.5,
        cmap="Set2",
        colorbar=True,
    )
    + hv.Scatter(
        {
            "emb1": emb[:, 0].cpu(),
            "emb2": emb[:, 1].cpu(),
            # "label": soft_interval_labels[density > cutoff][: len(emb)],
            "umap_cluster": umap_clusterer.labels_,
            "label": soft_interval_labels[: len(emb)],
        },
        "emb1",
        ["emb2", "label", "umap_cluster"],
    ).opts(
        width=500,
        height=500,
        color="umap_cluster",
        alpha=0.5,
        cmap="Category20",
        colorbar=True,
    )
).cols(2)

# %%
selection = (umap_clusterer.labels_ != -1) & (umap_clusterer.probabilities_ > 0.95)
(
    g
    + hv.Spikes(
        {
            "x": np.arange(ds_rate * len(trace_feats), step=ds_rate)[selection],
            "c": umap_clusterer.labels_[selection],
        },
        kdims="x",
        vdims="c",
    ).opts(width=700, height=100, color="c", spike_length=1, cmap="Set2")
).cols(1)

# %%
umap_exemplars = find_exemplar_ids(umap_clusterer)
for i, ex in enumerate(umap_exemplars):
    hv.output(
        reduce(
            operator.mul,
            [hv.Curve(trace_subseqs[c, :], label=f"cluster = {i}") for c in ex],
        )
    )
# %%
selection1 = (clusterer.labels_ != -1) & (clusterer.probabilities_ > 0.9)
selection2 = (umap_clusterer.labels_ != -1) & (umap_clusterer.probabilities_ > 0)
g2 = (
    # hv.Curve(arr).opts(width=1400, height=300, title="ECG Waveform", fontscale=2)
    # + hv.Spikes({"x": coordIdx[is_not_N], "c": coordLab[is_not_N]}).opts(
    #     width=1400,
    #     height=100,
    #     color="c",
    #     title="V (Premature Ventricular Contractions)",
    #     xaxis="bare",
    #     yaxis="bare",
    #     fontscale=2,
    # )
    hv.Spikes(
        {
            "x": np.arange(ds_rate * len(trace_feats), step=ds_rate)[selection1],
            "c1": clusterer.labels_[selection1],
        },
        kdims="x",
        vdims="c1",
    ).opts(
        width=1400,
        height=100,
        color="c1",
        spike_length=1,
        cmap="Set2",
        title="Quick Density-based Exemplars",
        xaxis="bare",
        yaxis="bare",
        fontscale=2,
    )
    + hv.Spikes(
        {
            "x": np.arange(ds_rate * len(trace_feats), step=ds_rate)[selection2],
            "c2": umap_clusterer.labels_[selection2],
        },
        kdims="x",
        vdims="c2",
    ).opts(
        width=1400,
        height=100,
        color="c2",
        spike_length=1,
        cmap="Set2",
        title="Density-based Subsequence Clustering",
        xaxis="bare",
        yaxis="bare",
        fontscale=2,
    )
).cols(1)
# %%
g2
# %%
# hv.save(g2, "plots/ecg_improved.png")
# %%
DATASET_ROOT = "../data/mpsubseq/"
subseq_dat = loadmat(
    f"{DATASET_ROOT}/heartbeat2.mat",
    squeeze_me=True,
    chars_as_strings=True,
    struct_as_record=True,
    simplify_cells=True,
)
subseq_dat_ex = loadmat(
    f"{DATASET_ROOT}/heartbeat2_c10_b8.mat",
    squeeze_me=True,
    chars_as_strings=True,
    struct_as_record=True,
    simplify_cells=True,
)
# %%
cutoff = np.where(np.diff(subseq_dat_ex["idxBitsize"]) > 0)[0][0]
subseq_idx = subseq_dat_ex["idxList"][:cutoff]
# %%
sort_order = np.argsort(subseq_dat_ex["idxList"])
is_not_N = coordLab != 3
g3 = (
    hv.Curve(subseq_dat["data"], "x", "value", label="ECG Waveform").opts(
        width=1400, height=300, fontscale=2
    )
    + hv.Spikes({"x": coordIdx[is_not_N], "c": coordLab[is_not_N]})
    .opts(width=1400, height=100, color="c", fontscale=2, yaxis="bare", xaxis="bare")
    .opts(title="V (Premature Ventricular Contractions)")
    + hv.Spikes(subseq_idx).opts(
        width=1400,
        height=100,
        title="Subsequences Selected by MDL",
        fontscale=2,
        yaxis="bare",
        xaxis="bare",
    )
).cols(1)
# %%
g4 = (g3 + g2).cols(1)
# %%
g4
# %%
hv.save(g4, "plots/mdl_combined.png")
# %%
