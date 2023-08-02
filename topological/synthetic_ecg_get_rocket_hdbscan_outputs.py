# loading ECG data
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

import holoviews as hv
import hdbscan
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    # silhouette_score,
    homogeneity_completeness_v_measure,
)

hv.extension("bokeh")

sys.path.append("..")
sys.path.append("../..")

from tsprofiles.functions import *
from sigtools.transforms import *
from topological.utils.data_loader import load_sim_ecg
from tqdm import tqdm
from rocket.rocket import Rocket
from scipy.stats import mode

hv.opts.defaults(hv.opts.Curve(width=700, height=200))

DATASET_ROOT = "../data/synthetic/"
output_file = Path("outputs/synthetic_ecg_rocket_hdbscan.csv")

if not output_file.is_file():
    header = pd.DataFrame(
        columns=[
            "id",
            "l_s",
            "skip_step",
            "ari",
            "ami",
            "homogeneity",
            "completeness",
            "v_measure",
        ]
    )
    header.to_csv(output_file.absolute(), index=False)

l_s = 200 * 2
skip_step = 8

for id in tqdm(range(0, 200)):
    ecg, af_bounds, af_ind, targets = load_sim_ecg(DATASET_ROOT, id)

    soft_interval_labels = (
        pd.Series(af_ind[::-1])
        .rolling(l_s, 1, False)
        .mean()[::-1]
        .dropna()
        .values[::skip_step]
    )

    trace_tensor = torch.tensor(ecg["II"].astype(np.float32))[:, None]
    trace_subseqs = extract_td_embeddings(trace_tensor, 1, l_s, skip_step, "p_td")
    trace_subseqs = znorm(trace_subseqs, -1).contiguous()

    td_labels, _ = mode(
        extract_td_embeddings(torch.tensor(af_ind)[:, None], 1, l_s, skip_step, "p_td"),
        -1,
        keepdims=False,
    )
    td_labels = td_labels.flatten().astype(int)

    rocket = Rocket(num_kernels=200, normalise=False, n_jobs=-1, random_state=7777)
    trace_subseqs = rocket.fit_transform(trace_subseqs[:, None, :].numpy()).squeeze()[
        :, ::2
    ]

    g1 = (
        hv.Curve(ecg, "index", "II")
        + hv.Curve(
            (ecg.index[::skip_step], soft_interval_labels),
            "index",
            "label",
        ).opts(height=100)
    ).cols(1)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=20)
    clusterer.fit(trace_subseqs)
    # pd.Series(clusterer.labels_).value_counts()

    gt_labels = td_labels[: len(clusterer.labels_)]
    ari = adjusted_rand_score(gt_labels, clusterer.labels_)
    ami = adjusted_mutual_info_score(gt_labels, clusterer.labels_)
    hmg, cpl, vm = homogeneity_completeness_v_measure(
        gt_labels, clusterer.labels_, beta=0.5
    )
    scores = pd.DataFrame(
        [
            pd.Series(
                {
                    "id": id,
                    "l_s": l_s,
                    "skip_step": skip_step,
                    # "silhouette": sil,
                    "ari": ari,
                    "ami": ami,
                    "homogeneity": hmg,
                    "completeness": cpl,
                    "v_measure": vm,
                }
            )
        ]
    )
    scores.to_csv(output_file.absolute(), index=False, mode="a", header=False)

    g3 = (
        g1
        + hv.Spikes(
            {
                "index": ecg.index[::skip_step][: len(clusterer.labels_)],
                "cluster": clusterer.labels_,
            },
            "index",
            "cluster",
        ).opts(
            width=700, height=100, spike_length=1, color="cluster", cmap="Category20"
        )
    ).cols(1)

    hv.save(g3, f"plots/synthetic_ecg_rocket_hdbscan/{id}_segment.png")
