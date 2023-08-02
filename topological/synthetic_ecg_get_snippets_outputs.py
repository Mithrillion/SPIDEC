# loading ECG data
import sys
from functools import reduce
import operator
import holoviews as hv
from tqdm import tqdm
import stumpy
from pathlib import Path
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score,
    homogeneity_completeness_v_measure,
)

hv.extension("bokeh")

sys.path.append("..")
sys.path.append("../..")

from tsprofiles.functions import *
from sigtools.transforms import *
from topological.utils.data_loader import load_sim_ecg

hv.opts.defaults(hv.opts.Curve(width=700, height=200))
DATASET_ROOT = "../data/synthetic/"
output_file = Path("outputs/synthetic_ecg_snippets.csv")

if not output_file.is_file():
    header = pd.DataFrame(
        columns=[
            "id",
            "l",
            "k",
            "frac",
            "ari",
            "ami",
            "homogeneity",
            "completeness",
            "v_measure",
        ]
    )
    header.to_csv(output_file.absolute(), index=False)

l = 200 * 2
k = 3
frac = 0.5

for id in tqdm(range(0, 200)):
    ecg, af_bounds, af_ind, _ = load_sim_ecg(DATASET_ROOT, id)

    g1 = (
        hv.Curve(ecg, "index", "II")
        + hv.Curve(af_ind, "index", "label").opts(height=100)
    ).cols(1)

    snippets, sn_i, sn_p, sn_fr, sn_ar, sn_reg = stumpy.snippets(
        ecg["II"].astype(float), l, k, frac
    )
    labels = np.zeros(int(sn_reg[:, 1:].max()))
    for c, s, e in sn_reg:
        labels[s:e] = c
    gt_labels = af_ind[: len(labels)]
    ari = adjusted_rand_score(gt_labels, labels)
    ami = adjusted_mutual_info_score(gt_labels, labels)
    hmg, cpl, vm = homogeneity_completeness_v_measure(gt_labels, labels)
    scores = pd.DataFrame(
        [
            pd.Series(
                {
                    "id": id,
                    "l": l,
                    "k": k,
                    "frac": frac,
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

    g2 = g1 + hv.Spikes(sn_i, "index").opts(width=700, height=100)
    g2 += reduce(
        operator.mul,
        [
            hv.Curve(p, "index", "mpdist_p").opts(width=700, height=150, alpha=0.5)
            for p in sn_p
        ],
    ) * hv.Curve(sn_p.min(0), "index", "mpdist_p").opts(
        width=700, height=150, line_dash="dashed"
    )
    g2 += reduce(
        operator.mul,
        [
            hv.VSpan(s, e).opts(
                color=["red", "yellow", "green", "blue", "purple"][c], alpha=0.3
            )
            for c, s, e in sn_reg
        ],
    ).opts(width=700, height=100, xlim=(0, len(ecg)))
    g2.cols(1)
    hv.save(g2, f"plots/synthetic_ecg/ecg_snippets_{id}.png")
