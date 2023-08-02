from topological.utils.tda_tools import *
from topological.utils.recofit_commons import *
from pathlib import Path

from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
)

from tqdm import tqdm
import colorcet as cc
from scipy.stats import mode
from sklearn.cluster import KMeans

dat, fs, groupings_map = load_data()
mats, all_activities_df, enc = prepare_data(dat, fs, groupings_map)

output_file = Path("outputs/recofit_ps_kmeans.csv")

if not output_file.is_file():
    header = pd.DataFrame(
        columns=[
            "id",
            "l",
            "ds_rate",
            "k",
            "ari",
            "ami",
            "homogeneity",
            "completeness",
            "v_measure",
        ]
    )
    header.to_csv(output_file.absolute(), index=False)

l = 250
ds_rate = 5
excl = max(1, l // 4 // ds_rate)
k = 10
# ki = 3

for subject_id in tqdm(range(0, len(mats))):
    lab_annot = generate_annotation(mats, all_activities_df, subject_id)
    lab_mask = generate_mask(mats, all_activities_df, subject_id)
    this_mat = mats[subject_id][:, 1:]

    g1 = (
        hv.Overlay([hv.Curve(x, "t", "x") for x in this_mat.T])
        + hv.Spikes((np.arange(len(lab_annot)), lab_annot.astype(str)), "t", "c").opts(
            width=700, height=100, color="c", spike_length=1, cmap="Category20"
        )
        + hv.Spikes((np.arange(len(lab_mask)), lab_mask.astype(str)), "t", "c").opts(
            width=700, height=75, color="c", spike_length=1
        )
    )
    g1.cols(1)

    trace_tensor = torch.tensor(this_mat)
    trace_subseqs = extract_td_embeddings(trace_tensor, 1, l, ds_rate, "pdt")
    trace_subseqs = znorm(trace_subseqs, -1)
    # trace_subseqs *= torch.kaiser_window(l)[None, None, :]

    td_labels, _ = mode(
        extract_td_embeddings(torch.tensor(lab_annot)[:, None], 1, l, ds_rate, "p_td"),
        -1,
        keepdims=False,
    )
    td_labels = td_labels.flatten().astype(int)

    td_mask, _ = mode(
        extract_td_embeddings(torch.tensor(lab_mask)[:, None], 1, l, ds_rate, "p_td"),
        -1,
        keepdims=False,
    )
    td_mask = td_mask.flatten().astype(int)

    trace_feats = (
        torch.fft.rfft(trace_subseqs).abs()[..., 1 : 100 + 1].flatten(1, 2).float()
    )

    # trace_feats = znorm(trace_feats, 0)
    trace_feats = unorm(trace_feats, -1)

    static_cmap = {k: v for k, v in zip(np.arange(len(cc.glasbey)), cc.glasbey)}
    static_cmap[-1] = "grey"

    clusterer = KMeans(
        n_clusters=len(np.unique(lab_annot)), random_state=7777, n_init="auto"
    )
    clusterer.fit(trace_feats)

    g3 = (
        g1
        + hv.Spikes(
            {
                "t": np.arange(ds_rate * len(trace_feats), step=ds_rate) + l // 2,
                "c": clusterer.labels_,
            },
            kdims="t",
            vdims="c",
        ).opts(width=700, height=100, color="c", spike_length=1, cmap=static_cmap)
    ).cols(1)

    gt_ind = ~td_mask.astype(bool)
    cl = clusterer.labels_[gt_ind]
    gt_labels = td_labels[gt_ind]
    ari = adjusted_rand_score(gt_labels, cl)
    ami = adjusted_mutual_info_score(gt_labels, cl)
    hmg, cpl, vm = homogeneity_completeness_v_measure(gt_labels, cl, beta=0.5)
    scores = pd.DataFrame(
        [
            pd.Series(
                {
                    "id": subject_id,
                    "l": l,
                    "ds_rate": ds_rate,
                    "k": k,
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

    hv.save(g3, f"plots/recofit_pkm/{subject_id}_segment.png")
