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

dat, fs, groupings_map = load_data()
mats, all_activities_df, enc = prepare_data(dat, fs, groupings_map)

output_file = Path("outputs/recofit_graph_mpdist.csv")

if not output_file.is_file():
    header = pd.DataFrame(
        columns=[
            "id",
            "l",
            "ds_rate",
            "k",
            "skip_step",
            "ari",
            "ami",
            "homogeneity",
            "completeness",
            "v_measure",
        ]
    )
    header.to_csv(output_file.absolute(), index=False)

l = 50 * 5
ds_rate = 2
excl = max(1, l // 4 // ds_rate)
k = 10
skip_step = 5
bag_size = l

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

    trace_tensor = torch.tensor(this_mat).float()
    trace_subseqs = extract_td_embeddings(trace_tensor, 1, l, ds_rate, "pdt")
    trace_subseqs = znorm(trace_subseqs, -1)

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

    D, I = mpdist_exclusion_knn_search(
        trace_subseqs.flatten(1, 2).contiguous(),
        k,
        bag_size // ds_rate,
        skip_step=skip_step,
        quantile=0,
    )

    I_ = I.clone()
    I_[I_ >= len(D) * skip_step] = len(D) * skip_step - 1
    I_ = I_ // skip_step

    SD = knn_entries_to_scipy_sparse(D, I_)
    SD = SD.maximum(SD.T)
    min_nns = pd.Series(SD.tocoo().row).value_counts().min()

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=100,
        min_samples=min_nns,
        cluster_selection_method="eom",
        metric="precomputed",
    )
    clusterer.fit(SD)

    static_cmap = {k: v for k, v in zip(np.arange(len(cc.glasbey)), cc.glasbey)}
    static_cmap[-1] = "grey"

    selection = clusterer.labels_ != -1
    timing = np.arange(ds_rate * len(trace_subseqs), step=ds_rate * skip_step)[
        : len(clusterer.labels_)
    ]
    g3 = (
        g1
        + hv.Spikes(
            {
                "t": timing[selection] + l // 2,
                "c": clusterer.labels_[selection],
            },
            kdims="t",
            vdims="c",
        ).opts(width=700, height=100, color="c", spike_length=1, cmap=static_cmap)
    ).cols(1)

    gt_ind = ~td_mask[::skip_step][: len(clusterer.labels_)].astype(bool)
    # cl = np.argmax(proba, -1)[gt_ind]
    cl = clusterer.labels_[gt_ind]
    gt_labels = td_labels[::skip_step][: len(clusterer.labels_)][gt_ind]
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
                    "skip_step": skip_step,
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

    hv.save(g3, f"plots/recofit_mpdist_graph/{subject_id}_segment.png")
