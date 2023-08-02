# loading ECG data
from topological.utils.tda_tools import *
from pathlib import Path
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score,
    homogeneity_completeness_v_measure,
)
from topological.utils.data_loader import load_sim_ecg
from scipy.stats import mode

DATASET_ROOT = "../data/synthetic/"

l = 200
ds_rate = 1
k = 10
skip_step = 8
bag_size = l

norm_method = "znorm"
output_file = Path(f"outputs/synthetic_ecg_mpdist.csv")

if not output_file.is_file():
    header = pd.DataFrame(
        columns=[
            "id",
            "l_s",
            "k",
            "skip_step",
            "bag_size",
            "ari",
            "ami",
            "homogeneity",
            "completeness",
            "v_measure",
        ]
    )
    header.to_csv(output_file.absolute(), index=False)

for id in range(200):
    ecg, af_bounds, af_ind, targets = load_sim_ecg(DATASET_ROOT, id)

    soft_interval_labels = (
        pd.Series(af_ind[::-1])
        .rolling(l, 1, False)
        .mean()[::-1]
        .dropna()
        .values[:: (ds_rate * skip_step)]
    )

    if norm_method == "znorm":
        trace_tensor = torch.tensor(ecg["II"].astype(np.float32))[:, None]
        trace_subseqs = extract_td_embeddings(trace_tensor, 1, l, ds_rate, "p_td")
        trace_subseqs = znorm(trace_subseqs, -1).contiguous()
    elif norm_method == "rolling":
        source = ecg["II"] - ecg["II"].rolling(l, 1, True).mean()
        trace_tensor = torch.tensor(source.astype(np.float32))[:, None]
        trace_subseqs = extract_td_embeddings(
            trace_tensor, 1, l, ds_rate, "p_td"
        ).contiguous()
    else:
        raise ValueError()

    td_labels, _ = mode(
        extract_td_embeddings(torch.tensor(af_ind)[:, None], 1, l, ds_rate, "p_td"),
        -1,
        keepdims=False,
    )
    td_labels = td_labels.flatten().astype(int)

    g1 = (
        hv.Curve(ecg, "index", "II")
        + hv.Curve(
            (ecg.index[:: (ds_rate * skip_step)], soft_interval_labels),
            "index",
            "label",
        ).opts(height=100)
    ).cols(1)

    D, I = mpdist_exclusion_knn_search(
        trace_subseqs, k, bag_size // ds_rate, skip_step=skip_step
    )
    I_ = I.clone().long()
    I_[I_ >= len(D) * skip_step] -= bag_size
    I_ = I_ // skip_step
    skd = find_local_density(D, k, local_connectivity=1, bandwidth=1)
    G, _ = compute_membership_strengths(I_, D, skd, False, True)
    G = add_id_diag(G)
    G = sparse_make_symmetric(G)
    # G = temporal_link(G, [-1, 1], [1 / k, 1 / k])
    emb, _ = simplicial_set_embedding(
        G.cuda(),
        2,
        batch_size=512,
        initial_alpha=1,
        random_state=7777,
        min_dist=0.1,
        n_epochs=200,
        gamma=0.1,
        prune_graph=True,
    )

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=200, min_samples=20, prediction_data=True
    )
    clusterer.fit(emb.cpu())

    proba = hdbscan.all_points_membership_vectors(clusterer)
    cl = np.argmax(proba, -1) if len(proba.shape) == 2 else np.zeros_like(proba)
    gt_labels = td_labels[::skip_step][: len(clusterer.labels_)]
    ari = adjusted_rand_score(gt_labels, cl)
    ami = adjusted_mutual_info_score(gt_labels, cl)
    hmg, cpl, vm = homogeneity_completeness_v_measure(gt_labels, cl, beta=0.5)
    sil = silhouette_score(emb.cpu(), cl)
    scores = pd.DataFrame(
        [
            pd.Series(
                {
                    "id": id,
                    "l_s": l,
                    "k": k,
                    "skip_step": skip_step,
                    "bag_size": bag_size,
                    "silhouette": sil,
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

    g2 = (
        hv.Scatter(
            {
                "emb1": emb[:, 0].cpu(),
                "emb2": emb[:, 1].cpu(),
                "label": soft_interval_labels[: len(emb)],
            },
            "emb1",
            ["emb2", "label"],
        ).opts(
            width=350, height=350, color="label", alpha=0.5, cmap="bkr", colorbar=True
        )
        + hv.Scatter(
            {
                "emb1": emb[:, 0].cpu(),
                "emb2": emb[:, 1].cpu(),
                "cluster": clusterer.labels_,
            },
            "emb1",
            ["emb2", "cluster"],
        ).opts(
            width=350,
            height=350,
            color="cluster",
            alpha=0.5,
            cmap="Set2",
            colorbar=True,
        )
    ).cols(2)

    g3 = (
        g1
        + hv.Spikes(
            {
                "index": ecg.index[:: (ds_rate * skip_step)][: len(clusterer.labels_)],
                "cluster": clusterer.labels_,
            },
            "index",
            "cluster",
        ).opts(
            width=700, height=100, spike_length=1, color="cluster", cmap="Category20"
        )
    ).cols(1)

    hv.save(g2, f"plots/synthetic_ecg_mpdist/{id}_scatter.png")
    hv.save(g3, f"plots/synthetic_ecg_mpdist/{id}_segment.png")
