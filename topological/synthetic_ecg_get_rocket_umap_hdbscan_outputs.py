# loading ECG data
from topological.utils.tda_tools import *
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    # silhouette_score,
    homogeneity_completeness_v_measure,
)
from pathlib import Path
from tqdm import tqdm
from topological.utils.data_loader import load_sim_ecg
from scipy.stats import mode

DATASET_ROOT = "../data/synthetic/"
output_file = Path("outputs/synthetic_ecg_rocket_umap_hdbscan.csv")

if not output_file.is_file():
    header = pd.DataFrame(
        columns=[
            "id",
            "l_s",
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

l_s = 200 * 2
k = 10
ds_rate = 8

for id in tqdm(range(0, 200)):
    ecg, af_bounds, af_ind, targets = load_sim_ecg(DATASET_ROOT, id)

    soft_interval_labels = (
        pd.Series(af_ind[::-1])
        .rolling(l_s, 1, False)
        .mean()[::-1]
        .dropna()
        .values[::ds_rate]
    )

    trace_tensor = torch.tensor(ecg["II"].astype(np.float32))[:, None]
    trace_subseqs = extract_td_embeddings(trace_tensor, 1, l_s, ds_rate, "p_td")
    trace_subseqs = znorm(trace_subseqs, -1).contiguous()

    td_labels, _ = mode(
        extract_td_embeddings(torch.tensor(af_ind)[:, None], 1, l_s, ds_rate, "p_td"),
        -1,
        keepdims=False,
    )
    td_labels = td_labels.flatten().astype(int)

    rocket = Rocket(num_kernels=200, normalise=False, n_jobs=-1, random_state=7777)
    trace_feats = rocket.fit_transform(trace_subseqs[:, None, :].numpy()).squeeze()[
        :, ::2
    ]
    trace_feats = znorm(torch.tensor(trace_feats).float(), 0).contiguous()

    g1 = (
        hv.Curve(ecg, "index", "II")
        + hv.Curve(
            (ecg.index[::ds_rate], soft_interval_labels),
            "index",
            "label",
        ).opts(height=100)
    ).cols(1)

    D, I = transitive_exclusion_knn_search(trace_feats, k, l_s // ds_rate // 4)

    skd = find_local_density(D, k, local_connectivity=1, bandwidth=1)
    G, _ = compute_membership_strengths(I.long(), D, skd, False, True)
    G = add_id_diag(G)
    G = sparse_make_symmetric(G)
    # G = purge_zeros_in_sparse_tensor(G, 0.1)
    # G = temporal_link(G, [-1, 1], [1, 1])
    emb, _ = simplicial_set_embedding(
        G.cuda(),
        2,
        batch_size=512,
        initial_alpha=1,
        random_state=7777,
        min_dist=0.1,
        n_epochs=200,
        gamma=0.1,
    )

    clusterer = hdbscan.HDBSCAN(min_cluster_size=200, min_samples=20)
    clusterer.fit(emb.cpu())

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
                    "k": k,
                    "skip_step": ds_rate,
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
                "index": ecg.index[::ds_rate][: len(clusterer.labels_)],
                "cluster": clusterer.labels_,
            },
            "index",
            "cluster",
        ).opts(
            width=700, height=100, spike_length=1, color="cluster", cmap="Category20"
        )
    ).cols(1)

    hv.save(g3, f"plots/synthetic_ecg_rocket_umap_hdbscan/{id}_segment.png")
