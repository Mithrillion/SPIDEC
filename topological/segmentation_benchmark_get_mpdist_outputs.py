from topological.utils.tda_tools import *
import re
from topological.utils.find_exemplar_ids import find_exemplar_ids
from pathlib import Path
from scipy.stats import mode
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    homogeneity_completeness_v_measure,
)
from tqdm import tqdm

DIR = "../data/datasets_seg/"
filenames = [x for x in os.listdir(DIR) if x.endswith(".txt")]

output_file = Path("outputs/sb_mpdist.csv")

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

ds_rate = 1
k = 10

for i, selected_filename in tqdm(enumerate(filenames)):
    params = [int(x) for x in re.findall(r"(?:_)(\d+)", selected_filename)]
    opt_win_len, breaks = params[0], params[1:]

    dat = pd.read_csv(
        os.path.join(DIR, f"{selected_filename}"), header=None
    ).values.flatten()

    annotation = np.searchsorted(breaks, np.arange(len(dat)))

    g1 = hv.Curve(dat, "pos", "val", label=f"{selected_filename}").opts(
        height=150
    ) * hv.Overlay([hv.VLine(x).opts(color="red", line_dash="dashed") for x in breaks])

    skip_step = max(1, len(dat) // 4000)

    trace_tensor = torch.tensor(dat.astype(np.float32))[:, None]

    def trial(l):
        bag_size = l // ds_rate

        trace_subseqs = extract_td_embeddings(trace_tensor, 1, l, ds_rate, "p_td")
        trace_subseqs = znorm(trace_subseqs, -1).contiguous()

        td_labels, _ = mode(
            extract_td_embeddings(
                torch.tensor(annotation).float()[:, None], 1, l, ds_rate, "p_td"
            ).long(),
            -1,
            keepdims=False,
        )
        td_labels = td_labels.flatten().astype(int)

        D, I = mpdist_exclusion_knn_search(
            trace_subseqs, k, bag_size, skip_step=skip_step, quantile=0
        )

        I_ = I.clone()
        I_[I_ >= len(D) * skip_step] -= bag_size
        I_ = I_ // skip_step
        skd = find_local_density(D, k, local_connectivity=1, bandwidth=1)
        G, _ = compute_membership_strengths(I_, D, skd, False, True)
        G = add_id_diag(G)
        G = sparse_make_symmetric(G)
        G = purge_zeros_in_sparse_tensor(G, 1e-2)
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

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=100,
            min_samples=10,
            prediction_data=True,
            cluster_selection_method="eom",
        )
        clusterer.fit(emb.cpu())

        proba = hdbscan.all_points_membership_vectors(clusterer)

        cl = np.argmax(proba, -1) if len(proba.shape) == 2 else np.zeros_like(proba)
        gt_labels = td_labels[::skip_step][: len(clusterer.labels_)]
        ari = adjusted_rand_score(gt_labels, cl)
        ami = adjusted_mutual_info_score(gt_labels, cl)
        hmg, cpl, vm = homogeneity_completeness_v_measure(gt_labels, cl, beta=0.5)

        return clusterer, proba, emb, ari, ami, hmg, cpl, vm

    # there ought to be an actual validation strategy here, but since most similar studies "eyeball"
    # their optimal segment lengths, here we also use "try and choose best" for now...
    best_l, best_vm, stored_best = None, 0, None
    for l in [opt_win_len * 2, opt_win_len, opt_win_len // 2]:
        clusterer, proba, emb, ari, ami, hmg, cpl, vm = trial(l)
        if vm > best_vm:
            best_l, best_vm = l, vm
            stored_best = clusterer, proba, emb, ari, ami, hmg, cpl, vm

        l = best_l
        clusterer, proba, emb, ari, ami, hmg, cpl, vm = stored_best

    diffs = np.abs(
        pd.DataFrame(proba[:-1]).expanding(l // ds_rate // skip_step).mean().values
        - pd.DataFrame(proba[1:][::-1])
        .expanding(l // ds_rate // skip_step)
        .mean()
        .values[::-1]
    )
    xx = np.arange(1, len(proba))
    norm_factor = np.sqrt(1 / xx + 1 / (len(proba) - xx))
    arcs = diffs / norm_factor[:, None]

    timings = (
        np.arange(len(dat))[:: (ds_rate * skip_step)][: len(clusterer.labels_)] + l // 2
    )

    g2 = hv.Scatter(
        {
            "emb1": emb[:, 0].cpu(),
            "emb2": emb[:, 1].cpu(),
            "cluster": clusterer.labels_,
        },
        "emb1",
        ["emb2", "cluster"],
    ).opts(
        width=500,
        height=500,
        color="cluster",
        alpha=0.5,
        cmap="Category20",
        colorbar=True,
        xaxis="bare",
        yaxis="bare",
        title="UMAP with cluster labels",
        fontscale=2,
        color_levels=len(np.unique(clusterer.labels_)),
    )

    g3 = (
        g1
        + hv.Spikes(
            {
                "pos": timings,
                "cluster": clusterer.labels_,
            },
            "pos",
            "cluster",
            label="clusters",
        ).opts(
            width=700,
            height=80,
            xaxis="bare",
            yaxis="bare",
            spike_length=1,
            color="cluster",
            cmap="Category20",
        )
        + hv.Overlay(
            [
                hv.Curve((timings, p), "pos", "proba").opts(height=70, xaxis="bare")
                for p in proba.T
            ]
        )
        + hv.Overlay(
            [
                hv.Curve((timings, p), "pos", "diffs").opts(height=100, xaxis="bare")
                for p in arcs.T
            ]
        )
        * hv.Overlay(
            [
                hv.VLine(p * ds_rate * skip_step + l // 2)
                for p in np.argmax(np.nan_to_num(arcs, 0), 0).flatten()
            ]
        )
    ).cols(1)

    scores = pd.DataFrame(
        [
            pd.Series(
                {
                    "id": i,
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

    hv.save(g2, f"plots/sb_mpdist/{i}_scatter.png")
    hv.save(g3, f"plots/sb_mpdist/{i}_segment.png")
