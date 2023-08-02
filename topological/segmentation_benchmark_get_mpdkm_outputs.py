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
from sklearn.cluster import KMeans

DIR = "../data/datasets_seg/"
filenames = [x for x in os.listdir(DIR) if x.endswith(".txt")]

output_file = Path("outputs/sb_mpdkm.csv")

if not output_file.is_file():
    header = pd.DataFrame(
        columns=[
            "id",
            "l",
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


def get_mpm(trace_subseqs, ks):
    with torch.no_grad():
        DM = torch.norm(trace_subseqs[:, None, :] - trace_subseqs[None, :, :], 2, -1)
        MPM = -F.max_pool2d(
            -DM[None, None, ...], kernel_size=(ks, ks), stride=1
        ).squeeze()
    return MPM


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

    skip_step = max(1, len(dat) // 2000)

    trace_tensor = torch.tensor(dat.astype(np.float32))[:, None]

    def trial(l, k):
        torch.cuda.empty_cache()
        trace_subseqs = extract_td_embeddings(trace_tensor, 1, l, skip_step, "p_td")
        trace_subseqs = znorm(trace_subseqs, -1).contiguous()

        td_labels, _ = mode(
            extract_td_embeddings(
                torch.tensor(annotation).float()[:, None], 1, l, skip_step, "p_td"
            ).long(),
            -1,
            keepdims=False,
        )
        td_labels = td_labels.flatten().astype(int)

        ks = max(1, l // skip_step) * 3
        MPM = get_mpm(trace_subseqs.contiguous().cuda(), ks)

        clusterer = KMeans(n_clusters=k, random_state=7777, n_init="auto")
        clusterer.fit(MPM.cpu())

        cl = clusterer.labels_
        gt_labels = td_labels[: len(clusterer.labels_)]
        ari = adjusted_rand_score(gt_labels, cl)
        ami = adjusted_mutual_info_score(gt_labels, cl)
        hmg, cpl, vm = homogeneity_completeness_v_measure(gt_labels, cl, beta=0.5)

        return clusterer, ari, ami, hmg, cpl, vm

    # there ought to be an actual validation strategy here, but since most similar studies "eyeball"
    # their optimal segment lengths, here we also use "try and choose best" for now...
    best_l, best_vm, stored_best = None, 0, None
    for l in [opt_win_len * 2, opt_win_len, opt_win_len // 2]:
        for k in [len(breaks) + 1, len(breaks) + 2]:
            clusterer, ari, ami, hmg, cpl, vm = trial(l, k)
            if vm > best_vm:
                best_l, best_vm = l, vm
                stored_best = clusterer, ari, ami, hmg, cpl, vm

            l = best_l
            clusterer, ari, ami, hmg, cpl, vm = stored_best

    timings = np.linspace(0, len(dat) - l * 4, len(clusterer.labels_))

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
    ).cols(1)

    scores = pd.DataFrame(
        [
            pd.Series(
                {
                    "id": i,
                    "l": l,
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

    hv.save(g3, f"plots/sb_mpdkm/{i}_segment.png")
