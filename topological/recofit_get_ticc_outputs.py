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
from external.TICC.TICC_solver import TICC

dat, fs, groupings_map = load_data()
mats, all_activities_df, enc = prepare_data(dat, fs, groupings_map)

output_file = Path("outputs/recofit_ticc.csv")

if not output_file.is_file():
    header = pd.DataFrame(
        columns=[
            "id",
            "k",
            "ari",
            "ami",
            "homogeneity",
            "completeness",
            "v_measure",
        ]
    )
    header.to_csv(output_file.absolute(), index=False)

for subject_id in tqdm(range(121, len(mats))):
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

    static_cmap = {k: v for k, v in zip(np.arange(len(cc.glasbey)), cc.glasbey)}
    static_cmap[-1] = "grey"

    k = len(np.unique(lab_annot))

    fname = f"ticc_temp/{subject_id}.csv"
    pd.DataFrame(this_mat).to_csv(fname, header=False, index=False)

    ticc = TICC(
        window_size=5,
        number_of_clusters=k,
        lambda_parameter=11e-2,
        beta=600,
        maxIters=100,
        threshold=2e-5,
        write_out_file=False,
        prefix_string="ticc_temp/output_folder/",
        num_proc=8,
    )

    (cluster_assignment, cluster_MRFs) = ticc.fit(input_file=fname)

    g3 = (
        g1
        + hv.Spikes(
            (np.arange(len(cluster_assignment)), cluster_assignment), "t", "cls"
        ).opts(
            width=700,
            height=60,
            spike_length=1,
            cmap=static_cmap,
            color_levels=k,
            color="cls",
            title="cluster",
            yaxis="bare",
            xaxis="bare",
        )
    ).cols(1)

    gt_ind = ~lab_mask.astype(bool)[:len(cluster_assignment)]
    cl = cluster_assignment[gt_ind]
    gt_labels = lab_annot[:len(cluster_assignment)][gt_ind]
    ari = adjusted_rand_score(gt_labels, cl)
    ami = adjusted_mutual_info_score(gt_labels, cl)
    hmg, cpl, vm = homogeneity_completeness_v_measure(gt_labels, cl, beta=0.5)
    scores = pd.DataFrame(
        [
            pd.Series(
                {
                    "id": subject_id,
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

    hv.save(g3, f"plots/recofit_ticc/{subject_id}_segment.png")
