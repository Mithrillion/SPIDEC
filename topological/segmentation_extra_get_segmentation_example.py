# %%
from topological.utils.tda_tools import *
import re
from topological.utils.find_exemplar_ids import find_exemplar_ids

DIR = "../data/datasets_seg/"
# %%
filenames = [x for x in os.listdir(DIR) if x.endswith(".txt")]
# %%
selected_filename = filenames[25]
print(selected_filename)
# %%
params = [int(x) for x in re.findall(r"(?:_)(\d+)", selected_filename)]
# %%
opt_win_len, breaks = params[0], params[1:]
# opt_win_len *= 2
# %%
dat = pd.read_csv(
    os.path.join(DIR, f"{selected_filename}"), header=None
).values.flatten()
# %%
g1 = hv.Curve(dat, "pos", "val", label=f"{selected_filename}").opts(
    height=150
) * hv.Overlay([hv.VLine(x).opts(color="red", line_dash="dashed") for x in breaks])
g1
# %%
l_s = (
    opt_win_len // 2
)  # a lot of the cases win_len * 2 works well, trial order: x2 -> x1 -> //2
ds_rate = 1
k = 10
skip_step = max(1, len(dat) // 4000)
bag_size = l_s // ds_rate

trace_tensor = torch.tensor(dat.astype(np.float32))[:, None]

trace_subseqs = extract_td_embeddings(trace_tensor, 1, l_s, ds_rate, "p_td")
trace_subseqs = znorm(trace_subseqs, -1).contiguous()
# %%
D, I = mpdist_exclusion_knn_search(
    trace_subseqs, k, bag_size, skip_step=skip_step, quantile=0
)

# %%
I_ = I.clone()
I_[I_ >= len(D) * skip_step] -= bag_size
I_ = I_ // skip_step
skd = find_local_density(D, k, local_connectivity=1, bandwidth=1)
G, SD = compute_membership_strengths(I_, D, skd, True, True)
G = add_id_diag(G)
G = sparse_make_symmetric(G)
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
# %%
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=100,
    min_samples=10,
    prediction_data=True,
    cluster_selection_method="eom",
)
clusterer.fit(emb.cpu())
pd.Series(clusterer.labels_).value_counts()
# %%
hv.Scatter(
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
)
# %%
proba = hdbscan.all_points_membership_vectors(clusterer)
diffs = np.abs(
    pd.DataFrame(proba[:-1]).expanding(l_s // ds_rate // skip_step).mean().values
    - pd.DataFrame(proba[1:][::-1])
    .expanding(l_s // ds_rate // skip_step)
    .mean()
    .values[::-1]
)
xx = np.arange(1, len(proba))
norm_factor = np.sqrt(1 / xx + 1 / (len(proba) - xx))
arcs = diffs / norm_factor[:, None]
# %%
timings = (
    np.arange(len(dat))[:: (ds_rate * skip_step)][: len(clusterer.labels_)] + l_s // 2
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
            hv.VLine(p * ds_rate * skip_step + l_s // 2)
            for p in np.argmax(np.nan_to_num(arcs, 0), 0).flatten()
        ]
    )
).cols(1)
g3
# %%
exemplar_ids = find_exemplar_ids(clusterer)

for i, ex in enumerate(exemplar_ids):
    hv.output(
        hv.Layout(
            [
                hv.Curve(
                    dat[c : c + 2 * l_s],
                    label=f"cluster = {i}",
                ).opts(width=200, height=200, alpha=0.8)
                * hv.VSpan(0, l_s).opts(color="green", alpha=0.3)
                for c in np.random.choice(ex, 10, replace=False)
            ]
        ).cols(5)
    )

# %%
clusterer.condensed_tree_.plot(select_clusters=True)
# %%
hv.save(g3, "plots/segment_cut.png")
# %%
