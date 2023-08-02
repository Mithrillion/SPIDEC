# %%
# loading ECG data
from topological.utils.tda_tools import *
from topological.utils.PAMAP_commons import *

# %%
dat = load_file(6)
# %%
small = dat.iloc[115000:125000].copy()
small["activityID"].loc[123100:123500] = 0  # fixing inaccurate labels
act_map = {4: "walking", 0: "non-activity", 7: "Nordic walking"}
g1 = preview_data(small)
g1
# %%
l = 100
ds_rate = 1
excl = max(1, l // 4 // ds_rate)

source = small[acc_features].values
trace_tensor = torch.tensor(source).float()
trace_subseqs = extract_td_embeddings(trace_tensor, 1, l, ds_rate, "pdt").contiguous()
# trace_subseqs = znorm(trace_subseqs, -1)
trace_subseqs.shape
# %%
k = 10
D, I = transitive_exclusion_knn_search(trace_subseqs.flatten(1, 2), k, excl)
I = I.long()
skd = find_local_density(D, k, local_connectivity=1, bandwidth=2)
G, _ = compute_membership_strengths(I, D, skd, False, True)
G = sparse_make_symmetric(G)
# G = temporal_link(G, [-1, 1], [1, 1])
emb, _ = simplicial_set_embedding(
    G.cuda(),
    2,
    batch_size=512,
    initial_alpha=1,
    random_state=7777,
    min_dist=0.1,
    n_epochs=300,
)
# %%
c_labels = small["activityID"].iloc[-len(emb) :]
hv.Overlay(
    [
        hv.Scatter(
            {
                "emb1": emb[:, 0].cpu()[c_labels.values == c],
                "emb2": emb[:, 1].cpu()[c_labels.values == c],
            },
            "emb1",
            "emb2",
            label=f"{act_map[c]}",
        ).opts(width=500, height=500, legend_position="top_left")
        for c in c_labels.unique()
    ]
)
# %%
l_s = 50
bag_size = 50
k = 10

trace_subseqs2 = extract_td_embeddings(trace_tensor, 1, l_s, 1, "pdt").contiguous()
trace_subseqs2 = znorm(trace_subseqs2, -1)
D2, I2 = mpdist_exclusion_knn_search(
    trace_subseqs2.flatten(1, 2), k, bag_size, quantile=0
)
# %%
I_ = I2.clone().long()
I_[I_ >= len(D2)] -= bag_size

skd2 = find_local_density(D2, k, local_connectivity=1, bandwidth=2)
G2, _ = compute_membership_strengths(I_, D2, skd2, False, True)
# G2 = sparse_make_symmetric(G2)
# G = temporal_link(G, [-1, 1], [1, 1])
emb2, _ = simplicial_set_embedding(
    G2.cuda(),
    2,
    batch_size=512,
    initial_alpha=1,
    random_state=7777,
    min_dist=0.1,
    n_epochs=300,
)

# %%
c_labels2 = small["activityID"].iloc[-len(emb2) :]
hv.Overlay(
    [
        hv.Scatter(
            {
                "emb1": emb2[:, 0].cpu()[c_labels2.values == c],
                "emb2": emb2[:, 1].cpu()[c_labels2.values == c],
            },
            "emb1",
            "emb2",
            label=f"{act_map[c]}",
        ).opts(width=500, height=500, legend_position="top_left")
        for c in c_labels2.unique()
    ]
)
# %%
