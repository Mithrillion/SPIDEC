# %%
from topological.utils.tda_tools import *
import re
from topological.utils.find_exemplar_ids import find_exemplar_ids
import stumpy
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import SpectralEmbedding
from umap.umap_ import simplicial_set_embedding as sse
from umap.umap_ import find_ab_params
import hdbscan
from scipy.sparse.csgraph import shortest_path, csgraph_from_masked
from numpy.ma import masked_array

DIR = "../data/datasets_seg/"
# %%
filenames = [x for x in os.listdir(DIR) if x.endswith(".txt")]
# %%  all time series
g = hv.Layout()
for fid in [6, 9, 13]:
    selected_filename = filenames[fid]  # 6, 9, 13
    params = [int(x) for x in re.findall(r"(?:_)(\d+)", selected_filename)]
    opt_win_len, breaks = params[0], params[1:]
    n_clusters = len(breaks) + 1
    dat = pd.read_csv(
        os.path.join(DIR, f"{selected_filename}"), header=None
    ).values.flatten()
    g += hv.Curve(dat, "pos", "val", label=f"{selected_filename}").redim(
        pos=f"x_{fid}", val=f"val_{fid}"
    ).opts(width=1400, height=150, yaxis="bare", xlabel="") * hv.Overlay(
        [
            hv.VLine(x).opts(color="red", line_dash="dashed", fontscale=1.5)
            for x in breaks
        ]
    )
# %%
g.cols(1)
# %%
hv.save(g, "plots/PD_series.png")
# %%  load data
selected_filename = filenames[13]  # 6, 9, 13
print(selected_filename)
params = [int(x) for x in re.findall(r"(?:_)(\d+)", selected_filename)]
opt_win_len, breaks = params[0], params[1:]
n_clusters = len(breaks) + 1
dat = pd.read_csv(
    os.path.join(DIR, f"{selected_filename}"), header=None
).values.flatten()

g1 = hv.Curve(dat, "pos", "val", label=f"{selected_filename}").opts(
    height=150
) * hv.Overlay([hv.VLine(x).opts(color="red", line_dash="dashed") for x in breaks])
# g1
# %%  extract subsequences
wlen = opt_win_len
ds_rate = len(dat) // 1000
ks = ds_rate = len(dat) // 1000 * 3

full_subseqs = extract_td_embeddings(torch.tensor(dat)[:, None], 1, wlen, 1, "p_td")
full_subseqs = znorm(full_subseqs, -1)

long_subseqs = extract_td_embeddings(
    torch.tensor(dat)[:, None], 1, wlen * 3, ds_rate, "p_td"
)
long_subseqs = znorm(long_subseqs, -1)

subseqs = extract_td_embeddings(torch.tensor(dat)[:, None], 1, wlen, ds_rate, "p_td")
subseqs = znorm(subseqs, -1)
subseqs.shape
# %%  kmeans
cls = KMeans(n_clusters=n_clusters)
y_km = cls.fit_predict(subseqs)
# %%
g2 = hv.Curve(y_km).opts(
    height=120,
    title=f"{selected_filename} - Subseq KMeans",
    xaxis="bare",
    yaxis="bare",
    fontscale=1.5,
)
# g2
# %%
# hv.save(g2, f"plots/PDs/{selected_filename}_kmeans.png")
# %%  subsequences PD
hom = ghm.WeightedRipsPersistence(
    reduced_homology=False, n_jobs=-1, weight_params={"n_neighbors": 5}
)
Xt = hom.fit_transform([subseqs])
gplot.plot_diagram(Xt[0])
# %%  phase-aligned PD
subseqs_pa = align_phase(subseqs.float())
# # %%  kmeans
# cls_pa = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
# y_km_pa = cls_pa.fit_predict(subseqs_pa[0, ...])
# # %%
# (g1 + hv.Curve(y_km_pa)).cols(1)
# # %%
# hom_pa = ghm.WeightedRipsPersistence(
#     reduced_homology=False, n_jobs=-1, weight_params={"n_neighbors": 5}
# )
# Xt_pa = hom_pa.fit_transform(subseqs_pa)
# gplot.plot_diagram(Xt_pa[0])
# %%  MPdist mat
DM = torch.norm(subseqs[:, None, :] - subseqs[None, :, :], 2, -1)
MPM = -F.max_pool2d(
    -DM[None, None, :, :], kernel_size=(ks, ks), stride=1
).squeeze()
# %%  kmeans
kmeans_mpd = KMeans(n_clusters=2, random_state=7777)
y_km_mpd = kmeans_mpd.fit_predict(MPM)

# cls_mpd = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
# y_km_mpd = cls_mpd.fit_predict(MPM)
# %%
g3 = hv.Curve(y_km_mpd).opts(
    height=120,
    title=f"{selected_filename} - Pairwise MPdist",
    xaxis="bare",
    yaxis="bare",
    fontscale=1.5,
)
g3
# %%
hv.save(g3, f"plots/PDs/{selected_filename}_MPdist.png")
# %%  MPdist PD
hom_mpd = ghm.WeightedRipsPersistence(
    reduced_homology=False, n_jobs=-1, weight_params={"n_neighbors": 10}
)
Xt_mpd = hom_mpd.fit_transform([MPM])
gplot.plot_diagram(Xt_mpd[0])
# %% sparse graph distance
D, I = exclusion_knn_search(subseqs.float().contiguous(), 20, wlen // ds_rate // 4)
sDM = to_scipy_sparse(knn_entries_to_sparse_dists(D, I))
shortest_DM = shortest_path(sDM, directed=False)
# %%
spec = SpectralClustering(
    n_clusters=n_clusters, random_state=7777, affinity="precomputed_nearest_neighbors"
)
y_km_graph = spec.fit_predict(shortest_DM)
# %%
g4 = hv.Curve(y_km_graph).opts(
    height=120,
    title=f"{selected_filename} - Graph Dist.",
    xaxis="bare",
    yaxis="bare",
    fontscale=1.5,
)
# g4
# %%
# hv.save(g4, f"plots/PDs/{selected_filename}_graph.png")
# %%
mapper = SpectralEmbedding(
    n_components=n_clusters, random_state=7777, affinity="precomputed"
)
emb_spec = mapper.fit_transform(np.exp(-(shortest_DM**2) / (0.1 * D.max() ** 2)))
# %%
hv.Scatter(emb_spec)
# %%
hom_graph = ghm.WeightedRipsPersistence(
    reduced_homology=False, n_jobs=-1, weight_params={"n_neighbors": 5}
)
Xt_graph = hom_graph.fit_transform([emb_spec])
gplot.plot_diagram(Xt_graph[0])
# # %%  Rocket
# rocket = Rocket(num_kernels=200, n_jobs=-1, random_state=7777)
# subseqs_roc = rocket.fit_transform(long_subseqs[:, None, :].numpy())[:, 1::2]
# # subseqs_roc = znorm(subseqs_roc, 0)
# # %%  kmeans
# cls_roc = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
# y_km_roc = cls_roc.fit_predict(subseqs_roc)
# # %%
# (g1 + hv.Curve(y_km_roc)).cols(1)
# # %%
# hom_roc = ghm.WeightedRipsPersistence(
#     reduced_homology=False, n_jobs=-1, weight_params={"n_neighbors": 5}
# )
# # hom_roc = ghm.VietorisRipsPersistence(reduced_homology=False, n_jobs=-1)
# Xt_roc = hom_roc.fit_transform([subseqs_roc])
# gplot.plot_diagram(Xt_roc[0])
# %%  UMAP
k = 5
skip_step = ds_rate
D, I = mpdist_exclusion_knn_search(
    full_subseqs.float(), k, wlen, skip_step=skip_step, quantile=0
)
# %%
I_ = I.clone()
I_[I_ >= len(D) * skip_step] -= wlen
I_ = I_ // skip_step
skd = find_local_density(D, k, local_connectivity=1, bandwidth=1)
G, SD = compute_membership_strengths(I_, D, skd, True, True)
G = add_id_diag(G)
G = sparse_make_symmetric(G)

a, b = find_ab_params(1, 0.1)
emb, _ = sse(
    None,
    to_scipy_sparse(G.cpu()),
    2,
    1.0,
    a,
    b,
    1.0,
    5,
    500,
    "spectral",
    np.random.RandomState(7777),
    "euclidean",
    {},
    False,
    {},
    False,
)
# %%
hv.Scatter(emb)
# %%  kmeans
cls = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
y_km_um = cls.fit_predict(emb)
# %%
g5 = hv.Curve(y_km_um).opts(
    height=120,
    title=f"{selected_filename} - SPIDEC",
    xaxis="bare",
    yaxis="bare",
    fontscale=1.5,
)
g5
# %%
hv.save(g5, f"plots/PDs/{selected_filename}_spidec.png")
# %%
hom_um = ghm.WeightedRipsPersistence(
    reduced_homology=False, n_jobs=-1, weight_params={"n_neighbors": 5}
)
Xt_um = hom_um.fit_transform([emb])
gplot.plot_diagram(Xt_um[0])
# %%
