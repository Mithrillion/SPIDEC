# %%
from topological.utils.tda_tools import *

# %%
kmeans = pd.read_csv("outputs/recofit_kmeans.csv")
mpdist = pd.read_csv("outputs/recofit_mpdist.csv")
ph = pd.read_csv("outputs/recofit_ps_hdbscan.csv")
pkm = pd.read_csv("outputs/recofit_ps_kmeans.csv")
puh = pd.read_csv("outputs/recofit_ps_umap_hdbscan.csv")
ticc = pd.read_csv("outputs/recofit_ticc.csv")
graph = pd.read_csv("outputs/recofit_graph_mpdist.csv")
# %%
score_cols = ["ami", "homogeneity", "completeness", "v_measure"]
# %%
dfs = {
    "k-Means": kmeans,
    "SPIDEC(MPdist)": mpdist,
    "SPIDEC(No UMAP)": graph,
    "FFT k-Means": pkm,
    "FFT HDBSCAN": ph,
    "SPIDEC(FFT)": puh,
    "TICC": ticc,
}
for n, df in dfs.items():
    df["method"] = n

# %%
dfa = pd.concat(dfs, axis=0)
# %%
means = dfa.groupby("method")[score_cols].mean()
stds = dfa.groupby("method")[score_cols].std()
# %%
print(means.to_latex(float_format="%.3f"))
# %%
means_stds = means.copy()
for col in score_cols:
    means_stds[col] = (
        means[col].map(lambda x: f"{x:.3f}")
        + "\u00B1"
        + stds[col].map(lambda x: f"{x:.3f}")
    )
# %%
means_stds.sort_values("v_measure")
# %%
print(means_stds.sort_values("v_measure").to_latex())
# %%
(
    hv.Distribution(
        dfa[dfa["method"] == "FFT HDBSCAN"]["v_measure"], label="FFT HDBSCAN"
    )
    * hv.Distribution(
        dfa[dfa["method"] == "FFT k-Means"]["v_measure"], label="FFT k-Means"
    )
    * hv.Distribution(
        dfa[dfa["method"] == "SPIDEC(MPdist)"]["v_measure"], label="SPIDEC(MPdist)"
    )
    * hv.Distribution(
        dfa[dfa["method"] == "TICC"]["v_measure"], label="TICC"
    )
).opts(width=600, height=600, fontscale=2, legend_position="top_left")
# %%
pd.Series(
    np.sign(
        dfa[dfa["method"] == "SPIDEC(FFT)"]["v_measure"].values
        - dfa[dfa["method"] == "FFT k-Means"]["v_measure"].values
    )
).value_counts()
# %%
pd.Series(
    np.sign(
        dfa[dfa["method"] == "FFT HDBSCAN"]["v_measure"].values
        - dfa[dfa["method"] == "FFT k-Means"]["v_measure"].values
    )
).value_counts()
# %%
g = hv.Scatter(
    (
        dfa[dfa["method"] == "FFT k-Means"]["v_measure"].values,
        dfa[dfa["method"] == "FFT HDBSCAN"]["v_measure"].values,
    ),
    "FFT KMeans v_measure",
    "FFT HDBSCAN v_measure",
) * hv.Segments([(0, 0, 1, 1)]).opts(color="red", line_dash="dashed")
g.opts(width=600, height=600, fontscale=2)
# %%
