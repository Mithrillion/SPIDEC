# %%
from topological.utils.tda_tools import *
from functools import reduce


# %%
def get_var_name(variable):
    globals_dict = globals()

    return [var_name for var_name in globals_dict if globals_dict[var_name] is variable]


# %%
kmeans = pd.read_csv("outputs/synthetic_ecg_kmeans.csv")
mpdist = pd.read_csv("outputs/synthetic_ecg_mpdist.csv")
mpdkm = pd.read_csv("outputs/synthetic_ecg_mpdkm.csv")
rh = pd.read_csv("outputs/synthetic_ecg_rocket_hdbscan.csv")
ruh = pd.read_csv("outputs/synthetic_ecg_rocket_umap_hdbscan.csv")
# %%
score_cols = ["ami", "homogeneity", "completeness", "v_measure"]
# %%
dfs = {
    "k-Means": kmeans,
    "SPIDEC(MPdist)": mpdist,
    "MPdist k-Means": mpdkm,
    "ROCKET HDBSCAN": rh,
    "SPIDEC(ROCKET)": ruh,
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
means_stds
# %%
print(means_stds.sort_values("v_measure").to_latex())