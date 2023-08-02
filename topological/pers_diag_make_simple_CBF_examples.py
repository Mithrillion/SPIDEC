# %%
import sys
sys.path.append("..")
from topological.utils.tda_tools import *
import gtda.homology as ghm
import sklearn.datasets as skds
from pyts.datasets import make_cylinder_bell_funnel

hv.extension("bokeh")

# %%
X, y = skds.make_blobs(
    n_samples=300, n_features=2, random_state=2222, center_box=(-5.0, 5.0)
)
# %%
hv.Scatter((X[:, 0], X[:, 1], y), "x", ["y", "c"]).opts(color="c", cmap="bkr").opts(
    width=500, height=500
)
# %%
vr = ghm.VietorisRipsPersistence(reduced_homology=False)
pd = vr.fit_transform_plot(X[None, ...])
# %%
vr = ghm.WeightedRipsPersistence(reduced_homology=False)
pd = vr.fit_transform_plot(X[None, ...])
# %%
Z, u = make_cylinder_bell_funnel(n_samples=15, random_state=2222)
Z = Z.flatten()
# %%
hv.Curve(Z).opts(width=500, height=150)
# %%
pcs, _ = to_td_pca(torch.tensor(Z[None, :, None]), 2, 1, 100, random_state=2222)
pcs = pcs.squeeze()
# %%
hv.Scatter(
    (pcs[:, 0], pcs[:, 1], np.repeat(u, 128)[-len(pcs) :]),
    ("x", "PC1"),
    [("y", "PC2"), "c"],
).opts(color="c", width=500, height=500) * hv.Curve(pcs).opts(color="grey", alpha=0.3)
# %%
vr = ghm.WeightedRipsPersistence(reduced_homology=False, n_jobs=-1)
pd = vr.fit_transform_plot(pcs[None, ...])
# %%
