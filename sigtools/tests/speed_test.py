#%%
import os
import sys
import numpy as np

import torch
import signatory

import holoviews as hv
from holoviews import opts

hv.extension("bokeh")

sys.path.append("..")
from tensor_product import *
from transforms import *

# %%
ns = 400
end = 1
depth = 4


def phi(t, pow=2):
    T = np.max(t) - np.min(t)
    return ((t - np.min(t)) / T) ** pow * T + np.min(t)


t0 = np.linspace(0, end, ns)
# t1 = np.linspace(0, end * 3, ns * 3)
# t2 = np.concatenate([t0, end + phi(t0, 2), 2 * end + phi(t0, 3)])


def f(t):
    return np.stack(
        [
            np.sin(2 * np.pi * t)
            + 0.7 * np.cos(4 * np.pi * t)
            + 0.2 * np.cos(2 * np.pi * t),
            np.cos(2 * np.pi * t)
            + 0.7 * np.sin(4 * np.pi * t)
            + 0.2 * np.sin(2 * np.pi * t),
            + np.cos(2 * np.pi * t)
            + 0.3 * np.sin(5 * np.pi * t)
            + 0.1 * np.sin(3 * np.pi * t),
        ],
        axis=-1,
    )


f0 = f(t0)
# f1 = f(t1)
# f2 = f(t2)

# f0, f1, f2 = (
#     torch.tensor(f0).float(),
#     torch.tensor(f1).float(),
#     torch.tensor(f2).float(),
# )

f0 = torch.tensor(f0).float()
f0 = rescale_path(f0, depth)
# %%
sig0 = signatory.signature(f0[None, ...], depth)
# %%
# %timeit signatory.signature_combine(sig0, sig0, 3, depth)
# %%
breaks, slots = prepare_tensor_product(3, depth)
# %%
sig0_ = append_scalar_term(sig0)
# %%
# %timeit tensor_product_prepared(sig0_, sig0_, depth, breaks, slots)
# %%
tensor_product_alt(sig0, sig0, depth, breaks)
# %%
# %timeit tensor_product_alt(sig0, sig0, depth, breaks)
# %%
sig0n = sig0.numpy()
sig0n_ = sig0_.numpy()
brn = breaks.numpy()
sln = slots.numpy()
# %%
tensor_product_numba(sig0n, sig0n, depth, brn)
# %%
# %timeit tensor_product_numba(sig0n, sig0n, depth, brn)
# %%
