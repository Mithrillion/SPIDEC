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
ns = 200
end = 1
depth = 4
channels = 3


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
            t + np.cos(2 * np.pi * t)
            + 0.7 * np.sin(4 * np.pi * t)
            + 0.2 * np.sin(2 * np.pi * t),
            + np.cos(2 * np.pi * t)
            + 0.3 * np.sin(5 * np.pi * t)
            + 0.1 * np.sin(3 * np.pi * t),
        ],
        axis=-1,
    )


def g(t):
    return np.stack(
        [
            np.sin(2 * np.pi * t)
            + 0.7 * np.cos(3 * np.pi * t)
            - 0.2 * np.cos(5 * np.pi * t),
            np.cos(2 * np.pi * t)
            - 0.7 * np.sin(3 * np.pi * t)
            + 0.2 * np.sin(5 * np.pi * t),
            + np.cos(2 * np.pi * t)
            + 0.3 * np.sin(5 * np.pi * t)
            + 0.1 * np.sin(3 * np.pi * t),
        ],
        axis=-1,
    )


f0 = f(t0)
g0 = g(t0)
# f1 = f(t1)
# f2 = f(t2)

# f0, f1, f2 = (
#     torch.tensor(f0).float(),
#     torch.tensor(f1).float(),
#     torch.tensor(f2).float(),
# )

f0 = torch.tensor(f0).float()
g0 = torch.tensor(g0).float()


f0 = rescale_path(2 * f0, depth)
g0 = rescale_path(2 * g0, depth)
# %%
sig0 = signatory.signature(f0[None, ...], depth)
sig1 = signatory.signature(g0[None, ...], depth)
# %%
logsig0 = signatory.logsignature(f0[None, ...], depth, mode="expand")
logsig1 = signatory.logsignature(g0[None, ...], depth, mode="expand")
# %%
breaks = get_term_level_breaks(channels, depth)
logsig_mul = tensor_product_no_id(logsig0, logsig1, depth, breaks)
logsig_sq = tensor_product_no_id(logsig0, logsig0, depth, breaks)
# %%
logsig_sq_s = signatory.signature_combine(logsig0, logsig0, channels, depth) - logsig0 - logsig0
# %%
logsig_sq_t = tensor_product_torch(logsig0, logsig0, channels, depth, 0)
# %%
print(logsig_sq)
print(logsig_sq_s)
print(logsig_sq_t)
# %%
print(signatory.all_words(channels, depth))
# %%
hv.Curve(logsig_sq[0, :]) * hv.Curve(logsig_sq_s[0, :]) * hv.Curve(logsig_sq_t[0, :])
# %%
signatory.signature_combine(logsig0, logsig1, channels, depth) - logsig0 - logsig1
# %%
tensor_product_torch(logsig0, logsig1, channels, depth, 0)
# %%
