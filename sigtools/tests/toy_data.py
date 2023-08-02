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
depth = 5


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

hv.Curve(f0)
# %%
sig0 = signatory.signature(f0[None, ...], depth)
# %%
sig1 = signatory.signature_combine(sig0, sig0, 2, depth)
# %%
hv.Curve(sig0[0]) * hv.Curve(sig1[0])
# %%
sig_prod = tensor_product_torch(sig0, sig0, 2, depth)
# %%
hv.Curve(sig0[0]) * hv.Curve(sig1[0]) * hv.Curve(sig_prod[0])
# %%
logsig0 = signatory.signature_to_logsignature(sig0, 2, depth, mode="expand")
#%%
# log_res = tensor_log1p(sig0, 2, depth)
#%%
# print(logsig0)
# print(log_res)
# print(logsig0 / log_res)
# %%
exp_res = tensor_exp_impl(logsig0, 2, depth)
# %%
print(exp_res)
print(sig0)
print(sig0 / exp_res)