import os
import sys
import numpy as np
import pandas as pd
import torch

import holoviews as hv
import hdbscan

hv.extension("bokeh")

sys.path.append("..")
sys.path.append("../..")

from tsprofiles.functions import *
from sigtools.transforms import *
from rocket.rocket import Rocket

from umap_torch.nonparametric_umap import (
    simplicial_set_embedding,
    compute_membership_strengths,
    add_id_diag,
    prune_small_nns,
)

from ripser import Rips
import persim

import gtda.homology as ghm
import gtda.diagrams as gpd
import gtda.time_series as gts
import gtda.plotting as gplot

import matplotlib.pyplot as plt

hv.opts.defaults(hv.opts.Curve(width=700, height=200))


def plot_pd(pers_diag):
    lim = (
        np.ceil(
            max(
                pers_diag[0][pers_diag[0] < np.inf].max(),
                *(x.max() for x in pers_diag[1:])
            )
        )
        + 0.5
    )
    res = (
        hv.Scatter(pers_diag[0])
        * hv.Overlay([hv.Scatter(p) for p in pers_diag[1:]])
        * hv.Slope(slope=1, y_intercept=0).opts(
            color="black",
            line_dash="dashed",
            alpha=0.3,
            xlim=(-0.5, lim),
            ylim=(-0.5, lim),
        )
        * hv.HLine(lim - 0.5).opts(color="black", line_dash="dashed", alpha=0.3)
    )
    return res


def align_phase(X):
    w = X.shape[-1]
    X_rfft = torch.fft.rfft(X)
    sf_argmax_freq = torch.argmax(X_rfft.abs(), dim=-1, keepdim=True)
    max_freqs = torch.gather(X_rfft, -1, sf_argmax_freq)
    max_freq_angles = torch.angle(max_freqs).squeeze()
    X_rfft_pa = (
        X_rfft
        * torch.exp(-torch.complex(torch.tensor(0.0), max_freq_angles))[None, :, None]
    )
    return torch.fft.irfft(X_rfft_pa, n=w, dim=-1)