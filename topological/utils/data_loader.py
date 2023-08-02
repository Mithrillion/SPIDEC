import os
from scipy.io import loadmat
import pandas as pd
import numpy as np
import wfdb


def load_sim_ecg(dir, id):
    cls_filename_list = os.listdir(dir)

    filename = cls_filename_list[id]
    dat = loadmat(
        f"{dir}/{filename}",
        squeeze_me=True,
        chars_as_strings=True,
        struct_as_record=True,
        simplify_cells=True,
    )
    lead_names = dat["labels"]["ecgLeads"]
    ecg = pd.DataFrame(dat["signals"]["multileadECG"].T, columns=lead_names)
    ecg["index"] = ecg.index
    rr = dat["signals"]["rr"]
    rindices = dat["signals"]["Rindices"]
    targets = pd.DataFrame(
        {
            "index": rindices,
            "AF": dat["signals"]["targets_SR_AF"],
            "APB": dat["signals"]["targets_APB"],
        }
    )
    af_bounds = dat["signals"]["AFboundaries"]
    try:
        af_ind_bounds = rindices[af_bounds].reshape(-1, 2)
    except IndexError:
        af_bounds = np.row_stack(
            [
                x
                for x in af_bounds[[hasattr(x, "__len__") for x in af_bounds]]
                if len(x) > 0
            ]
        )
        af_ind_bounds = rindices[af_bounds].reshape(-1, 2)
    except ValueError:
        af_ind_bounds = rindices[[]].reshape(-1, 2)
    af_ind = np.zeros(len(ecg))
    for i in range(af_ind_bounds.shape[0]):
        af_ind[af_ind_bounds[i, 0] : af_ind_bounds[i, 1]] = 1
    return ecg, af_bounds, af_ind, targets


def load_time_series_segmentation_datasets(root, names=None):
    """
    Loads and parses the TSSB dataset as a pandas dataframe.
    Parameters
    -----------
    :param names: dataset names to load, default: all
    :return: a pandas dataframe with (TS name, window size, CPs, TS) rows
    Examples
    -----------
    >>> tssb = load_time_series_segmentation_datasets()
    """
    desc_filename = os.path.join(root, "datasets", "desc.txt")
    desc_file = []

    with open(desc_filename, "r") as file:
        for line in file.readlines():
            line = line.split(",")

            if names is None or line[0] in names:
                desc_file.append(line)

    df = []

    for row in desc_file:
        (ts_name, window_size), change_points = row[:2], row[2:]

        ts = np.loadtxt(
            fname=os.path.join(root, "datasets", ts_name + ".txt"), dtype=np.float64
        )
        df.append(
            (ts_name, int(window_size), np.array([int(_) for _ in change_points]), ts)
        )

    return pd.DataFrame.from_records(
        df, columns=["dataset", "window_size", "change_points", "time_series"]
    )


def get_closest(array, values):
    # make sure array is a numpy array
    array = np.array(array)

    # get insert positions
    idxs = np.searchsorted(array, values, side="left")

    # find indexes where previous index is closer
    prev_idx_is_less = (idxs == len(array)) | (
        np.fabs(values - array[np.maximum(idxs - 1, 0)])
        < np.fabs(values - array[np.minimum(idxs, len(array) - 1)])
    )
    idxs[prev_idx_is_less] -= 1

    return array[idxs], idxs


# load_sim_ecg("../data/synthetic/", 78)
