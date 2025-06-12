#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sqlalchemy 
import json
from tqdm import tqdm_notebook
from collections import OrderedDict
from astropy.cosmology import WMAP9 as cosmo
from rubinrate import models
from rubinrate.utils import * 
from rubinrate.sim import run_sim
from rubinrate.rates import *
import argparse
import os
import astropy.units as u
from rubinrate.metrics import *
from rubinrate.analytical_models.slsn import SLSNModel
from rubinrate.analytical_models.csm import CSMModel
from rubinrate.analytical_models.arnett import ArnettModel
from rubinrate.analytical_models.gaussian import GaussianModel
from rubinrate.dust_models import *

def load_model(model_info):
    model_type = model_info["type"]
    params = model_info["params"]
    if model_type == "GaussianModel":
        return GaussianModel(**params)
    elif model_type == "SLSNModel":
        return SLSNModel(**params)
    elif model_type == "CSMModel":
        return CSMModel(**params)
    elif model_type == "ArnettModel":
        return ArnettModel(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def load_dust_model(dust_info):
    dust_type = dust_info["type"]
    params = dust_info["params"]
    if dust_type == "LogUniformDust":
        return LogUniformDust(**params)
    else:
        raise ValueError(f"Unknown dust model type: {dust_type}")

def main():
    parser = argparse.ArgumentParser(description='rubinrate helpers')
    parser.add_argument("input_file", help="Input configuration file", type=str)
    args = parser.parse_args()

    with open(args.input_file, 'r') as f:
        config = json.load(f)

    patience = config["patience"]
    bigN = config["bigN"]
    save = config["save"]
    filename = config["filename"]
    outdir = config["outdir"]
    model_info = config["model"]
    dust_info = config["dustModel"]
    metric_list_info = config["metric_list"]
    redshifts_info = config["redshifts"]

    if outdir[-1] != '/':
        outdir += '/'

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    my_model = load_model(model_info)
    dustModel = load_dust_model(dust_info)

    # Convert metric_list_info to actual metric list
    metric_list = {}
    for key, value in metric_list_info.items():
        func_name, func_params = value
        if func_name == "snr":
            metric_list[key] = (snr, tuple(func_params))
        elif func_name == "snr_one_filt":
            metric_list[key] = (snr_one_filt, tuple(func_params))
        elif func_name == "brighter_than_mag":
            metric_list[key] = (brighter_than_mag, tuple(func_params))
        else:
            raise ValueError(f"Unknown function name: {func_name}")

    if redshifts_info["kind"] == "linear":
        redshifts = np.linspace(*redshifts_info["values"])

    metric_tracker = run_sim(metric_list, my_model, redshifts, patience, bigN, dust_Model = dustModel, keep_LCs = True)
    efficiencies = calc_efficiences(metric_tracker)
    plt.plot(redshifts, efficiencies)
    plt.show()
    integrand, total_rate = calc_rate(redshifts, efficiencies, rate_strolger, rate_z0=0.04)
    print(efficiencies)
    print(total_rate)
    if save:
        my_file_name = os.path.join(outdir, filename + '.npz')
        np.savez(my_file_name, redshifts=redshifts, metric_tracker = metric_tracker, metric_list = metric_list)

if __name__ == "__main__":
    main()
