#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sqlalchemy 
import json
from tqdm import tqdm_notebook
from collections import OrderedDict
from astropy.cosmology import WMAP9 as cosmo
import models
from utils import * 
from sim import run_sim
from rates import *
import argparse
import os
import astropy.units as u
from metrics import *
from analytical_models.slsn import SLSNModel



parser = argparse.ArgumentParser(description='rubinrate helpers')

parser.add_argument("--outdir", help="Output directory", dest='outdir',
                        type=str, default='./products/')

args = parser.parse_args()

if args.outdir[-1] != '/':
    args.outdir += '/'

if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)


band_list = ['u','g','r','i','z','y']
patience = 5
bigN = 100
save = True
filename = 'run0'

#Central filter wavelength in 1/micron
band_wvs = 1./ (0.0001 * np.asarray([3751.36, 4741.64, 6173.23, 7501.62, 8679.19, 9711.53]))
band_wvs = band_wvs * (1./u.micron)

# Define your model generator or model. This can be a mosfit model file with priors.

#my_model = models.SedonaModel()
#my_model.load('sed_m60.0_mrp5.60_xlan1.67e-04_m560.000.dat')

'''
my_model = models.PlasticcModel()
#my_model.load('ibc_391.dat.gz.npz', './data/ibc_models_new/')
my_model_grid = models.ModelGrid(my_model, './data/slsne_new/')
'''
#my_model = models.SNCosmoModel()
#my_model.load()

# Making SLSNe for Adam:
my_model = models.AnalyticalModel(SLSNModel)



# https://www.aanda.org/articles/aa/pdf/2016/03/aa26760-15.pdf


# Define metrics we want to track

metric_list = {
    "snr3": (snr, (10,3,)),  # Example cutoff value of 3
    "snr5": (snr, (5,5,)),
    "snr-r-3": (snr_one_filt, ('g', 3, 5,)),
    "rise3": (during_rise, (3,)),
    "fall3": (during_fall, (3,)),
    "afterexp": (after_exp, (5,1,5)),
    "brighterthan22": (brighter_than_mag, (22, 3, 3)),
    "brighterthan21": (brighter_than_mag, (22.5, 3, 3))
}

# Run the simulation to calculate efficiencies for each metric
redshifts = np.linspace(0.0001,2,100)
metric_tracker = run_sim(metric_list, my_model, redshifts, patience, bigN, keep_LCs = True)
efficiencies = calc_efficiences(metric_tracker)

plt.plot(redshifts, efficiencies)
plt.show()

# Define your rate as a function of redshift or a constant.
# Ibc
#integrand, total_rate = calc_rate(redshifts, efficiencies, rate_strolger, rate_z0=0.3)

# SLSNe
integrand, total_rate = calc_rate(redshifts, efficiencies, rate_madau, rate_z0=20)

# Ia
#integrand, total_rate = calc_rate(redshifts, efficiencies, kessler_ia)


print(efficiencies)
print(total_rate)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

if save:
    my_file_name = './products/'+filename + '.npz'
    np.savez(my_file_name, redshifts=redshifts, metric_tracker = metric_tracker, 
                metric_list = metric_list)


