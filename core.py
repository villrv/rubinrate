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
import utils
from metrics import snr
from sim import run_sim
from rates import calc_rate, rate_strolger,kessler_ia
import argparse
import os
import astropy.units as u



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

#Central filter wavelength in 1/micron
band_wvs = 1./ (0.0001 * np.asarray([3751.36, 4741.64, 6173.23, 7501.62, 8679.19, 9711.53]))
band_wvs = band_wvs * (1./u.micron)

# Define your model generator or model. This can be a mosfit model file with priors.

#my_model = models.SedonaModel()
#my_model.load('sed_m60.0_mrp5.60_xlan1.67e-04_m560.000.dat')


#my_model = models.PlasticcModel()
#my_model_grid = models.ModelGrid(my_model, './data/ibc_models/')

my_model = models.SNCosmoModel()
my_model.load()


# https://www.aanda.org/articles/aa/pdf/2016/03/aa26760-15.pdf


# Define metrics we want to track
metric_list = [snr]

# Run the simulation to calculate efficiencies for each metric
redshifts = np.linspace(0.0001,1.0,10)
efficiencies = run_sim(metric_list, my_model, redshifts, patience)

plt.plot(redshifts, efficiencies)
plt.show()

# Define your rate as a function of redshift or a constant.
# Ibc
# integrand, total_rate = calc_rate(redshifts, efficiencies, rate_strolger, rate_z0=0.3)

# Ia
integrand, total_rate = calc_rate(redshifts, efficiencies, kessler_ia)


print(integrand, total_rate)

if not os.path.exists(args.outdir):
    os.makedirs(args.outdir)

