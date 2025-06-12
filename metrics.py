import numpy as np
from rubinrate.lightcurves import LightCurve
import matplotlib.pyplot as plt

def snr(lc, num=3, sigma=3):
    snr_count = np.sum(lc.snrs>sigma)
    if snr_count < num:
        return 0
    else:
        return 1

def snr_one_filt(lc, good_filt = 'r', num=3, sigma=3):
    snr_count = np.sum((lc.snrs>sigma) & (lc.filters == good_filt))
    if snr_count < num:
        return 0
    else:
        return 1

# pre-rise
def during_rise(lc,  num=3, sigma=3):
    point_count = np.sum((lc.times<lc.tpeak) & (lc.snrs > sigma))
    if point_count < num:
        return 0
    else:
        return 1

# during-fall
def during_fall(lc,  num=3, sigma=3):
    point_count = np.sum((lc.times>lc.tpeak) & (lc.snrs > sigma))
    if point_count < num:
        return 0
    else:
        return 1

# peak mag
def brighter_than_mag(lc,  peakmag=22, sigma=3, num = 1):
    # figure out within duration
    gind = np.where(lc.snrs>sigma)
    if len(gind[0]) < 1:
        return 0
    point_count = np.sum((np.min(lc.mags[gind]) < peakmag) & (lc.snrs > sigma))
    if point_count < num:
        return 0
    else:
        return 1

# during peak
def during_peak(lc,  num=3, sigma=3, nmag = 1):
    # figure out within duration
    point_count = np.sum((lc.times<lc.tpeak) & (lc.snrs > sigma))
    if point_count < num:
        return 0
    else:
        return 1

# < N days after explosion
def after_exp(lc,  since_exp = 3, num = 1, sigma=3):
    # figure out within duration
    point_count = np.sum((lc.times<(lc.texp+since_exp * (1 + lc.redshift))) & (lc.times > lc.texp) & 
                        (lc.snrs > sigma))
    if point_count < num:
        return 0
    else:
        return 1
