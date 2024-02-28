import numpy as np


def snr(observed_snr, num=3, sigma=3):
	snr_count = np.sum(observed_snr>sigma)
	if snr_count < num:
		return False
	else:
		return True

def snr_one_filt(observed_snr, filts, good_filt = 'r', num=3, sigma=3):
	snr_count = np.sum(observed_snr>sigma)
	if snr_count < num:
		return False
	else:
		return True

def prerise(time,filts, mags, tmax, num=3, sigma=3):
	snr_count = np.sum(observed_snr>sigma)
	if snr_count < num:
		return False
	else:
		return True