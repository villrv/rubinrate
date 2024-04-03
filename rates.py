from scipy import interpolate
from astropy.cosmology import WMAP9 as cosmo
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt


def kessler_ia(z, z0_scaling = 1.0):
	"""
	From https://iopscience.iop.org/article/10.1088/1538-3873/ab26f1/pdf
	Eqns 1/2
	Note -- there is a 3% disconinuity
	"""
	rate = (2.5e-5 * (1 + z)**1.5) * (z<1) + \
			(9.7e-5 * (1 + z)**-0.5) * (z>=1)
	rate = rate * 1e9
	return rate

def rate_madau(z):
	"""
	From https://www.annualreviews.org/doi/full/10.1146/annurev-astro-081811-125615
	Eqn 15
	"""
	A = 2.7
	B = 2.9
	C = 5.6
	# An addition 1+z due to time dilation
	rate = (1 + z)**A / (1 + ((1 + z)/B)**C) / (1 + z)
	return rate



def rate_strolger(z):
	"""
	From https://iopscience.iop.org/article/10.1088/0004-637X/813/2/93/pdf
	Eqn 9
	"""
	A = 0.015
	B = 1.5
	C = 5.0 
	D = 6.1
	k = 0.006
	rate = 1e9 * k * A * (1 + z)**C / (((1 + z)/B)**D + 1)
	return rate


def calc_rate(redshifts, efficiencies, rate_func, rate_z0 = 1):
	eff_func = interpolate.interp1d(redshifts, efficiencies, axis=0, bounds_error=False, fill_value=0.0)

	#Do volumetric calculation
	if (type(rate_func) == float) | (type(rate_func) == int):
		rate = rate * u.Gpc**-3 * u.year**-1 / (1 + redshifts)
	else:
		rate = rate_z0 * rate_func(redshifts) *  u.Gpc**-3 * u.year**-1
	print(rate)
	rate = np.repeat(rate[None,:], np.shape(efficiencies[-1]), axis=0).T
	dVs = cosmo.differential_comoving_volume(redshifts)
	dVs = np.repeat(dVs[None, :], np.shape(efficiencies[-1]), axis=0).T
	integrand =  4. * np.pi * rate * dVs * eff_func(redshifts)
	integral = np.trapz(integrand, redshifts, axis=0)
	total_rate = integral * u.year.decompose()

	return(integrand, total_rate.decompose(bases=u.cgs.bases))
