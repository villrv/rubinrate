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


def rate_strolger(z, z0_scaling = 1.0):
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


def calc_rate(redshifts, efficiencies, rate_func, rate_z0 = 1, save=True, filename='rates'):
	eff_func = interpolate.interp1d(redshifts, efficiencies, bounds_error=False, fill_value=0.0)

	#Do volumetric calculation
	high_densityredshifts = np.linspace(np.min(redshifts),np.max(redshifts),100)
	if (type(rate_func) == float) | (type(rate_func) == int):
		rate = rate * u.Gpc**-3 * u.year**-1 / (1 + redshifts)
	else:
		rate = rate_z0 * rate_func(redshifts) * u.Gpc**-3 * u.year**-1
	integrand =  4. * np.pi * rate * cosmo.differential_comoving_volume(redshifts) * eff_func(redshifts)
	integral = np.trapz(integrand, redshifts)
	total_rate = integral * u.year.decompose()
	if save:
		my_file_name = './products/'+filename + '.npz'
		np.savez(my_file_name, redshifts=redshifts, rates=integrand)

	return(integrand, total_rate.decompose(bases=u.cgs.bases))
