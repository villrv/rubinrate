from astropy.cosmology import WMAP9 as cosmo
from scipy.integrate import simps
from scipy import interpolate
import sqlite3
import pandas as pd
import random
from astropy.coordinates import SkyCoord
from dust_extinction.averages import GCC09_MWAvg
from dustmaps.sfd import SFDQuery
import extinction
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import models
import time
from lightcurves import LightCurve
import analytical_models


# This defines our dust model
sfd = SFDQuery()



c_AAs     = 2.99792458e18                       # Speed of light in Angstrom/s

band_list = ['u','g','r','i','z','y']
band_wvs = 1./ (0.0001 * np.asarray([3751.36, 4741.64, 6173.23, 7501.62, 8679.19, 9711.53]))
#band_wvs = band_wvs * (1./u.micron)

band_wvs = np.asarray([3751.36, 4741.64, 6173.23, 7501.62, 8679.19, 9711.53]) # in angstrom
band_wvs = np.asarray([3751.36, 4741.64, 6173.23, 7501.62, 8679.19, 9711.53]) # in angstrom

def run_sim(metrics, my_model, redshifts, patience, bigN, dust_Model = None, keep_LCs = False):
	# This reads in the OpSim File
	# This file is a database containing many,many pointings of LSST
	conn = sqlite3.connect("./data/baseline_v3.3_10yrs.db")  
	#conn = sqlite3.connect("./data/baseline_nexp1_v1.7_10yrs.db")  

	#Now in order to read in pandas dataframe we need to know table name
	cursor = conn.cursor()
	cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
	df = pd.read_sql_query('SELECT fieldRA, fieldDec, seeingFwhmEff, observationStartMJD, filter, fiveSigmaDepth, skyBrightness  FROM observations', conn)
	conn.close()

	patience_counter = 0

	# This is all to just set up the OpSim Equations
	function_list = np.asarray([])
	filter_list = np.asarray([])
	for band in band_list:
		blah = np.loadtxt('./filters/LSST_LSST.'+band+'.dat')
		function_list = np.append(function_list,np.trapz(blah[:,1]/blah[:,0],blah[:,0]))
		filter_list = np.append(filter_list,interpolate.interp1d(blah[:,0],blah[:,1],bounds_error=False,fill_value=0.0))
	func_dict = {}
	bands_and_func = zip(band_list, function_list)
	for band, func in bands_and_func:
		func_dict[band] = func
	filt_dict = {}
	bands_and_filts = zip(band_list, filter_list)
	for band, func in bands_and_filts:
		filt_dict[band] = func

	# Inject as a function of redshift
	tmin = np.min(df['observationStartMJD'])
	tmax = np.max(df['observationStartMJD'])
	metric_tracker = np.zeros((len(redshifts), len(metrics), bigN))
	all_lc = []
	for kk, redshift in enumerate(redshifts):
		print('Redshift:',redshift)
		if patience_counter > patience:
			break

		#First, move the LC to be at the time of injection
		counter = 0 
		# I'm going to inject this LC bigN times
		d = cosmo.luminosity_distance(redshift).value # in Mpc
		d = d * 3.086e+24 #in cm

		for j in np.arange(bigN):
			model_theta = []
			if type(my_model) == models.ModelGrid:
				my_specific_model = my_model.sample()
				t = my_specific_model.times
				lamS_full = my_specific_model.wavelengths
				spec_full = my_specific_model.fluxes
			elif type(my_model) == models.AnalyticalModel:
				my_specific_model = my_model.sample()
				t = my_specific_model.times
				lamS_full = my_specific_model.wavelengths
				spec_full = my_specific_model.fluxes
				model_theta = my_specific_model.theta
			elif type(my_model) == analytical_models.gaussian.GaussianModel:
				my_specific_model = my_model
				my_specific_model.sample()
				t = my_specific_model.times
				lamS_full = my_specific_model.wavelengths
				spec_full = my_specific_model.fluxes
			else:
				t = my_model.times
				lamS_full = my_model.wavelengths
				spec_full = my_model.fluxes

			t = t * (1. + redshift)
			max_phase = np.max(t) * 1.15741e-5
			mags = np.zeros((len(t),6))
			for jj, my_filt in enumerate(band_list):
				lamF,filt = np.loadtxt('./filters/LSST_LSST.'+my_filt+'.dat'  ,unpack=True) #Two columns with wavelength and response in the range [0,1]
				lamS = lamS_full * (1.+redshift)
				spec = spec_full / (1. + redshift)
				spec_int  = interpolate.interp1d(lamS,spec, axis=1, bounds_error=False,fill_value = 0)(lamF)
				I1        = simps(spec_int*filt*lamF,lamF, axis=1) 
				I2        = simps(filt/lamF,lamF) 
				fnu       = I1/I2 / c_AAs  / (4. * np.pi * d**2)   
				with np.errstate(divide='ignore'):
					mAB       = -2.5*np.log10(fnu) - 48.6              #AB magnitude
				mags[:,jj] = mAB

			times = t * 1.15741e-5
			#Pick a random injection time
			start_mjd = random.uniform(tmin,tmax)
			shifted_times = times + start_mjd
			model_theta.append(start_mjd)

			#Pick a random location on the sky
			ra = np.random.uniform(0,360)
			dec = np.arcsin(np.random.uniform(-1,1)) * 180 / np.pi

			#See if LSST is pointing at this location:
			new_db = df.where((np.abs(df['fieldRA'] - ra)<1.75) & \
				(np.abs(df['fieldDec'] - dec)<1.75) & \
				(df['observationStartMJD']>start_mjd) & \
				(df['observationStartMJD']<(start_mjd+max_phase))).dropna()
			if len(new_db) == 0:
				continue


			# Now let's add reddening from MW dust

			#First look up the amount of mw dust at this location
			coords = SkyCoord(ra,dec, frame='icrs', unit="deg")
			ebv = sfd(coords)

			# Now figure out how much the magnitude is affected by this dust...this is only for the MW
			ext_list = extinction.fitzpatrick99(band_wvs, ebv * 3.1, 3.1)  # I know this says use Av, but it clearly uses ebv...
			# now do the host...remember that the wavelengths are different...
			if dust_Model is not None:
				host_ebv = dust_Model.sample()
				host_ext_list = extinction.fitzpatrick99(band_wvs / (1. + redshift), host_ebv * 3.1)
			else:
				host_ext_list = ext_list * 0.0
			#For each filter, dim the model light curve by this dust
			#blue filters will get more affected than red
			reddened_mags = np.zeros(np.shape(mags))
			lsst_mags = np.zeros(len(new_db))

			for bandcounter, myband in enumerate(band_list):
				gind2 = np.where(new_db['filter']== myband)
				reddened_mags[:,bandcounter] = mags[:,bandcounter] + ext_list[bandcounter] + host_ext_list[bandcounter]

				# Calculate the peak time and brightness
				if myband == 'r':
					tpeak = shifted_times[np.argmin(reddened_mags[:,bandcounter])]
					peakmag = np.argmin(reddened_mags[:,bandcounter])

				#Resample to match LSST cadence
				#Youre going to want to replace this with something that repeats periodically
				my_model_function = interpolate.interp1d(shifted_times, reddened_mags[:,bandcounter],
										bounds_error=False, fill_value=30.0)

				new_model_mags = my_model_function(new_db['observationStartMJD'].where(new_db['filter']==myband).dropna().values)
				lsst_mags[gind2] = new_model_mags



			#now lets add noise to the LC...this involves eqns..
			g = 2.2
			h = 6.626e-27
			expTime = 30.0
			my_integrals = 10.**(-0.4*(lsst_mags+48.6)) * [func_dict.get(key) for key in new_db['filter'].values]

			C= expTime * np.pi * 321.15**2 / g / h * my_integrals
			fwhmeff = new_db['seeingFwhmEff'].values
			pixscale = 0.2#''/pixel
			neff = 2.266*(fwhmeff/pixscale)**2
			sig_in = 12.7
			neff = 2.266*(new_db['seeingFwhmEff'].values/pixscale)**2
			my_integrals = 10.**(-0.4*(new_db['skyBrightness'].values+48.6)) * [func_dict.get(key) for key in new_db['filter'].values]
			B= expTime * np.pi * 321.15**2 / g / h * my_integrals * (pixscale)**2
			snr = C/np.sqrt(C/g+(B/g+sig_in**2)*neff)
			gind_snr = np.where(snr > 20)
			snr[gind_snr] = 20
			gind_snr = np.where(snr < 0.0001)
			snr[gind_snr] = 0.0001
			with np.errstate(divide='ignore'):
				err = 1.09/snr

			lsst_mags = lsst_mags + np.random.normal(0, err)
			mylc = LightCurve(new_db['observationStartMJD'].values, lsst_mags, 
								new_db['filter'].values, snr, start_mjd, 
								tpeak, peakmag, redshift, model_theta, mwebv = ebv)



			# Save LCs if the user requests them...
			if keep_LCs:
				all_lc.append(mylc)
			# Calculate FOMs as function of redshift
			current_metric_true = False
			for i, val in enumerate(metrics.items()):
				func_name, (func, args) = val
				my_metric = func(mylc, *args)
				metric_tracker[kk,i,j] = my_metric
				if my_metric:
					current_metric_true = True
			# Save a few LCs as a function of redshift
			if not current_metric_true:
				continue
			counter+=1

		if counter == 0:
			patience_counter+=1

	if keep_LCs:
		np.savez('./products/lcs.npz', lcs = all_lc)
	return metric_tracker