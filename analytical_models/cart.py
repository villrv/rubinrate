import numpy as np
import matplotlib
#matplotlib.use('AGG') 
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz
import astropy.constants as c
import astropy.units as u
from scipy import interpolate
from astropy.cosmology import WMAP9 as cosmo
import extinction
from .analytical_model import AnalyticalModel
import math
import scipy.stats as stats
import random



DAY_CGS = 86400.0
M_SUN_CGS = c.M_sun.cgs.value
C_CGS = c.c.cgs.value
beta = 13.7
KM_CGS = u.km.cgs.scale
RAD_CONST = KM_CGS * DAY_CGS
STEF_CONST = 4. * np.pi * c.sigma_sb.cgs.value
ANG_CGS = u.Angstrom.cgs.scale
MPC_CGS = u.Mpc.cgs.scale

DIFF_CONST = 2.0 * M_SUN_CGS / (beta * C_CGS * KM_CGS)
TRAP_CONST = 3.0 * M_SUN_CGS / (4. * np.pi * KM_CGS ** 2)
FLUX_CONST = 4.0 * np.pi * (
        2.0 * c.h * c.c ** 2 * np.pi).cgs.value * u.Angstrom.cgs.scale
X_CONST = (c.h * c.c / c.k_B).cgs.value


# Some helper functions for our rejection sampling...
def calc_mag7(times, mags):
    peak_mag_ind = np.argmin(mags)
    peak_mag_time = times[peak_mag_ind]
    closest_to_7_days_ind = np.argmin(np.abs(times - peak_mag_time -7))
    peak_mag = mags[peak_mag_ind]
    mag_7_days = mags[closest_to_7_days_ind]
    return(np.log10(mag_7_days - peak_mag)[0])


def rejection_sample_light_curves(peak_mag, delta_mag, target_peak_mag = (-16.19, 0.76), target_delta_m7 = (-0.382, 0.218)):
    """
    Rejection sampling for light curves based on target Gaussian distributions.

    Args:
        target_peak_mag (tuple): (mean, std) for peak magnitude Gaussian distribution.
        target_delta_m7 (tuple): (mean, std) for delta_m_7 Gaussian distribution.
        num_samples (int): Number of accepted samples to generate.

    Returns:
        list: List of accepted theta values.
        list: List of accepted light curves (times, magnitudes).
    """
    mean_peak_mag, std_peak_mag = target_peak_mag
    mean_delta_m7, std_delta_m7 = target_delta_m7


    # Evaluate acceptance probabilities
    prob_peak_mag = stats.norm.pdf(peak_mag, loc=mean_peak_mag, scale=std_peak_mag)
    prob_delta_m7 = stats.norm.pdf(delta_mag, loc=mean_delta_m7, scale=std_delta_m7)

    # Acceptance probability (product of probabilities)
    acceptance_prob = prob_peak_mag * prob_delta_m7

    # Perform rejection sampling
    if np.random.rand() < acceptance_prob:
        return(1)
    else:
        return(0)


def blackbody_flux(temperature, radius, wavelength):
    # Convert wavelength from Angstrom to cm
    wavelength_cm = wavelength * 1e-8

    all_fluxes = np.zeros((len(temperature), len(wavelength)))
    for i in range(len(temperature)):
        temp = temperature[i]
        rad = radius[i]

        # Calculate the black body flux density using Planck's law
        numerator = (4 * np.pi) * 2 * c.h * C_CGS**2 / wavelength_cm**5
        exponent = c.h * C_CGS / (wavelength_cm * c.k_B * temp)
        denominator = np.exp(exponent.value) - 1
        flux_density_erg_per_s_per_cm2_per_angstrom = numerator / denominator
        flux_density_erg_per_s_per_per_angstrom = flux_density_erg_per_s_per_cm2_per_angstrom * 4. * np.pi * rad**2
        all_fluxes[i,:] = flux_density_erg_per_s_per_per_angstrom

    return all_fluxes 



class CaRTModel(AnalyticalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization for SLSN model

        # Build SLSN Model...including improved Arnett model
    def gen_cart_model(self, t, wvs, theta):
        mej, fni, vej, tfloor = theta
        # Convert to grams
        mej = mej * M_SUN_CGS
        mni = mej * fni
        # Convert velocity to cm/s
        vej = vej * 1e8
        tni = 8.8  # days
        tco = 111.3  # days
        epco = 6.8e9  # erg/g/s
        epni = 3.9e10  # erg/g/s
        opac = 0.2
        texp = 0
        # Diffusion timescale in days
        td = np.sqrt(2 * opac * mej / (13.7 * C_CGS * vej)) / 86400  # convert seconds to days

        integrand1 = (t / td) * np.exp(t**2 / td**2 - t / tni)
        integrand2 = (t / td) * np.exp(t**2 / td**2 - t / tco)

        # Luminosity calculation
        luminosities = 2 * mni / (td) * np.exp(-t**2 / td**2) * \
              ((epni - epco) * cumtrapz(integrand1, t, initial=0) + 
               epco * cumtrapz(integrand2, t, initial=0))


        #Do BB calculation
        radius = RAD_CONST * vej * ((t - texp) * ((t-texp)>0))
        temperature = (luminosities / (STEF_CONST * radius**2))**0.25# * (1e52)**0.25
        gind = (temperature < tfloor) | np.isnan(temperature)
        temperature = np.nan_to_num(temperature)
        notgind = np.invert(gind)
        temperature = (0. * temperature) + (temperature * notgind) + (tfloor * gind)

        radius = np.sqrt(luminosities / (STEF_CONST * temperature**4))
        fluxes = blackbody_flux(temperature, radius, wvs)
        return fluxes

    def sample(self):

        # From Yadavalli+24, we want
        # mean mag (r-band) -16.19 0.76
        # delta_m_7 -0.382 0.218

        accepted = False
        while not accepted:

            lower, upper = -2,0.5
            mu, sigma = 3.0, 0.5
            #mej = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]
            mej = 10.**random.uniform(lower,upper)

            lower, upper = 0,1
            mu, sigma = 10.0, 1
            #vej = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]
            vej = 10.**random.uniform(lower,upper)

            lower, upper = -1,0
            mu, sigma = 0.2, 0.02
            #fni = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]
            fni = 10.**random.uniform(lower,upper)

            tfloor = 3000

            theta = [mej, fni, vej, tfloor]
            self.times = np.linspace(0.1,100,1000) * 86400.0
            self.wavelengths = np.array([6415])
            self.fluxes = self.gen_cart_model(self.times / 86400, self.wavelengths, theta)
            flux_new = self.fluxes * 6415**2/ (4. * np.pi * 3.086e19**2 * 3e18)
            mag = -2.5*np.log10(flux_new) - 48.6
            times_days = self.times / 86400

            delta_mag = calc_mag7(times_days, mag)
            peak_mag = np.min(mag)
            accept = rejection_sample_light_curves(peak_mag, delta_mag)

            if accept:
                self.theta = theta
                accepted = True
                self.wavelengths = np.linspace(2000,10000,200)
                self.fluxes = self.gen_cart_model(self.times / 86400, self.wavelengths, theta)
