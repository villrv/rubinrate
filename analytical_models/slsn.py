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



#.blackbody import blackbody_lambda
def blackbody_flux(temperature, radius, wavelength):
    # Convert wavelength from Angstrom to cm
    wavelength_cm = wavelength * 1e-8

    all_fluxes = np.zeros((len(temperature), len(wavelength)))
    for i in range(len(temperature)):
        temp = temperature[i]
        rad = radius[i]

        # Calculate the black body flux density using Planck's law
        numerator = 2 * c.h * C_CGS**2 / wavelength_cm**5
        exponent = c.h * C_CGS / (wavelength_cm * c.k_B * temp)
        denominator = np.exp(exponent.value) - 1
        flux_density_erg_per_s_per_cm2_per_angstrom = numerator / denominator
        flux_density_erg_per_s_per_per_angstrom = flux_density_erg_per_s_per_cm2_per_angstrom * 4. * np.pi * rad**2
        all_fluxes[i,:] = flux_density_erg_per_s_per_per_angstrom

    return all_fluxes 



class SLSNModel(AnalyticalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization for SLSN model

        # Build SLSN Model...including improved Arnett model
    def gen_magnetar_model(self, t, wvs, theta):
        pspin, bfield, mns, \
        thetapb, texp, kappa, \
        kappagamma, mej, vej, tfloor = theta

        Ep = 2.6 * (mns / 1.4) ** (3. / 2.) * pspin ** (-2)
        # ^ E_rot = 1/2 I (2pi/P)^2, unit = erg
        tp = 1.3e5 * bfield ** (-2) * pspin ** 2 * (
            mns / 1.4) ** (3. / 2.) * (np.sin(thetapb)) ** (-2)
        tau_diff = np.sqrt(DIFF_CONST * kappa *
                                    mej / vej) / DAY_CGS

        A = (TRAP_CONST * kappagamma * mej / (vej ** 2)) / DAY_CGS ** 2
        td2 =  tau_diff ** 2

        lum_inp = 2.0 * Ep / tp / (1. + 2.0 * t * DAY_CGS / tp) ** 2

        integrand = 2* lum_inp * t/tau_diff * np.exp((t/tau_diff)**2)  * 1e52

        multiplier =  (1.0 - np.exp(-A*t**-2)) * np.exp(-(t/tau_diff)**2) 
        luminosities = multiplier * cumtrapz(integrand, t, initial = 0)

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
        bfield = 10.**np.random.uniform(-2,1)
        pspin = np.random.uniform(0.7, 20)
        mns = 1.4
        thetapb = np.pi/2.0
        texp = 0.0
        kappa = 0.1126
        kappagamma = 0.1
        mej = 10.**np.random.uniform(-1,1.3)
        lower, upper = 0.1, 3
        mu, sigma = 1.47, 4.3
        vej = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0] * 1e4
        tfloor = 6000

        theta = [pspin, bfield, mns, \
        thetapb, texp, kappa, \
        kappagamma, mej, vej, tfloor]
        self.times = np.linspace(0.1,200,200) * 86400.0
        self.wavelengths = np.linspace(2000,10000,200)
        self.fluxes = self.gen_magnetar_model(self.times / 86400, self.wavelengths, theta)
        self.theta = theta
