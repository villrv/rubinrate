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
from scipy.stats import norm  # Import norm from scipy.stats



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


class GaussianModel(AnalyticalModel):
    def __init__(self, peak_luminosity, duration, temperature, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peak_luminosity = peak_luminosity
        self.duration = duration
        self.temperature = temperature
        self.times = None
        self.wavelengths = None
        self.fluxes = None

    def gen_gaussian_light_curve(self, times):
        # Gaussian function centered at the midpoint of times with FWHM = duration
        midpoint = np.mean(times)
        sigma = self.duration / (2 * np.sqrt(2 * np.log(2)))
        luminosities = self.peak_luminosity * norm.pdf(times, midpoint, sigma)
        return luminosities

    def sample(self):
        self.times = np.linspace(0, 100, 200) * 86400.0  # Example: 0 to 100 days in seconds
        self.wavelengths = np.linspace(2000, 10000, 200)  # Example: 2000 to 10000 Angstrom
        luminosities = self.gen_gaussian_light_curve(self.times / 86400)  # Generate light curve
        
        # Calculate the blackbody flux for the generated luminosities
        radius = np.sqrt(luminosities / (4 * np.pi * c.sigma_sb.cgs.value * self.temperature**4))
        self.fluxes = blackbody_flux([self.temperature] * len(self.times), radius, self.wavelengths)
