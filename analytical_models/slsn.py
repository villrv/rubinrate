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
import numexpr as ne
import pkg_resources
import pandas as pd
from scipy.interpolate import interp1d


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

# Needed constants from Seb
NI56_LUM = 6.45e43
CO56_LUM = 1.45e43
NI56_LIFE = 8.8
CO56_LIFE = 111.3
DAY_CGS = 86400.0
C_CGS = 29979245800.0
KM_CGS = 100000.0
M_SUN_CGS = 1.9884754153381438e+33
FOUR_PI = 12.566370614359172
N_INT_TIMES = 100
MIN_LOG_SPACING = -3
MW_RV = 3.1
LYMAN = 912.0
MPC_CGS = 3.085677581467192e+24
MAG_FAC = 2.5
AB_ZEROPOINT = 3631
JY_TO_GS2 = 1.0E-23
ANGSTROM_CGS = 1.0E-8
LIGHTSPEED = 2.9979245800E10
DIFF_CONST = 2.0 * M_SUN_CGS / (13.7 * C_CGS * KM_CGS)
TRAP_CONST = 3.0 * M_SUN_CGS / (FOUR_PI * KM_CGS ** 2)
STEF_CONST = (FOUR_PI * c.sigma_sb).cgs.value
RAD_CONST = KM_CGS * DAY_CGS
C_CONST = c.c.cgs.value
FLUX_CONST = FOUR_PI * (2.0 * c.h * c.c ** 2 * np.pi).cgs.value * u.Angstrom.cgs.scale
X_CONST = (c.h * c.c / c.k_B).cgs.value
STEF_CONST = (4.0 * np.pi * c.sigma_sb).cgs.value
N_TERMS = 1000
ANG_CGS = u.Angstrom.cgs.scale
KEV_CGS = u.keV.cgs.scale
H_C_CGS = c.h.cgs.value * c.c.cgs.value
FLUX_STD = AB_ZEROPOINT * JY_TO_GS2 / ANGSTROM_CGS * LIGHTSPEED


import numpy as np
from astropy import constants as c

import numpy as np
from astropy import constants as c


def nickelcobalt(times, fnickel, mejecta, rest_t_explosion):
    """
    This function calculates the luminosity of a nickel-cobalt powered
    supernova light curve.

    Parameters
    ----------
    times : numpy.ndarray
        The times at which to calculate the luminosity in days.
    fnickel : float
        The mass fraction of nickel in the ejecta.
    mejecta : float
        The total ejecta mass in solar masses.
    rest_t_explosion : float
        The time of explosion in rest frame days.

    Returns
    -------
    luminosities : numpy.ndarray
        The luminosity at each time in erg/s.
    """

    # Get Nickel mass
    mnickel = fnickel * mejecta

    # Calculations from 1994ApJS...92..527N
    ts = np.empty_like(times)
    t_inds = times >= rest_t_explosion
    ts[t_inds] = times[t_inds] - rest_t_explosion

    luminosities = np.zeros_like(times)
    luminosities[t_inds] = mnickel * (
        NI56_LUM * np.exp(-ts[t_inds] / NI56_LIFE) +
        CO56_LUM * np.exp(-ts[t_inds] / CO56_LIFE))

    # Make sure nan's are zero
    luminosities[np.isnan(luminosities)] = 0.0

    return luminosities


def magnetar(times, Pspin, Bfield, Mns, thetaPB, rest_t_explosion):
    """
    This function calculates the luminosity of a magnetar powered
    supernova light curve.

    Parameters
    ----------
    times : numpy.ndarray
        The times at which to calculate the luminosity in days.
    Pspin : float
        The spin period of the magnetar in milliseconds.
    Bfield : float
        The magnetic field of the magnetar in units of 10^14 Gauss.
    Mns : float
        The mass of the neutron star in solar masses.
    thetaPB : float
        The angle between the magnetic and rotation axes in radians.
    rest_t_explosion : float
        The time of explosion in rest frame days.

    Returns
    -------
    luminosities : numpy.ndarray
        The luminosity at each time in erg/s.
    """

    # Rotational Energy
    # E_rot = 1/2 I (2pi/P)^2, unit = erg
    Ep = 2.6e52 * (Mns / 1.4) ** (3. / 2.) * Pspin ** (-2)

    # tau_spindown = P/(dP/dt), unit = s
    # Magnetic dipole: power = 2/(3c^3)*(R^3 Bsin(theta))^2 * (2pi/P)^4
    # Set equal to -d/dt(E_rot) to derive tau
    tp = 1.3e5 * Bfield ** (-2) * Pspin ** 2 * (
        Mns / 1.4) ** (3. / 2.) * (np.sin(thetaPB)) ** (-2)

    ts = [
        np.inf
        if rest_t_explosion > x else (x - rest_t_explosion)
        for x in times
    ]

    # From Ostriker and Gunn 1971 eq 4
    luminosities = [2 * Ep / tp / (
        1. + 2 * t * DAY_CGS / tp) ** 2 for t in ts]
    luminosities = [0.0 if np.isnan(x) else x for x in luminosities]

    return luminosities


def total_luminosity(times, fnickel, mejecta, Pspin, Bfield, Mns, thetaPB, rest_t_explosion):
    """
    This function calculates the total luminosity of a supernova light curve by
    summing the nickel-cobalt and magnetar contributions.

    Parameters
    ----------
    times : numpy.ndarray
        The times at which to calculate the luminosity in days.
    fnickel : float
        The mass fraction of nickel in the ejecta.
    mejecta : float
        The total ejecta mass in solar masses.
    Pspin : float
        The spin period of the magnetar in milliseconds.
    Bfield : float
        The magnetic field of the magnetar in units of 10^14 Gauss.
    Mns : float
        The mass of the neutron star in solar masses.
    thetaPB : float
        The angle between the magnetic and rotation axes in radians.
    rest_t_explosion : float
        The time of explosion in rest frame days.

    Returns
    -------
    luminosities : numpy.ndarray
        The luminosity at each time in erg/s.
    """
    nickel_lum = nickelcobalt(times, fnickel, mejecta, rest_t_explosion)
    magnetar_lum = magnetar(times, Pspin, Bfield, Mns, thetaPB, rest_t_explosion)
    luminosities = nickel_lum + magnetar_lum
    return luminosities


def diffusion(times, input_luminosities, kappa, kappa_gamma, mejecta, v_ejecta, rest_t_explosion):
    """
    This function calculates the diffusion of the light from a supernova.

    Parameters
    ----------
    times : numpy.ndarray
        The times at which to calculate the luminosity in days.
    input_luminosities : numpy.ndarray
        The luminosity at each time in erg/s.
    kappa : float
        The opacity of the ejecta in cm^2/g.
    kappa_gamma : float
        The opacity of the gamma-rays in cm^2/g.
    mejecta : float
        The total ejecta mass in solar masses.
    v_ejecta : float
        The ejecta velocity in km/s.
    rest_t_explosion : float
        The time of explosion in rest frame days.

    Returns
    -------
    luminosities : numpy.ndarray
        The luminosity at each time in erg/s.
    """

    # Calculate the diffusion timescale
    tau_diff = np.sqrt(DIFF_CONST * kappa * mejecta / v_ejecta) / DAY_CGS
    trap_coeff = (TRAP_CONST * kappa_gamma * mejecta / (v_ejecta ** 2)) / DAY_CGS ** 2
    td2, A = tau_diff ** 2, trap_coeff

    # Times since explosion
    times_since_explosion = times-rest_t_explosion

    # Interpolate the input luminosities
    tau_diff = np.sqrt(DIFF_CONST * kappa * mejecta / v_ejecta) / DAY_CGS
    trap_coeff = (TRAP_CONST * kappa_gamma * mejecta / (v_ejecta ** 2)) / DAY_CGS ** 2
    td2, A = tau_diff ** 2, trap_coeff

    # Calculate the luminosities
    luminosities = np.zeros_like(times_since_explosion)
    min_te = min(times_since_explosion)
    tb = max(0.0, min_te)
    linterp = interp1d(times_since_explosion, input_luminosities, copy=False, assume_sorted=True)

    # Interpolate the input luminosities
    lu = len(times_since_explosion)
    num = int(round(N_INT_TIMES / 2.0))
    lsp = np.logspace(np.log10(tau_diff / times_since_explosion[-1]) + MIN_LOG_SPACING, 0, num)
    xm = np.unique(np.concatenate((lsp, 1 - lsp)))

    # Calculate the integral
    int_times = np.clip(tb + (times_since_explosion.reshape(lu, 1) - tb) * xm, tb, times_since_explosion[-1])
    int_te2s = int_times[:, -1] ** 2
    int_lums = linterp(int_times)  # noqa: F841
    int_args = int_lums * int_times * np.exp((int_times ** 2 - int_te2s.reshape(lu, 1)) / td2)
    int_args[np.isnan(int_args)] = 0.0

    # Return the final luminosities
    uniq_lums = np.trapz(int_args, int_times)

    # Make sure they are positive
    int_te2s[int_te2s <= 0] = np.nan
    luminosities = uniq_lums * (-2.0 * np.expm1(-A / int_te2s) / td2)
    luminosities[np.isnan(luminosities)] = 0.0

    return luminosities


def photosphere(times, luminosities, v_ejecta, temperature, rest_t_explosion):
    """
    This function calculates the photospheric radius and temperature of a
    light curve.

    Parameters
    ----------
    times : numpy.ndarray
        The times at which to calculate the luminosity in days.
    luminosities : numpy.ndarray
        The luminosity at each time in erg/s.
    v_ejecta : float
        The ejecta velocity in km/s.
    temperature : float
        The temperature floor of the photosphere in Kelvin.
    rest_t_explosion : float
        The time of explosion in rest frame days.

    Returns
    -------
    rphot : numpy.ndarray
        The photospheric radius at each time in cm.
    Tphot : numpy.ndarray
        The photospheric temperature at each time in Kelvin.
    """

    # Calculate radius squared
    radius2_in = [(RAD_CONST * v_ejecta * max(x - rest_t_explosion, 0.0)) ** 2 for x in times]
    rec_radius2_in = [
        x / (STEF_CONST * temperature ** 4)
        for x in luminosities
    ]
    rphot = []
    Tphot = []
    for li, lum in enumerate(luminosities):

        radius2 = radius2_in[li]
        rec_radius2 = rec_radius2_in[li]
        if lum == 0.0:
            temperature_out = 0.0
        elif radius2 < rec_radius2:
            temperature_out = (lum / (STEF_CONST * radius2)) ** 0.25
        else:
            radius2 = rec_radius2
            temperature_out = temperature

        rphot.append(np.sqrt(radius2))

        Tphot.append(temperature_out)

    return np.array(rphot), np.array(Tphot)


def mod_blackbody(lam, T, R2, sup_lambda, power_lambda):
    '''
    Calculate the corresponding blackbody radiance for a set
    of wavelengths given a temperature and radiance and a
    suppresion factor

    Parameters
    ---------------
    lam: Reference wavelengths in Angstroms
    T:   Temperature in Kelvin
    R2:   Radius in cm, squared

    Output
    ---------------
    Spectral radiance in units of erg/s/Angstrom

    (calculation and constants checked by Sebastian Gomez)
    '''

    # Planck Constant in cm^2 * g / s
    h = 6.62607E-27
    # Speed of light in cm/s
    c = 2.99792458E10

    # Convert wavelength to cm
    lam_cm = lam * 1E-8

    # Boltzmann Constant in cm^2 * g / s^2 / K
    k_B = 1.38064852E-16

    # Calculate Radiance B_lam, in units of (erg / s) / cm ^ 2 / cm
    if T > 0:
        exponential = (h * c) / (lam_cm * k_B * T)
        B_lam = ((2 * np.pi * h * c ** 2) / (lam_cm ** 5)) / (np.exp(exponential) - 1)
    else:
        B_lam = np.zeros_like(lam_cm) * np.nan

    # Multiply by the surface area
    A = 4*np.pi*R2

    # Output radiance in units of (erg / s) / Angstrom
    Radiance = B_lam * A / 1E8

    # Apply Supression below sup_lambda wavelength
    blue = lam < sup_lambda
    Radiance[blue] *= (lam[blue]/sup_lambda)**power_lambda

    return Radiance


def blackbody_supressed(times, luminosities, rphot, Tphot, cutoff_wavelength, alpha, sample_wavelengths):
    """
    This function calculates the blackbody spectrum using a modified blackbody fuction that is
    suppressed bluewards of a certain wavelength.

    Parameters
    ----------
    times : numpy.ndarray
        The times at which to calculate the luminosity in days.
    luminosities : numpy.ndarray
        The luminosity at each time in erg/s.
    rphot : numpy.ndarray
        The photospheric radius at each time in cm.
    Tphot : numpy.ndarray
        The photospheric temperature at each time in Kelvin.
    cutoff_wavelength : float
        The wavelength at which to start the suppression in Angstroms.
    alpha : float
        The power of the suppression.
    sample_wavelengths : numpy.ndarray
        The wavelengths at which to sample the blackbody spectrum in Angstroms.

    Returns
    -------
    seds : numpy.ndarray
        The spectral energy distribution at each time in erg/s/Angstrom.
    """

    # Constants
    xc = X_CONST  # noqa: F841
    fc = FLUX_CONST  # noqa: F841
    cc = C_CONST  # noqa: F841
    ac = ANG_CGS
    cwave_ac = cutoff_wavelength * ac
    cwave_ac2 = cwave_ac * cwave_ac
    cwave_ac3 = cwave_ac2 * cwave_ac  # noqa: F841
    zp1 = 1.0

    lt = len(times)
    seds = np.empty(lt, dtype=object)
    rp2 = np.array(rphot) ** 2
    tp = Tphot

    # Calculate the rest wavelengths
    rest_wavs = sample_wavelengths * ac / zp1

    # The power needs to add up to 5
    sup_power = alpha
    wavs_power = (5 - sup_power)  # noqa: F841

    for li, lum in enumerate(luminosities):
        # Apply absorption to SED only bluewards of cutoff wavelength
        ab = rest_wavs < cwave_ac  # noqa: F841
        tpi = tp[li]  # noqa: F841
        rp2i = rp2[li]  # noqa: F841

        sed = ne.evaluate(
            "where(ab, fc * (rp2i / cwave_ac ** sup_power/ "
            "rest_wavs ** wavs_power) / expm1(xc / rest_wavs / tpi), "
            "fc * (rp2i / rest_wavs ** 5) / "
            "expm1(xc / rest_wavs / tpi))"
            )

        sed[np.isnan(sed)] = 0.0
        seds[li] = sed

    bb_wavelengths = np.linspace(100, 100000, N_TERMS)

    norms = np.array([(R2 * STEF_CONST * T ** 4) /
                      np.trapz(mod_blackbody(bb_wavelengths, T, R2, cutoff_wavelength, alpha),
                      bb_wavelengths) for T, R2 in zip(tp, rp2)])

    # Apply renormalisation
    seds *= norms

    # Units of `seds` is ergs / s / Angstrom.
    return seds



class SLSNModel(AnalyticalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Load in datafile...
        data_path = pkg_resources.resource_filename('rubinrate',f'data/slsne_sebastian/supernovae_combined_params.txt') #only using this because it is local

        self.all_slsn_data = pd.read_csv(data_path, sep='\t')
        # Additional initialization for SLSN model

        # Build SLSN Model...including improved Arnett model
    def gen_magnetar_model(self, t, wvs, theta):
        pspin, bfield, mns, \
        thetapb, texp, kappa, \
        kappagamma, mej, vej, tfloor, fnickel, \
        alpha, cutoff_wavelength = theta

        input_luminosities = total_luminosity(t, fnickel, mej, pspin, bfield, mns, thetapb, texp)
        luminosities = diffusion(t, input_luminosities, kappa, kappagamma, mej, vej, texp)
        rphot, Tphot = photosphere(t, luminosities, vej, tfloor, texp)

        seds = blackbody_supressed(t, luminosities, rphot, Tphot, 
                            cutoff_wavelength, alpha, wvs)
        fluxes = np.vstack(seds)

        return fluxes

    def sample(self):
        sample_from_gomez = True
        if sample_from_gomez:
            random_row = self.all_slsn_data.sample(n=1).iloc[0]

            pspin = random_row['Pspin']
            bfield = 10.**(random_row['log(Bfield)'] - 14) # to units of 1e14 G
            mns = random_row['Mns']
            thetapb = random_row['thetaPB']
            texp = 0
            kappa = random_row['kappa']
            kappagamma = random_row['kappagamma']
            mej = random_row['mejecta']
            vej = random_row['vejecta']
            tfloor = random_row['temperature']
            fnickel = random_row['fnickel']

            alpha = random_row['alpha']
            cutoff_wavelength = random_row['cutoff_wavelength']

            theta = [pspin, bfield, mns, \
            thetapb, texp, kappa, \
            kappagamma, mej, vej, tfloor, fnickel, alpha, cutoff_wavelength]

        else:
            #bfield = 10.**np.random.uniform(-2,1)
            lower, upper = 0.01, 10
            mu, sigma = 0.8, 1.1
            bfield = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]

    #        pspin = np.random.uniform(0.7, 20)
            lower, upper = 0.7, 20
            mu, sigma = 2.4, 1.0
            pspin = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]

            mns = 1.4
            thetapb = np.pi/2.0
            texp = 0.0
            kappa = 0.1126
            kappagamma = 0.1
            #mej = 10.**np.random.uniform(-1,1.3)
            lower, upper = 0.1, 20
            mu, sigma = 4.8, 2.0
            mej = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]



            lower, upper = 0.1, 3
            mu, sigma = 1.47, 4.3
            vej = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0] * 1e4
            tfloor = 6000

            theta = [pspin, bfield, mns, \
            thetapb, texp, kappa, \
            kappagamma, mej, vej, tfloor]


        self.times = np.linspace(0.1,200,200) * 86400.0
        self.wavelengths = np.linspace(100,20000,500)
        self.fluxes = self.gen_magnetar_model(self.times / 86400, self.wavelengths, theta)
        '''
        plt.plot(self.fluxes[:,250])
        plt.show()
        plt.plot(self.fluxes[100,:])
        plt.show()
        sys.exit()
        '''
        self.theta = theta
