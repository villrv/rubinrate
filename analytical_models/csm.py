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
from scipy.integrate import solve_ivp
import scipy.integrate as integrate
from os import listdir
from os.path import isfile, join
import json


DAY_CGS = 86400.0
M_SUN_CGS = c.M_sun.cgs.value
C_CGS = c.c.cgs.value
beta = 13.7
KM_CGS = u.km.cgs.scale
RAD_CONST = KM_CGS * DAY_CGS
STEF_CONST = 4. * np.pi * c.sigma_sb.cgs.value
ANG_CGS = u.Angstrom.cgs.scale
MPC_CGS = u.Mpc.cgs.scale
pc10 = 3.086e19

DIFF_CONST = 2.0 * M_SUN_CGS / (beta * C_CGS * KM_CGS)
TRAP_CONST = 3.0 * M_SUN_CGS / (4. * np.pi * KM_CGS ** 2)
FLUX_CONST = 4.0 * np.pi * (
        2.0 * c.h * c.c ** 2 * np.pi).cgs.value * u.Angstrom.cgs.scale
X_CONST = (c.h * c.c / c.k_B).cgs.value





def ratiofinder(s: float, n: float) -> tuple:
  """
  Returns R1/Rc, R2/Rc and A values for given s and n
  Expects n in range (6,14), s in range (0,2) and returns a tuple with (R1Rc,R2Rc,A)
  """
  lmbda = (n-s)/(n-3.0)
  gamma = 5/3.0                             

  def model_chevalier(eta: float, y: np.ndarray) -> np.ndarray: 
    """
    Returns the ode value at (eta, y) for the inner shocks
    """
    U = y[0]
    P = y[1]
    C = y[2]

    repeat = (lmbda * U -1) * eta

    lhs_matrix = np.zeros((3,3)) 
    lhs_matrix[0,0] = repeat 
    lhs_matrix[0,1] = (lmbda * eta * (C) **2) / (gamma * P)
    lhs_matrix[1,0] = lmbda * eta
    lhs_matrix[1,1] = repeat / P
    lhs_matrix[1,2] = -2.0 * repeat / C
    lhs_matrix[2,1] = repeat * (1-gamma)
    lhs_matrix[2,2] = (2.0 * P *gamma * repeat) / C

    rhs_matrix = np.array([-(2-n)* (C) **2 /gamma - (U) ** 2 + U, 
     -(n-3)*(1-U), 
     -P*((n-5) -gamma*(n-3) -U*(n-2-n *gamma))])

    return np.linalg.solve(lhs_matrix, rhs_matrix)

  # initial conditions for U, P, C
  gn = 1                      
  y0 = [1/4 * (3/lmbda + 1), 3/4 * gn* (1-1/lmbda)**2, np.sqrt((gamma * 3/4 * (1-1/lmbda)**2)/4)]

  # solve ODE
  def u_boundary(eta, y): return y[0] -1/lmbda # terminal boundary at Rc
  u_boundary.terminal = True   
  sol_inner = solve_ivp(model_chevalier, (1,5), y0, events = u_boundary, atol = 1e-3)
  
  # finding R2/RC
  eta_at_boundary = sol_inner.t[-1]
  r2rc = (1/eta_at_boundary) ** (1.0/lmbda)

  # finding A  
  Avalue = sol_inner.y[1][-1]/sol_inner.y[1][0] * ((3-s)/(n-3)) **2 
  
  lhs_matrix = np.zeros((3,3)) 
  def model_parker(eta, y): 
    """
    Returns the ode value at (eta, y) for the outer shocks
    """
    U = y[0]
    P = y[1]
    C = y[2]

    repeat = ((1-lmbda * U) **2 - lmbda **2 * C**2) * eta

    lhs_matrix[0,0] = repeat 
    lhs_matrix[1,1] = repeat / P
    lhs_matrix[2,2] = (2.0 * repeat) / C 

    rhs_matrix = np.array([
      U *(1-U) * (1-lmbda * U) + C **2 * ((2 * lmbda -2 + s) - 3 * lmbda * gamma * U)/ gamma, 
      2 + U * (s-2-2 * lmbda + lmbda * gamma - 3*gamma) + lmbda * U **2 * (2-s + 2 * gamma) + lmbda * C **2 *( s-2), 
      2 + U* (1- 3 * lmbda - 3 * gamma + gamma * lmbda) + 2 * gamma * lmbda * U **2 + 
        C**2 /(1- lmbda * U) * (((-2 *lmbda **2 /gamma)- (s  * lmbda / gamma)
        -2*lmbda + s * lmbda + (2 * lmbda /gamma)) + 2 * lmbda **2 * U)])
    return np.linalg.solve(lhs_matrix, rhs_matrix) 

  # initial conditions
  pc = 1 
  y0 = [2/(lmbda * (gamma + 1)), 2 * pc /(lmbda ** 2 *(gamma + 1)), np.sqrt((2 * gamma * (gamma -1) / (lmbda **2 * (gamma + 1) **2)))]
  
  # solve ODE
  def u_boundary(eta, y): return y[0] -1/lmbda # terminal boundary at Rc
  u_boundary.terminal = True 
  sol_outer = solve_ivp(model_parker, (1,5), y0, events = u_boundary, atol = 1e-3)
  
  # finding R1/Rc 
  r1rc = (sol_outer.t[0]/sol_outer.t[-1])** (-1/lmbda)
  
  # finding A 
  Avalue *= sol_outer.y[1][0]/sol_outer.y[1][-1]
  
  return (r1rc, r2rc, Avalue)



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



class CSMModel(AnalyticalModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Additional initialization for SLSN model

        # Build CSM/IIn Model...including improved Arnett model
    def gen_csm_model(self, times, wvs, theta):

        t_exp, s,n, delta, R0, Mcsm, mej, vej, Rho, efficiency, tfloor = theta
        # Shift times 
        times = (times + t_exp) * 60 * 60 * 24  # convert days to seconds

        # Uniform grid 
        time_array = np.linspace(0, np.max(times), num = len(times) * 10, endpoint = True)

        # Times to evaluate l_in at 
        num_lin_samples_between_lobs = 250  
        time_lin = np.linspace(0, time_array[-1], num = (1 + (len(time_array) - 1) * num_lin_samples_between_lobs))


        light_speed = 2.998e10  #cm/s
        AU = 1.496e13  #1 AU to cm
        sun_mass = 1.989e33  #solar mass to g

        # Obtain ratio  
        Bfs, Brs, A = ratiofinder(s, n)

        # Parameters 
        opac = 0.34 # fixing opacity
        Rinitial = R0 * AU 
        rho = 10.**Rho
        mcsm = Mcsm * sun_mass 
        mejecta = sun_mass * mej
        vej = vej * 1e5 #convert km/s to cm/s

        # Refer to https://github.com/guillochon/MOSFiT/blob/master/mosfit/modules/engines/csm.py for variable definitions 
        q = rho * Rinitial**s 
        Esn = 3. * vej**2 * mejecta / 10.
        ti = 1.0
        Rcsm = (((3.0 - s) / (4.0 * np.pi * q) * mcsm 
             + Rinitial ** (3.0 - s)) ** (1.0 / (3.0 - s)))

        # There's a discontinuity here where when s = 1, we have a divide by zero error
        if np.isclose(s, 1):
            Rph = np.exp(-2 / (3.0 * opac * q)) * Rcsm
        else: 
            Rph = abs(
                (-2.0 * (1.0 - s) / (3.0 * opac * q) 
                + Rcsm**(1.0 - s)) ** (1.0 /(1.0 - s)))

        Mcsm_th = np.abs(4.0 * np.pi * q / (3.0 - s) * (Rph**(3.0 - s) 
                - Rinitial ** (3.0 - s)))
        g_n = (1.0 / (4.0 * np.pi * (n - delta)) * (
            2.0 * (5.0 - delta) * (n - 5.0) * Esn)**(
            (n - 3.) / 2.0) / (
            (3.0 - delta) * (n - 3.0) * mejecta)**(
            (n - 5.0) / 2.0))
        Tfs = ( 
            abs((3.0 - s) * q**((3.0 - n) / (n - s)) * (A * g_n) ** ((s - 3.0) / (n - s)) /
            (4.0 * np.pi * Bfs**(3.0 - s)))**((n - s) / ((n - 3.0) * (3.0 - s))) *
            (Mcsm_th) ** ((n - s) / ((n - 3.0) * (3.0 - s))))   
        Trs = (vej / (Brs * (A * g_n / q) ** (1.0 / (n - s))) *
            (1.0 - (3.0 - n) * mejecta / (4.0 * np.pi * vej ** (3.0 - n) * g_n))**(1.0 / (3.0 - n))) ** ((n - s) / (s - 3.0))

        L_in = efficiency *(2.0 * np.pi / (n - s)**3 *  # Finding Input luminosity
          g_n**((5.0 - s) / (n - s)) * 
          q**((n - 5.0) / (n - s)) *
          (n - 3.0)**2 *
          (n - 5.0) * Bfs**(5.0 - s) *
          A**((5.0 - s) / (n - s)) *
          (time_lin  + ti)**((2.0 * n + 6.0 * s - n * s - 15.) / (n - s)) * 
          ((Tfs - time_lin  ) > 0) + # heaviside function 
          2.0 * np.pi * (A * g_n / q)**((5.0 - n) / (n - s)) *
          Brs**(5.0 - n) * g_n *
          ((3.0 - s) / (n - s))**3 *
          (time_lin + ti)**((2.0 * n + 6.0 * s - n * s - 15.0) / (n - s)) *
            ((Trs - time_lin  ) > 0)) 
        L_in[0] = 0 
        beta =  4. * np.pi ** 3. / 9.
        t0 = opac * (Mcsm_th) / (beta * light_speed * Rph)

        L_obs = np.zeros((len(time_array), ))
        L_obs[0] = 0
        integral_funcion_evals = np.e ** (time_lin/t0) * L_in


        L_obs = (1./t0) * np.e ** (-time_lin/t0) *\
                    integrate.cumtrapz(integral_funcion_evals, time_lin, initial=0)
        
        luminosities = np.interp(times, time_lin, L_obs)
        
        #..and now I need to calc lums!
        #Do BB calculation
        radius = vej * ((times - t_exp) * ((times-t_exp)>0))
        temperature = (luminosities / (STEF_CONST * radius**2))**0.25# * (1e52)**0.25
        gind = (temperature < tfloor) | np.isnan(temperature)
        temperature = np.nan_to_num(temperature)
        notgind = np.invert(gind)
        temperature = (0. * temperature) + (temperature * notgind) + (tfloor * gind)

        radius = np.sqrt(luminosities / (STEF_CONST * temperature**4))
        fluxes = blackbody_flux(temperature, radius, wvs)
        return fluxes

    def sample(self):
      # NOTE TO ME: THIS IS SUPER INEFFICIENT BUT SHOULD WORK -- PUT INTO MODELGRID INSTEAD!!
        # THIS IS NOW DOUBLE INEFFICIENT...LUMINOSITY FUNCTION
        t_exp = 0
        self.times = np.linspace(0.1,200,1000) * 86400.0
        self.wavelengths = np.linspace(2000,10000,200)
        delta = 0
        epsilon = 0.5

        dirname = './data/iin_walkers/'
        fileList = [f for f in listdir(dirname) if isfile(join(dirname, f))]
        # Get all files...
        sn_not_selected = True

        #debugging counter
        counter = 0
        peak_mags = []
        while sn_not_selected:

            #debugging counter
            counter = counter + 1
            # Select random walker file..
            my_walker_file = np.random.choice(fileList)
            myfile = open(dirname+my_walker_file)
            all_walker_data = json.load(myfile)
            myfile.close()
            # Choose random realization...
            my_realization = np.random.choice(all_walker_data[list(all_walker_data.keys())[0]]['models'][0]['realizations'])

            # Set my parameters...
            tfloor = my_realization['parameters']['temperature']['value']
            s = my_realization['parameters']['s']['value']
            n = my_realization['parameters']['n']['value']
            R0 = my_realization['parameters']['r0']['value']
            mcsm = my_realization['parameters']['mcsm']['value']
            mej = my_realization['parameters']['mejecta']['value']
            vej = my_realization['parameters']['vejecta']['value']
            rho = np.log10(my_realization['parameters']['rho']['value'])

            theta = [t_exp, s,n, delta, \
                    R0, mcsm, mej, vej, rho, \
                    epsilon, tfloor]
            #calculate the r-band...
            rband_mag = self.gen_csm_model(self.times / 86400, np.asarray([6366]), theta)
            constant = 4. * np.pi * pc10**2# convert a luminosity to a flux
            rband_mag = rband_mag / ANG_CGS# first convert to erg/s/cm
            rband_mag = rband_mag *  (6366 * ANG_CGS)**2 / C_CGS
            rband_mag = -2.5 * np.log10(rband_mag / constant) - 48.6
            peak_mag = np.min(rband_mag)

            # THIS REPRESENTS THE OBSERVED DIST
            mu_proposal, sigma_proposal = -20.0886, 1.243
            # THIS IS THE DESIRED ONE...
            mu_target, sigma_target = -18.49, 1


            # Step 2: Calculate the PDF values for the target and proposal distributions
            target_pdf = (1 / (sigma_target * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((peak_mag - mu_target) / sigma_target)**2)
            proposal_pdf = (1 / (sigma_proposal * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((peak_mag - mu_proposal) / sigma_proposal)**2)

            # Step 3: Calculate the acceptance ratio
            normalizing_constant = 13.0 #found by hand - max ratio of one gaussian to the other
            acceptance_ratio = target_pdf / (proposal_pdf * normalizing_constant)

            # Step 4: Generate uniform random numbers for acceptance criteria
            uniform_random = np.random.uniform(0, 1)

            # Step 5: Accept or reject samples
            if uniform_random < acceptance_ratio:
                sn_not_selected = False


        self.fluxes = self.gen_csm_model(self.times / 86400, self.wavelengths, theta)
        self.theta = theta
