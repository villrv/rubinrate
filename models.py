import numpy as np
import sncosmo
import matplotlib.pyplot as plt
import gzip
from os import listdir
from os.path import isfile, join

class TransientModel():
    """
    A basic transient model
    """

    def __init__(self):
        """
        Parameters:
        ----------
        ...

        """

    def load(file_names):
        return 0

    def generate(time, param):
        return 0

    def sample():
        return 0

class SNCosmoModel(TransientModel):
    def __init__(self):
        """
        """
    
    def load(self, source='hsiao'):
        model = sncosmo.Model(source=source)
        model.set(z=0.001, t0=0., amplitude=1)
        model.set_source_peakabsmag(-19.2, 'bessellb', 'ab')

        self.times = np.linspace(-20,100,100) * 86400.0
        self.wavelengths = np.linspace(2500, 20000,1000)
        self.fluxes = model.flux(self.times / 86400, self.wavelengths)* 4.0 * np.pi * (4.3 * 3.086e24)**2

class PlasticcModel(TransientModel):
    def __init__(self):
        """
        """
    
    def load(self, filename, dirname):

        data = np.load(dirname+filename)
        self.times = data['times'] * 86400
        self.wavelengths = data['wvs']
        self.fluxes = data['fluxes']


class SedonaModel(TransientModel):
    """
    A basic Sedona model
    """

    def __init__(self):
        """
        Parameters:
        ----------
        ...
        times
        wavelengths
        fluxes

        """
        self.times = []
        self.fluxes = []
        self.wavelengths = []
    def load(self, filename, dirname):
        all_models = []
        model_file = dirname+filename
        time_model, wavelength_model, flux_model = np.loadtxt(model_file, \
                                                    skiprows=1, unpack=True)
        self.times = np.unique(time_model)
        self.wavelengths =np.unique(wavelength_model)
        # Create an empty 2D array to hold fluxes
        self.fluxes = np.zeros((len(self.times), len(self.wavelengths)))

        # Map the original time and wavelength arrays to the unique indices
        time_indices = np.searchsorted(self.times, time_model)
        wavelength_indices = np.searchsorted(self.wavelengths, wavelength_model)

        # Populate the flux grid by mapping each flux value to its (time, wavelength) position
        for t_idx, w_idx, flux in zip(time_indices, wavelength_indices, flux_model):
            self.fluxes[t_idx, w_idx] = flux



    def generate(time, param):
        return 0

    def sample():
        return 0

class ModelGrid():
    def __init__(self, myModel, dirname):
        self.myModel = myModel
        self.dirname = dirname
        self.modelList = [f for f in listdir(dirname) if isfile(join(dirname, f))]

    def sample(self):
        my_selection = int(np.random.choice(np.arange(len(self.modelList)), 1))
        self.myModel.load(self.modelList[my_selection], self.dirname)
        return self.myModel

class AnalyticalModel():
    def __init__(self, myModel):
        self.myModel = myModel

    def sample(self):
        self.times = 0
        my_model = self.myModel()
        my_model.sample()
        return my_model



