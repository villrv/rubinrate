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

        with gzip.open(dirname+filename,'rt') as f:
            data = np.loadtxt(f)

        self.times = data[:,0] * 86400.0
        self.wavelengths = data[:,1]
        self.fluxes = data[:,2]


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
    def load(self, filename):
        all_models = []
        model_file = './data/'+filename
        time_model, wavelength_model, flux_model = np.loadtxt(model_file, \
                                                    skiprows=1, unpack=True)
        self.times = time_model
        self.fluxes = flux_model
        self.wavelengths = wavelength_model

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



