#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
import importlib
import sys

# Add the parent directory to the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rubinrate.sim import run_sim
from rubinrate.rates import calc_rate, kessler_ia, rate_madau, rate_strolger
from rubinrate.metrics import *
from rubinrate.dust_models import *
from rubinrate.utils import *

def load_config(config_file):
    """Load configuration from JSON file"""
    with open(config_file, 'r') as f:
        return json.load(f)

def create_model(config):
    """Create model based on configuration"""
    model_type = config['model']['type']
    params = config['model']['params']
    
    # Import the right module based on model type
    if model_type == 'SNCosmoModel':
        from rubinrate.models import SNCosmoModel
        model = SNCosmoModel()
        model.load()
    elif model_type == 'GaussianModel':
        from rubinrate.analytical_models.gaussian import GaussianModel
        model = GaussianModel(**params)
    elif model_type == 'ArnettModel':
        from rubinrate.analytical_models.arnett import ArnettModel
        model = ArnettModel(**params)
    elif model_type == 'PlasticcModel':
        from rubinrate.models import PlasticcModel
        model = PlasticcModel()
        if 'file' in params:
            model.load(params['file'], params.get('directory', './data/'))
    elif model_type == 'SedonaModel':
        from rubinrate.models import SedonaModel
        model = SedonaModel()
        if 'file' in params:
            model.load(params['file'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def create_dust_model(config):
    """Create dust model based on configuration"""
    dust_type = config['dustModel']['type']
    params = config['dustModel']['params']
    
    if dust_type == 'LogUniformDust':
        return LogUniformDust(params['minEBV'], params['maxEBV'])
    else:
        raise ValueError(f"Unknown dust model type: {dust_type}")

def create_metric_list(config):
    """Create metric list based on configuration"""
    metric_list = {}
    for name, (metric_name, params) in config['metric_list'].items():
        # Get the metric function from the globals
        metric_func = globals()[metric_name]
        metric_list[name] = (metric_func, tuple(params))
    return metric_list

def get_rate_function(rate_model):
    """Get the appropriate rate function"""
    if rate_model == 'kessler_ia':
        return kessler_ia
    elif rate_model == 'rate_madau':
        return rate_madau
    elif rate_model == 'rate_strolger':
        return rate_strolger
    else:
        raise ValueError(f"Unknown rate model: {rate_model}")

def main():
    parser = argparse.ArgumentParser(description='Run RubinRate simulations with JSON config')
    parser.add_argument('config_file', help='Path to JSON configuration file')
    parser.add_argument('--outdir', help='Output directory (overrides config)', default=None)
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config_file)
    
    # Override outdir if specified
    if args.outdir:
        config['outdir'] = args.outdir
    if config['outdir'][-1] != '/':
        config['outdir'] += '/'
    
    # Create output directory if it doesn't exist
    if not os.path.exists(config['outdir']):
        os.makedirs(config['outdir'])
    
    # Create model and dust model
    model = create_model(config)
    dust_model = create_dust_model(config)
    
    # Create metric list
    metric_list = create_metric_list(config)
    
    # Set up redshifts
    if config['redshifts']['kind'] == 'linear':
        z_min, z_max, n_points = config['redshifts']['values']
        redshifts = np.linspace(z_min, z_max, int(n_points))
    
    # Run simulation
    print(f"Running simulation with {config['model']['type']} model...")
    metric_tracker = run_sim(
        metric_list, 
        model, 
        redshifts, 
        config['patience'], 
        config['bigN'], 
        dust_Model=dust_model, 
        keep_LCs=True
    )
    
    # Calculate efficiencies
    efficiencies = calc_efficiences(metric_tracker)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(redshifts, efficiencies)
    plt.xlabel('Redshift')
    plt.ylabel('Efficiency')
    plt.title(f"{config['model']['type']} Efficiency vs Redshift")
    plt.grid(True)
    plt.savefig(os.path.join(config['outdir'], f"{config['filename']}_efficiency.png"))
    plt.show()
    
    # Calculate rates
    if 'rate_model' in config:
        rate_func = get_rate_function(config['rate_model'])
        
        # Check if rate_z0 is specified
        rate_kwargs = {}
        if 'rate_z0' in config:
            rate_kwargs['rate_z0'] = config['rate_z0']
        
        integrand, total_rate = calc_rate(redshifts, efficiencies, rate_func, **rate_kwargs)
        
        print(f"Rate model: {config['rate_model']}")
        print(f"Total rate: {total_rate}")
    
    # Save results if configured
    if config.get('save', True):
        output_file = os.path.join(config['outdir'], f"{config['filename']}.npz")
        np.savez(
            output_file,
            redshifts=redshifts,
            metric_tracker=metric_tracker,
            metric_list=metric_list,
            efficiencies=efficiencies
        )
        print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main() 