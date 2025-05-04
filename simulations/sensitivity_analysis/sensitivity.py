# Sensitivity analyses for IPSSGB and IPSSRF

import logging

import numpy as np
import pickle

from ipss_sensitivity.main import ipss_sensitivity

import sys
sys.path.append('..')
from generate_data import generate_data

#--------------------------------
# Simulation setup
#--------------------------------
"""
- n_trials: Number of trials to run
- feature_matrix: Number between 0 and 1 to simulate Gaussian data with Toeplitz covariance 
	matrix (the number is the correlation parameter); np.load(f'./data/ovarian_rnaseq.npy') 
	to use the ovarian cancer RNA-sequencing data
- experiment_name: 'oc' for ovarian cancer simulations; 'toeplitz_0.5' for, e.g., the Toeplitz
	simulation with correlation parameter 0.5.
- response_types: 'reg' for regression; 'class' for classification
- group features: True if features are grouped together (non-additive model; this is what's
	used in the RNA-seq simulations); False for an additive model (thisi s what's used in the
	Gaussian simulations) 
- p_grid: Grid of feature dimensions; one experiment consisting of n_trials trials will be
	performed for each p in p_grid
- method_name: Method to run
- parameter: Which parameter to test; only used for naming the simulation
"""
################################################################
save_results = False
################################################################
n_trials = 100
################################################################
feature_matrix = np.load(f'./data/ovarian_rnaseq.npy')
################################################################
experiment_name = f'oc'
################################################################
response_types = ['reg', 'class']
################################################################
group_features = True if experiment_name == 'oc' else False
################################################################
p_grid = [500, 2000, 5000]
################################################################
method_name = 'ipssgb'
################################################################
parameter = 'delta'
################################################################

if save_results:
	logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s')

for response_type in response_types:
	for p in p_grid:

		# set random seed
		random_seed = 302
		np.random.seed(random_seed)

		simulation_name = f'{experiment_name}_{response_type}_{p}_{method_name}_{parameter}'

		selector = 'gb' if method_name == 'ipssgb' else 'rf'

		# simulation parameters
		simulation_config = {
			'simulation_name':simulation_name,
			'n_trials':n_trials,
			'p':p,
			'n':500,
			'p_true':[5, 15],
			'snr':[1/2,2] if response_type == 'reg' else [1,3],
			'fdr_list':np.linspace(0, 0.5, 51),
			'function_grid':['h3'],
			'cutoff_grid':[0.05],
			'delta_grid':[0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5],
			'random_seed':random_seed
		}

		# unpack variables into local scope
		locals().update(simulation_config)

		# Run simulation
		if save_results:
			logging.info(f'Starting {simulation_name}')
			logging.info(f'--------------------------------')
		else:
			print(f'Starting {simulation_name}')
			print(f'--------------------------------')

		results = {}

		fdr_matrix = {
				function: {
					cutoff: {
						delta: np.zeros((n_trials, len(fdr_list))) for delta in delta_grid
					} for cutoff in cutoff_grid
				} for function in function_grid
			}

		tpr_matrix = {
				function: {
					cutoff: {
						delta: np.zeros((n_trials, len(fdr_list))) for delta in delta_grid
					} for cutoff in cutoff_grid
				} for function in function_grid
			}

		runtimes = []

		for trial in range(n_trials):

			if save_results:
				logging.info(f'trial {trial + 1}/{n_trials}')
			else:
				print(f'trial {trial + 1}/{n_trials}')

			# Update random seed
			trial_seed = random_seed + trial

			# Generate data
			X, y, true_features = generate_data(n, p, p_true, snr, response_type, feature_matrix, 
				group_features=group_features, random_seed=trial_seed)

			# Run method
			selector_args = None #{'importance_type':'permutation'}
			preselector_args = None #{'expansion_factor':1.35}
			result = ipss_sensitivity(X, y, true_features, fdr_list, selector=selector, selector_args=selector_args, 
				preselector_args=preselector_args, function_grid=function_grid, cutoff_grid=cutoff_grid, delta_grid=delta_grid)
			runtimes.append(result['runtime'])
			for function in function_grid:
				for cutoff in cutoff_grid:
					for delta in delta_grid:
						tpr_matrix[function][cutoff][delta][trial, :] = result['tprs'][function][cutoff][delta]
						fdr_matrix[function][cutoff][delta][trial, :] = result['fdrs'][function][cutoff][delta]

		results['tpr_matrix'] = tpr_matrix
		results['fdr_matrix'] = fdr_matrix
		results['runtimes'] = runtimes
		results['simulation_config'] = simulation_config

		if save_results:
			logging.info(f'Finished {simulation_name}')

		if save_results:
			file_name = simulation_name + '.pkl'
			with open(file_name, 'wb') as f:
				pickle.dump(results, f)

		else:
			import matplotlib.pyplot as plt

			runtime_mean = np.mean(runtimes)
			print(f'Average runtime = {runtime_mean:.1f} seconds')

			fig, ax = plt.subplots(2, 1, figsize=(14,6))

			for function in function_grid:
				linestyle = '--' if function == 'h2' else '-'
				for cutoff in cutoff_grid:
					for delta in delta_grid:
						tpr_means = np.mean(tpr_matrix[function][cutoff][delta], axis=0)
						fdr_means = np.mean(fdr_matrix[function][cutoff][delta], axis=0)
						ax[0].plot(fdr_list, tpr_means, linestyle=linestyle, lw=2, label=f'{function}')
						ax[1].plot(fdr_list, fdr_means, linestyle=linestyle, lw=2, label=f'{function}')
			ax[1].plot(fdr_list, fdr_list, linestyle='--', lw=2, color='k')

			plt.tight_layout()
			plt.show()





