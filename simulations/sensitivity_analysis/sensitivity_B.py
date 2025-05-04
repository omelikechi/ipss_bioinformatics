# Sensitivity analyses for Nonparametric IPSS methods

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["XGBOOST_NUM_THREADS"] = "1"

import logging
import sys

import numpy as np
import pickle

from ipss_sensitivity.main import ipss_sensitivity

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from generate_data import generate_data

#--------------------------------
# Simulation setup
#--------------------------------
################################################################
save_results = True
################################################################
n_trials = 100
################################################################
feature_matrix = np.load('ovarian_rnaseq.npy')
################################################################
experiment_name = f'oc'
################################################################
response_types = ['reg']
################################################################
p_grid = [5000]
################################################################
method_name = 'ipssgb'
################################################################
parameter = 'B'
################################################################
B_grid = [50, 100, 150, 200]
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
			'p_true':[10, 30],
			'snr':[1/2,2] if response_type == 'reg' else [1,3],
			'fdr_list':np.linspace(0, 0.5, 51),
			'function_grid':['h3'],
			'cutoff_grid':[0.05],
			'delta_grid':[5/4] if response_type == 'reg' else [1],
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

		results = {B: {} for B in B_grid}

		for B in B_grid:

			if save_results:
				logging.info(f'B = {B}')
			else:
				print(f'B = {B}')

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
					logging.info(f' trial {trial + 1}/{n_trials}')
				else:
					print(f' trial {trial + 1}/{n_trials}')

				# Update random seed
				trial_seed = random_seed + trial

				# Generate data
				X, y, true_features = generate_data(n, p, p_true, snr, response_type, feature_matrix, random_seed=trial_seed)

				# Run method
				preselector_args = {'expansion_factor':1.25}
				result = ipss_sensitivity(X, y, true_features, fdr_list, selector=selector, preselector_args=preselector_args,
					function_grid=function_grid, cutoff_grid=cutoff_grid, delta_grid=delta_grid, B=B)
				runtimes.append(result['runtime'])
				for function in function_grid:
					for cutoff in cutoff_grid:
						for delta in delta_grid:
							tpr_matrix[function][cutoff][delta][trial,:] = result['tprs'][function][cutoff][delta]
							fdr_matrix[function][cutoff][delta][trial,:] = result['fdrs'][function][cutoff][delta]

			results[B]['tpr_matrix'] = tpr_matrix
			results[B]['fdr_matrix'] = fdr_matrix
			results[B]['runtimes'] = runtimes
			results[B]['simulation_config'] = simulation_config

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

			for B in B_grid:
				simulation_config = results[B]['simulation_config']
				fdr_list = simulation_config['fdr_list']
				function_grid = simulation_config['function_grid']
				cutoff_grid = simulation_config['cutoff_grid']
				delta_grid = simulation_config['delta_grid']

				tpr_matrix = results[B]['tpr_matrix']
				fdr_matrix = results[B]['fdr_matrix']
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





