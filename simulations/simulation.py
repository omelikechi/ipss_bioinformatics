# Simulation experiments (FDR control)

import logging
import sys

import numpy as np
import pickle
from scipy.linalg import toeplitz

from generate_data import generate_data
from methods import run_method
from SSBoost.ssboost import compute_q_list

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
"""
################################################################
save_results = False
################################################################
n_trials = 100
################################################################
feature_matrix = np.load('./data/ovarian_rnaseq.npy')
################################################################
experiment_name = f'oc'
################################################################
response_types = ['reg', 'class']
################################################################
group_features = True if experiment_name == 'oc' else False
################################################################
p_grid = [500, 2000, 5000]
################################################################
method_name = 'koglm'
################################################################

#--------------------------------
# Method args
#--------------------------------
"""
Specific arguments for each individual method
"""

method_args = {}
method_args['boruta'] = {}
method_args['deeppink'] = {'preselect':False, 'preselector':'rf', 'n_keep':100}
method_args['ipssgb'] = {}
method_args['ipssl'] = {}
method_args['ipssrf'] = {}
method_args['kobt'] = {'preselect':True}
method_args['koglm'] = {'preselect':True, 'preselector':'rf', 'n_keep':100}
method_args['kol'] = {'preselect':True, 'preselector':'l1', 'n_keep':200}
method_args['korf'] = {'preselect':True, 'preselector':'rf', 'n_keep':200}
method_args['vita'] = {}
method_args['vsurf'] = {}

# ssboost args (takes 30 to 90 seconds to compute when assumption = 'r-concave')
if method_name == 'ssboost':
	assumption = 'r-concave'
	efp_list = np.linspace(0, 10, 41)
	n_keep = 150
	B = 100
	tau = 0.75
	q_list = compute_q_list(efp_list, tau=tau, method=assumption, p=n_keep, B=B)
	method_args['ssboost'] = {'assumption':assumption, 'efp_list':efp_list, 'n_keep':n_keep, 'B':B, 'tau':tau, 'q_list':q_list}

#--------------------------------
# Run simulation
#--------------------------------
if save_results:
	logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s')

for response_type in response_types:
	for p in p_grid:

		# set random seed
		random_seed = 302
		np.random.seed(random_seed)

		# simulation name
		simulation_name = f'{experiment_name}_{response_type}_{p}_{method_name}'

		# simulation parameters
		simulation_config = {
			'simulation_name':simulation_name,
			'n_trials':n_trials,
			'p':p,
			'n':500,
			'p_true':[10, 30] if experiment_name == 'oc' else [5,15],
			'snr':[1/2,2] if response_type == 'reg' else [1,3],
			'fdr_list':np.linspace(0, 0.5, 51),
			'random_seed':random_seed
		}

		# unpack variables into local scope
		locals().update(simulation_config)

		# run simulation
		if save_results:
			logging.info(f'Starting {simulation_name}')
			logging.info(f'--------------------------------')
		else:
			print(f'Starting {simulation_name}')
			print(f'--------------------------------')

		results = {}

		fdr_matrix = np.zeros((n_trials, len(fdr_list)))
		tpr_matrix = np.zeros((n_trials, len(fdr_list)))
		runtimes = []

		# provide knockoff methods with true mean and covariance matrix when known
		if method_name in ['deeppink', 'koglm', 'kol', 'korf']:
			if isinstance(feature_matrix, (int, float)) and 0 <= feature_matrix <= 1:
				mu = np.zeros(p)
				first_col = feature_matrix ** np.arange(p)
				Sigma = toeplitz(first_col)
				method_args[method_name]['mu'] = mu
				method_args[method_name]['Sigma'] = Sigma
			else:
				method_args[method_name]['mu'] = None
				method_args[method_name]['Sigma'] = None

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
			result = run_method(X, y, true_features, fdr_list, method_name, kwargs=method_args[method_name])
			tpr_matrix[trial, :] = result['tprs']
			fdr_matrix[trial, :] = result['fdrs']
			runtimes.append(result['runtime'])

		results['tpr_matrix'] = tpr_matrix
		results['fdr_matrix'] = fdr_matrix
		results['runtimes'] = runtimes
		results['simulation_config'] = simulation_config
		results['method_args'] = method_args[method_name]

		if save_results:
			logging.info(f'Finished {simulation_name}')

		if save_results:
			file_name = simulation_name + '_preRF100.pkl'
			with open(file_name, 'wb') as f:
				pickle.dump(results, f)

		else:
			import matplotlib.pyplot as plt

			runtime_mean = np.mean(runtimes)
			print(f'Average runtime = {runtime_mean:.1f} seconds')

			tpr_mean = np.mean(tpr_matrix, axis=0)
			fdr_mean = np.mean(fdr_matrix, axis=0)

			fig, ax = plt.subplots(2, 1, figsize=(14,6))
			ax[0].plot(fdr_list, tpr_mean)
			ax[1].plot(fdr_list, fdr_mean)
			ax[1].plot(fdr_list, fdr_list, linestyle='--', color='k')
			plt.tight_layout()
			plt.show()





