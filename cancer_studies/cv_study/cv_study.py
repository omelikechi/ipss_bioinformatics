# Parsimony of feature selection algorithms on real data

import logging
import os
import sys

import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, StratifiedKFold

# from data_functions import load_data
from cv_helpers import compute_gb_errors, compute_linear_errors, compute_rf_errors

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from methods import run_method
from stability_selection.ss import compute_q_list

#--------------------------------
# Setup
#--------------------------------
################################################################
save_results = True
################################################################
cancer_type = 'ovarian'
feature_type = ['mirna']
response = [('clinical', 'Tumor_purity')]
expression_threshold = None if cancer_type == 'ovarian' else 50
variance_threshold = None if cancer_type == 'ovarian' else 0
################################################################
experiment_name = f'{cancer_type}_{feature_type[0]}_{response[0][1]}'
################################################################
n_folds = 20
################################################################
fdr_list = np.arange(0, 0.51, 0.01)
# fdr_list2 = np.arange(0.075, 0.21, 0.025)
# fdr_list3 = np.arange(0.25, 0.96, 0.05)
# fdr_list = np.concatenate([fdr_list1, fdr_list2, fdr_list3])
################################################################
method_list = ['all_features']
################################################################

# random seed
seed = 302
np.random.seed(seed)

#--------------------------------
# Method args
#--------------------------------
method_args = {}
expansion_factor = 2.5 if response[0][0] == 'clinical' else 1.5
n_keep = 100 #if response[0][0] == 'clinical' and feature_type[0] == 'rnaseq' else 100

method_args['all_features'] = {}
method_args['boruta'] = {}
method_args['deeppink'] = {'preselect':True, 'preselector':'rf', 'n_keep':100}
method_args['ipssgb'] = {'B':100, 'preselector_args':{'expansion_factor':expansion_factor}}
method_args['ipssl'] = {'B':50}
method_args['ipssrf'] = {'B':50, 'preselector_args':{'n_keep':n_keep}}
method_args['koglm'] = {'preselect':True, 'preselector':'rf', 'n_keep':200}
method_args['kol'] = {'preselect':True, 'preselector':'l1', 'n_keep':200}
method_args['korf'] = {'preselect':True, 'preselector':'rf', 'n_keep':200}
method_args['vita'] = {}
method_args['vsurf'] = {}

# ssboost args
if 'ssboost' in method_list:
	assumption = 'r-concave'
	efp_list = np.linspace(0, 10, 21)
	n_keep = 150
	B = 100
	tau = 0.75
	q_list = compute_q_list(efp_list, tau=tau, method=assumption, p=n_keep, B=B)
	method_args['ssboost'] = {'assumption':assumption, 'efp_list':efp_list, 'n_keep':n_keep, 'B':B, 'tau':tau, 'q_list':q_list}

#--------------------------------
# Start experiment
#--------------------------------
verbose = True
if save_results:
	logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(message)s')
	verbose = False
	logging.info(f'\nStarting {experiment_name}')
	logging.info(f'--------------------------------')

# load data
if save_results:
	sys.path.append('/n/home11/omelikechi/data/tcga')
else:
	sys.path.append('/Users/omm793/iCloud/code/research/data/tcga')
from load_cancer_data import load_data

data = load_data(cancer_type, feature_type, responses=response, expression_threshold=expression_threshold, 
	variance_threshold=variance_threshold, verbose=verbose)
X, y, feature_names = data['X'], data['Y'], data['feature_names']
y = y.ravel()

print(f'\nn = {X.shape[0]}')
print(f'p = {X.shape[1]}')

# metadata
metadata = {
	'cancer_type':cancer_type, 'feature_type':feature_type[0], 'feature_names':feature_names, 'response':response[0], 
	'n':X.shape[0], 'p':X.shape[1], 'n_folds':n_folds, 'fdr_list':fdr_list, 'random_seed':seed
}

binary_response = len(np.unique(y)) <= 2

# start cross-validation
n_fdrs = len(fdr_list)
results = {method: {} for method in method_list}
for method in method_list:

	if save_results:
		logging.info(f'\nStarting {method}')
	else:
		print(f'\nStarting {method}')
		print(f'--------------------------------')

	# initialize arrays
	Q = np.ones((n_folds, X.shape[1]))
	n_selected = np.zeros((n_folds, n_fdrs))
	gb_errors = np.zeros((n_folds, n_fdrs))
	linear_errors = np.zeros((n_folds, n_fdrs))
	rf_errors = np.zeros((n_folds, n_fdrs))
	runtimes = np.zeros(n_folds)

	# construct folds for cross-validation
	if binary_response:
		kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
		folds = kf.split(X, y)
	else:
		kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
		folds = kf.split(X)

	for fold, (train_index, test_index) in enumerate(folds):
		if save_results:
			logging.info(f'fold {fold + 1}/{n_folds}')
		else:
			print(f'fold {fold + 1}/{n_folds}')

		# split data
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		if not binary_response:
			y_train_mean, y_train_std = np.mean(y_train), np.std(y_train)
			y_train = (y_train - y_train_mean) / y_train_std
			y_test = (y_test - y_train_mean) / y_train_std

		# run method
		if method != 'all_features':
			output = run_method(X_train, y_train, fdr_list, method, kwargs=method_args[method])
			q_values = output['q_values']
			runtimes[fold] = output['runtime']
			# print(output['runtime'])

			for feature, q_value in q_values.items():
				Q[fold, feature] = q_value

			for fdr_idx, fdr in enumerate(fdr_list):
				selected_features = np.where(Q[fold,:] < fdr)[0]
				n_selected[fold,fdr_idx] = len(selected_features)
				gb_errors[fold,fdr_idx] = compute_gb_errors(X_train, X_test, y_train, y_test, selected_features, binary_response)
				linear_errors[fold,fdr_idx] = compute_linear_errors(X_train, X_test, y_train, y_test, selected_features, binary_response)
				rf_errors[fold,fdr_idx] = compute_rf_errors(X_train, X_test, y_train, y_test, selected_features, binary_response)

		else:
			selected_features = np.arange(X.shape[1])
			gb_error = compute_gb_errors(X_train, X_test, y_train, y_test, selected_features, binary_response)
			linear_error = compute_linear_errors(X_train, X_test, y_train, y_test, selected_features, binary_response)
			rf_error = compute_rf_errors(X_train, X_test, y_train, y_test, selected_features, binary_response)

			for fdr_idx, fdr in enumerate(fdr_list):
				gb_errors[fold,fdr_idx] = gb_error
				linear_errors[fold,fdr_idx] = linear_error
				rf_errors[fold,fdr_idx] = rf_error

	results[method]['metadata'] = metadata.copy()
	results[method]['method_args'] = method_args[method]
	results[method]['Q'] = Q
	results[method]['n_selected'] = n_selected
	results[method]['gb_errors'] = gb_errors
	results[method]['linear_errors'] = linear_errors
	results[method]['rf_errors'] = rf_errors
	results[method]['runtimes'] = runtimes

	if save_results:
		file_name = f'{method}_{experiment_name}.pkl'
		with open(file_name, 'wb') as f:
			pickle.dump(results[method], f)

if not save_results:

	# q-values
	for method in method_list:
		print(f'\n{method}')
		print(f'--------------------------------')
		feature_names = results[method]['metadata']['feature_names']
		Q = results[method]['Q']
		median_q_value = np.median(Q, axis=0)
		sorted_features = np.argsort(median_q_value)
		for feature in sorted_features:
			if median_q_value[feature] < 1:
				print(f'{feature_names[feature]}: {median_q_value[feature]:.2f}')

	# plot results
	fdr_start_idx = 0
	fig, ax = plt.subplots(1, 3, figsize=(18, 6))

	for method in method_list:
		mean_n_selected = np.mean(results[method]['n_selected'], axis=0)
		mean_gb_error = np.mean(results[method]['gb_errors'], axis=0)
		mean_linear_error = np.mean(results[method]['linear_errors'], axis=0)
		mean_rf_error = np.mean(results[method]['rf_errors'], axis=0)
		best_error = np.minimum.reduce([mean_gb_error, mean_linear_error, mean_rf_error])

		ax[0].plot(fdr_list[fdr_start_idx:], mean_n_selected[fdr_start_idx:], label=method)
		ax[1].plot(fdr_list[fdr_start_idx:], best_error[fdr_start_idx:], label=method)
		ax[2].plot(mean_n_selected[fdr_start_idx:], best_error[fdr_start_idx:], label=method)

	ax[0].set_xlabel('Target FDR')
	ax[0].set_ylabel('Number of selected features')
	# ax[0].set_title('Parsimony Across Methods')
	# ax[0].legend()
	ax[0].grid(True)

	ax[1].set_xlabel('Target FDR')
	ax[1].set_ylabel('Prediction Error')
	# ax[1].set_title('Prediction error vs')
	# ax[1].legend()
	ax[1].grid(True)

	ax[2].set_xlabel('Number of selected features')
	ax[2].set_ylabel('Prediction Error')
	# ax[2].set_title('Prediction Error Across Methods')
	ax[2].legend()
	ax[2].grid(True)

	plt.tight_layout()
	plt.show()










