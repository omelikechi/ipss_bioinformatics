# Integrated path stability selection for sensitvity analysis

import time
import warnings

from joblib import Parallel, delayed
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso, lasso_path, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from ipss_sensitivity.helpers import (check_response_type, compute_alphas, compute_qvalues, integrate, 
	return_null_result, score_based_selection, selector_and_args)
from ipss_sensitivity.preselection import preselection

#--------------------------------
# IPSS: Sensitivity to parameters
#--------------------------------
def ipss_sensitivity(X, y, true_features, fdr_list, selector='gb', selector_args=None, preselect=True, preselector_args=None,
		target_fp=None, target_fdr=None, B=None, n_alphas=None, function_grid=['h3'], cutoff_grid=[0.05], delta_grid=[1], 
		standardize_X=None, center_y=None, n_jobs=1, verbose=False):

	tprs = {
		function: {
			cutoff: {
				delta: [] for delta in delta_grid
			} for cutoff in cutoff_grid
		} for function in function_grid
	}

	fdrs = {
		function: {
			cutoff: {
				delta: [] for delta in delta_grid
			} for cutoff in cutoff_grid
		} for function in function_grid
	}

	# start timer
	start = time.time()

	p_full = X.shape[1]
	p_true = len(true_features)

	# number of subsamples
	B = B if B is not None else 100 if selector == 'gb' else 50

	output = compute_stability_paths(X, y, selector=selector, selector_args=selector_args, preselect=preselect, 
		preselector_args=preselector_args, B=B, n_alphas=n_alphas, standardize_X=standardize_X, center_y=center_y, n_jobs=n_jobs)

	stability_paths = output['stability_paths']
	alphas = output['alphas']
	average_selected = output['average_selected']
	preselect_indices = output['preselect_indices']
	p = len(preselect_indices)

	first_pass = True

	for function in function_grid:
		for cutoff in cutoff_grid:
			for delta in delta_grid:
				scores, integral, _, _ = ipss_scores(stability_paths, B, alphas, average_selected, function, delta, cutoff)
				efp_scores = np.round(integral / np.maximum(scores, integral / p), decimals=8)
				efp_scores = dict(zip(preselect_indices, efp_scores))
				# reinsert features removed during preselection
				if p_full != p:
					all_features = set(range(p_full))
					missing_features = all_features - efp_scores.keys()
					for feature in missing_features:
						efp_scores[feature] = p
					efp_scores = {feature: (p_full if score >= p - 1 else score) for feature, score in efp_scores.items()}

				# q_values
				q_values = compute_qvalues(efp_scores)

				if first_pass:
					runtime = time.time() - start
					first_pass = False

				for target_fdr in fdr_list:
					tp, fp = 0, 0
					for j in range(p_full):
						if q_values[j] <= target_fdr:
							if j in true_features:
								tp += 1
							else:
								fp += 1
					fdr = 0 if tp + fp == 0 else fp / (fp + tp)
					tpr = tp / p_true
					tprs[function][cutoff][delta].append(tpr)
					fdrs[function][cutoff][delta].append(fdr)

	return { 
		'tprs': tprs,
		'fdrs':fdrs,
		'runtime': runtime, 
		}


# compute stability paths
def compute_stability_paths(X, y, selector, selector_args, preselect, preselector_args, B, n_alphas, standardize_X, center_y, n_jobs):
	# empty set for selector args if none specified
	selector_args = selector_args or {}

	# reshape response
	if len(y.shape) > 1:
		y = y.ravel()
	
	# check response type
	binary_response, selector = check_response_type(y, selector)

	# standardize and center data if using l1 selectors
	if selector in ['lasso', 'logistic_regression']:
		if standardize_X is None:
			X = StandardScaler().fit_transform(X)
		if center_y is None:
			if not binary_response:
				y -= np.mean(y)

	# preselect features to reduce dimension
	p_full = X.shape[1]
	if preselect:
		X, preselect_indices = preselection(X, y, selector, preselector_args)
		if preselect_indices.size == 0:
			output = return_null_result(p_full)
			warnings.warn('Preselection step removed all features. Returning null result.', UserWarning)
			return output
	else:
		preselect_indices = np.arange(p_full)
	
	# dimensions post-preselection
	n, p = X.shape
	
	# maximum number of features for l1 regularized selectors (to avoid computational issues)
	max_features = 0.75 * p if selector in ['lasso', 'logistic_regression'] else None

	# alphas
	if n_alphas is None:
		n_alphas = 25 if selector in ['lasso', 'logistic_regression'] else 100
	alphas = compute_alphas(X, y, n_alphas, max_features, binary_response) if selector in ['lasso', 'logistic_regression'] else None

	# selector function and args
	selector_function, selector_args = selector_and_args(selector, selector_args)

	# estimate selection probabilities
	results = np.array(Parallel(n_jobs=n_jobs)(delayed(selection)(X, y, alphas, selector_function, **selector_args) for _ in range(B)))

	# score-based selection
	if alphas is None:
		results, alphas = score_based_selection(results, n_alphas)

	# aggregate results
	Z = np.zeros((n_alphas, 2*B, p))
	for b in range(B):
		Z[:, 2*b:2*(b + 1), :] = results[b,:,:,:]

	# average number of features selected (denoted q in ipss papers)
	average_selected = np.array([np.mean(np.sum(Z[i,:,:], axis=1)) for i in range(n_alphas)])

	# stability paths
	stability_paths = np.empty((n_alphas,p))
	for i in range(n_alphas):
		stability_paths[i] = Z[i].mean(axis=0)

	# stop if all stability paths stop changing (after burn-in period where mean selection probability < 0.01)
	stop_index = n_alphas
	for i in range(2,len(alphas)):
		if np.isclose(stability_paths[i,:], np.zeros(p)).all():
			continue
		else:
			diff = stability_paths[i,:] - stability_paths[i-2,:]
			mean = np.mean(stability_paths[i,:])
			if np.isclose(diff, np.zeros(p)).all() and mean > 0.01:
				stop_index = i
				break

	# truncate stability paths at stop index
	stability_paths = stability_paths[:stop_index,:]
	alphas = alphas[:stop_index]
	average_selected = average_selected[:stop_index]

	return {'stability_paths':stability_paths, 'average_selected':average_selected, 'alphas':alphas, 'preselect_indices':preselect_indices}

# compute ipss scores and theoretical E(FP) bounds
def ipss_scores(stability_paths, B, alphas, average_selected, ipss_function, delta, cutoff):
	n_alphas, p = stability_paths.shape

	if ipss_function not in ['h1', 'h2', 'h3']:
		raise ValueError(f"ipss_function must be 'h1', 'h2', or 'h3', but got ipss_function = {ipss_function} instead")

	m = 1 if ipss_function == 'h1' else 2 if ipss_function == 'h2' else 3

	# function to apply to selection probabilities
	def h_m(x):
		return 0 if x <= 0.5 else (2*x - 1)**m

	# evaluate ipss bounds for specific functions
	if m == 1:
		integral, stop_index = integrate(average_selected**2 / p, alphas, delta, cutoff=cutoff)
	elif m == 2:
		term1 = average_selected**2 / (p * B)
		term2 = (B-1) * average_selected**4 / (B * p**3)
		integral, stop_index  = integrate(term1 + term2, alphas, delta, cutoff=cutoff)
	else:
		term1 = average_selected**2 / (p * B**2)
		term2 = (3 * (B-1) * average_selected**4) / (p**3 * B**2)
		term3 = ((B-1) * (B-2) * average_selected**6) / (p**5 * B**2)
		integral, stop_index = integrate(term1 + term2 + term3, alphas, delta, cutoff=cutoff)

	# compute ipss scores
	alphas_stop = alphas[:stop_index]
	scores = np.zeros(p)
	for i in range(p):
		values = np.empty(stop_index)
		for j in range(stop_index):
			values[j] = h_m(stability_paths[j,i])
		scores[i], _ = integrate(values, alphas_stop, delta)

	return scores, integral, alphas, stop_index

# subsampler for estimating selection probabilities
def selection(X, y, alphas, selector, **kwargs):
	n, p = X.shape
	indices = np.arange(n)
	np.random.shuffle(indices)
	n_split = int(len(indices) / 2)

	if alphas is None:
		indicators = np.empty((2,p))
		for half in range(2):
			idx = indices[:n_split] if half == 0 else indices[n_split:]
			indicators[half, :] = np.array(selector(X[idx,:], y[idx], **kwargs))
	else:
		indicators = np.empty((len(alphas), 2, p))
		for half in range(2):
			idx = indices[:n_split] if half == 0 else indices[n_split:]
			indicators[:, half, :] = np.array(selector(X[idx,:], y[idx], alphas, **kwargs))

	return indicators



