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

from SSBoost.helpers import (check_response_type, compute_alphas, compute_qvalues, integrate, 
	return_null_result, score_based_selection, selector_and_args)
from SSBoost.preselection import preselection

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



