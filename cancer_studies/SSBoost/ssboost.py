# Integrated path stability selection, beta version

import time
import warnings

from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, lasso_path, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from SSBoost.helpers import compute_qvalues
from SSBoost.main import compute_stability_paths

from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import IntVector, FloatVector, StrVector, BoolVector, NULL
stabs = importr('stabs')

#--------------------------------
# Main function
#--------------------------------
def ssboost(X, y, selector='gb', selector_args=None, preselect=True, preselector_args=None, assumption='r-concave', efp_list=None,
		q_list=None, tau=0.75, B=None, n_alphas=None, standardize_X=None, center_y=None, n_jobs=1):

	# start timer
	start = time.time()

	p_full = X.shape[1]
	B = B if B is not None else 100 if selector == 'gb' else 50

	output = compute_stability_paths(X, y, selector=selector, selector_args=selector_args, preselect=preselect, 
		preselector_args=preselector_args, B=B, n_alphas=n_alphas, standardize_X=standardize_X, center_y=center_y, n_jobs=n_jobs)

	stability_paths = output['stability_paths']
	average_selected = output['average_selected']
	preselect_indices = output['preselect_indices']

	n_alphas, p = stability_paths.shape

	efp_scores = p * np.ones(p)
	already_selected = []
	for idx, efp in enumerate(efp_list):
		if efp == 0:
			continue
		q = q_list[idx]
		mask = average_selected <= q
		stop_index = np.where(mask)[0][np.argmax(average_selected[mask])] if np.any(mask) else 1
		stop_index = min(stop_index, n_alphas)
		stability_scores = np.max(stability_paths[:stop_index + 1, :], axis=0)
		for j in range(p):
			if stability_scores[j] >= tau and j not in already_selected:
				efp_scores[j] = efp
				already_selected.append(j)

	efp_scores = dict(zip(preselect_indices, efp_scores))

	# reinsert features removed during preselection
	if p_full != p:
		all_features = set(range(p_full))
		missing_features = all_features - efp_scores.keys()
		for feature in missing_features:
			efp_scores[feature] = p
		efp_scores = {feature: (p_full if score >= p - 1 else score) for feature, score in efp_scores.items()}

	q_values = compute_qvalues(efp_scores)

	runtime = time.time() - start

	return { 
		'efp_scores': efp_scores,
		'q_values': q_values,
		'runtime': runtime, 
		'selected_features': [], 
		'stability_paths': stability_paths
		}

#--------------------------------
# Helpers
#--------------------------------
# construct list of average number of features selected cutoffs
def compute_q_list(efp_list, tau, method, p, B, sampling_type='SS'):
	q_list = []
	for efp in efp_list:
		if efp == 0:
			q_list.append(0)
		else:
			q = stabsel_q(p, cutoff=tau, PFER=efp, B=B, assumption=method, sampling_type=sampling_type)
			q_list.append(q)
	return q_list

def stabsel_q(p, cutoff, PFER, B=50, assumption='unimodal', sampling_type='SS', verbose=False):
	p_r = IntVector([p])
	cutoff_r = FloatVector([cutoff])
	PFER_r = FloatVector([PFER])
	B_r = IntVector([B]) if B is not None else NULL
	assumption_r = StrVector([assumption])
	sampling_type_r = StrVector([sampling_type])
	verbose_r = BoolVector([verbose])
	
	# Call the stabsel_parameters function with PFER and cutoff, letting it find q
	result = stabs.stabsel_parameters(
		p=p_r, cutoff=cutoff_r, PFER=PFER_r, 
		B=B_r, assumption=assumption_r, 
		sampling_type=sampling_type_r, verbose=verbose_r
	)
	
	# Extract the value of q from the result
	q = int(result.rx2('q')[0])
	
	return q



