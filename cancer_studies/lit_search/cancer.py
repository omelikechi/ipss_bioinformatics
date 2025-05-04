# Analyze human cancer data from The Cancer Genome Atlas via LinkedOmics
# source: https://www.linkedomics.org/login.php

import os
import sys

from ipss import ipss
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

sys.path.append('..')
from load_cancer_data import load_data
from methods import run_method
from SSBoost.ssboost import compute_q_list

#--------------------------------
# Setup
#--------------------------------
################################################################
save_results = False
################################################################
cancer_type = 'glioma'
feature_type = ['rnaseq']
response = [('clinical', 'status')]
expression_threshold = None if cancer_type == 'ovarian' else 50
variance_threshold = None if cancer_type == 'ovarian' else None
################################################################
experiment_name = f'{cancer_type}_{feature_type[0]}_{response[0][1]}'
################################################################
fdr_list = np.arange(0, 0.51, 0.01)
################################################################
method_list = ['ipssgb', 'ipssrf']
################################################################

# random seed
seed = 302
np.random.seed(seed)

#--------------------------------
# Load data
#--------------------------------
data = load_data(cancer_type, feature_types=feature_type, responses=response, expression_threshold=expression_threshold, 
	variance_threshold=variance_threshold, verbose=True)
X, y, feature_names = data['X'], data['Y'], data['feature_names']
y = y.ravel()
X = StandardScaler().fit_transform(X)

n, p = X.shape
print(f'X.shape = {X.shape}')

# metadata
metadata = {
	'cancer_type':cancer_type, 'feature_type':feature_type[0], 'feature_names':feature_names, 'response':response[0], 
	'n':X.shape[0], 'p':X.shape[1], 'fdr_list':fdr_list, 'random_seed':seed
}

#--------------------------------
# Method args
#--------------------------------
method_args = {}
# preselect more features for large RNA-seq datasets
if response[0][0] == 'clinical':
	if feature_type[0] == 'rnaseq':
		expansion_factor = 2.5
	else:
		expansion_factor = 2
else:
	expansion_factor = 1.5
print(expansion_factor)

method_args['boruta'] = {}
method_args['deeppink'] = {'preselect':True, 'preselector':'rf', 'n_keep':100}
method_args['ipssgb'] = {'preselector_args':{'expansion_factor':expansion_factor}}
method_args['ipssl'] = {}
method_args['ipssrf'] = {}
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
# Run methods
#--------------------------------
results = {method: {} for method in method_list}
for method in method_list:
	print(f'\n-----{method}-----')
	kwargs = method_args[method]
	result = run_method(X, y, fdr_list, method, kwargs=kwargs)
	runtime = result['runtime']
	print(f'runtime = {runtime:.2f}')
	scores = result['q_values']
	scores = dict(sorted(scores.items(), key=lambda item: item[1]))

	results[method]['q_values'] = scores
	results[method]['method_args'] = method_args[method]
	results[method]['runtime'] = runtime
	results[method]['metadata'] = metadata

	if save_results:
		file_name = f'./cancer_results/{method}/{method}_{experiment_name}.pkl'
		with open(file_name, 'wb') as f:
			pickle.dump(results[method], f)
	for feature, score in scores.items():
		if score < 1:
			print(f'{feature_names[feature]}: {score:.2f}')












