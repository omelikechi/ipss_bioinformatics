# Different feature selection methods

import os
import time
import warnings

from ipss import ipss
from ipss.preselection import preselection
from knockpy import KnockoffFilter
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold, KFold
import xgboost as xgb

# r related packages
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import IntVector, FloatVector, StrVector, BoolVector, NULL

numpy2ri.activate()

from SSBoost.ssboost import ssboost

#--------------------------------
# Boruta (R)
#--------------------------------
def run_boruta(X, y, fdr_list, **kwargs):
	classifier = True if len(np.unique(y)) <= 2 else False
	r = robjects.r
	# r.source("methods.r")
	script_dir = os.path.dirname(__file__)
	r_script_path = os.path.join(script_dir, "methods.r")
	r.source(r_script_path)
	with localconverter(robjects.default_converter + numpy2ri.converter):
		X_r = robjects.conversion.py2rpy(X)
		y_r = robjects.conversion.py2rpy(y)
	run_boruta_r = robjects.globalenv['run_boruta']
	result = run_boruta_r(X_r, y_r, classifier)
	runtime = float(result.rx2('runtime'))
	selected_features = list(result.rx2('selected_features'))
	selected_features = [i - 1 for i in selected_features]
	q_values = {feature: 1 for feature in np.arange(X.shape[1])}
	for feature in selected_features:
		q_values[feature] = 0
	return {'q_values':q_values, 'runtime':runtime}

#--------------------------------
# DeepPINK
#--------------------------------
def run_deeppink(X, y, fdr_list, **kwargs):
	y = y.ravel()
	start = time.time()
	X, preselect_indices = preselect_features(X, y, **kwargs)
	kfilter = KnockoffFilter(ksampler='gaussian', fstat='deeppink')
	kfilter.forward(X=X, y=y)
	W = kfilter.W
	runtime = time.time() - start 
	q_values = {}
	for target_fdr in fdr_list:
		selected_features = np.where(kfilter.make_selections(W, fdr=target_fdr))[0]
		for j in selected_features:
			if j not in q_values.keys():
				q_values[preselect_indices[j]] = target_fdr
	return {'q_values':q_values, 'runtime':runtime}

#--------------------------------
# IPSSGB
#--------------------------------
def run_ipssgb(X, y, fdr_list, **kwargs):
	result = ipss(X, y, selector='gb', **kwargs)
	runtime = result['runtime']
	q_values = result['q_values']
	return {'q_values':q_values, 'runtime':runtime}

#--------------------------------
# IPSSL
#--------------------------------
def run_ipssl(X, y, fdr_list, **kwargs):
	result = ipss(X, y, selector='l1', **kwargs)
	runtime = result['runtime']
	q_values = result['q_values']
	return {'q_values':q_values, 'runtime':runtime}

#--------------------------------
# IPSSRF
#--------------------------------
def run_ipssrf(X, y, fdr_list, **kwargs):
	result = ipss(X, y, selector='rf', **kwargs)
	runtime = result['runtime']
	q_values = result['q_values']
	return {'q_values':q_values, 'runtime':runtime}

#--------------------------------
# KOBT
#--------------------------------
def run_kobt(X, y, fdr_list, **kwargs):

	p_true = len(true_features)

	start = time.time()
	preselect = True
	if preselect:
		unique_values = np.unique(y)
		binary_response = len(unique_values) == 2
		# selector = 'rf_classifier' if binary_response else 'rf_regressor'
		selector = 'gb_classifier' if binary_response else 'gb_regressor'
		X, preselect_indices = preselection(X, y, selector, preselector_args={'n_keep':100})
	else:
		preselect_indices = np.arange(X.shape[1])

	# Load the R script containing run_knockoffs()
	r = robjects.r
	# r.source("methods.r")
	script_dir = os.path.dirname(__file__)
	r_script_path = os.path.join(script_dir, "methods.r")
	r.source(r_script_path)

	# Retrieve the function from R environment
	run_kobt_r = robjects.globalenv['run_kobt']

	# Convert Python objects to R format
	with localconverter(robjects.default_converter + numpy2ri.converter):
		X_r = robjects.conversion.py2rpy(X)
		y_r = robjects.conversion.py2rpy(y)
		fdr_list_r = robjects.FloatVector(fdr_list)

	# Call the R function
	result = run_kobt_r(X_r, y_r, fdr_list_r, num_knockoffs = 2)

	runtime = time.time() - start

	# Convert results back to Python
	q_values = np.array(result.rx2('scores'))
	tprs, fdrs = [], []
	for target_fdr in fdr_list:
		tp, fp = 0, 0
		for j in range(len(q_values)):
			if q_values[j] <= target_fdr:
				if preselect_indices[j] in true_features:
					tp += 1
				else:
					fp += 1
		fdr = 0 if tp + fp == 0 else fp / (fp + tp)
		tpr = tp / p_true
		tprs.append(tpr)
		fdrs.append(fdr)
	return {'tprs': tprs, 'fdrs': fdrs, 'runtime': runtime}

#--------------------------------
# KOGLM
#--------------------------------
def run_koglm(X, y, fdr_list, **kwargs):
	y = y.ravel()
	start = time.time()
	X, preselect_indices = preselect_features(X, y, **kwargs)
	# Load the R script containing run_knockoffs()
	r = robjects.r
	# r.source("methods.r")
	script_dir = os.path.dirname(__file__)
	r_script_path = os.path.join(script_dir, "methods.r")
	r.source(r_script_path)
	# Retrieve the function from R environment
	run_knockoffs_r = robjects.globalenv['run_knockoffs']
	# Convert Python objects to R format
	with localconverter(robjects.default_converter + numpy2ri.converter):
		X_r = robjects.conversion.py2rpy(X)
		y_r = robjects.conversion.py2rpy(y)
		fdr_list_r = robjects.FloatVector(fdr_list)
	# Call the R function
	result = run_knockoffs_r(X_r, y_r, fdr_list_r, stat="stat.glmnet_coefdiff")
	runtime = time.time() - start
	# Convert results back to Python
	q_values = np.array(result.rx2('scores'))
	q_values = dict(zip(preselect_indices,q_values))
	return {'q_values':q_values, 'runtime': runtime}

#--------------------------------
# KOL
#--------------------------------
def run_kol(X, y, fdr_list, **kwargs):
	y = y.ravel()
	start = time.time()
	unique_values = np.unique(y)
	binary_response = len(unique_values) == 2
	stat = 'stat.lasso_coefdiff_bin' if binary_response else 'stat.lasso_coefdiff'
	X, preselect_indices = preselect_features(X, y, **kwargs)
	# Load the R script containing run_knockoffs()
	r = robjects.r
	# r.source("methods.r")
	script_dir = os.path.dirname(__file__)
	r_script_path = os.path.join(script_dir, "methods.r")
	r.source(r_script_path)
	# Retrieve the function from R environment
	run_knockoffs_r = robjects.globalenv['run_knockoffs']
	# Convert Python objects to R format
	with localconverter(robjects.default_converter + numpy2ri.converter):
		X_r = robjects.conversion.py2rpy(X)
		y_r = robjects.conversion.py2rpy(y)
		fdr_list_r = robjects.FloatVector(fdr_list)
	# Call the R function
	result = run_knockoffs_r(X_r, y_r, fdr_list_r, stat="stat.lasso_coefdiff")
	runtime = time.time() - start
	# Convert results back to Python
	q_values = np.array(result.rx2('scores'))
	q_values = dict(zip(preselect_indices,q_values))
	return {'q_values':q_values, 'runtime': runtime}

#--------------------------------
# KORF
#--------------------------------
def run_korf(X, y, fdr_list, **kwargs):
	y = y.ravel()
	start = time.time()
	X, preselect_indices = preselect_features(X, y, **kwargs)
	# Load the R script containing run_knockoffs()
	r = robjects.r
	# r.source("methods.r")
	script_dir = os.path.dirname(__file__)
	r_script_path = os.path.join(script_dir, "methods.r")
	r.source(r_script_path)
	# Retrieve the function from R environment
	run_knockoffs_r = robjects.globalenv['run_knockoffs']
	# Convert Python objects to R format
	with localconverter(robjects.default_converter + numpy2ri.converter):
		X_r = robjects.conversion.py2rpy(X)
		y_r = robjects.conversion.py2rpy(y)
		fdr_list_r = robjects.FloatVector(fdr_list)
	# Call the R function
	result = run_knockoffs_r(X_r, y_r, fdr_list_r, stat="stat.random_forest")
	runtime = time.time() - start
	# Convert results back to Python
	q_values = np.array(result.rx2('scores'))
	q_values = dict(zip(preselect_indices,q_values))
	return {'q_values':q_values, 'runtime': runtime}

#--------------------------------
# RFEGB
#--------------------------------
def run_rfegb(X, y, fdr_list, **kwargs):
	start = time.time()
	# XGBoost args
	kwargs = {'max_depth':1, 'colsample_bynode':1/3, 'n_estimators':100, 'importance_type':'gain'}
	unique_values = np.unique(y)
	binary_response = len(unique_values) == 2
	if binary_response:
		kwargs['eval_metric'] = 'logloss'
		model = xgb.XGBClassifier(**kwargs)
		cv = StratifiedKFold(n_splits=5, shuffle=True)
	else:
		kwargs['eval_metric'] = 'rmse'
		model = xgb.XGBRegressor(**kwargs)
		cv = KFold(n_splits=5, shuffle=True)
	preselect = False
	if preselect:
		selector = 'gb_classifier' if binary_response else 'gb_regressor'
		X, preselect_indices = preselection(X, y, selector, preselector_args={'n_keep':200})
	else:
		preselect_indices = np.arange(X.shape[1])
	selector = RFECV(estimator=model, step=1, cv=cv)
	selector.fit(X, y)
	selected_features = np.where(selector.support_)[0]
	runtime = time.time() - start
	print(runtime)
	q_values = {feature: 1 for feature in np.arange(X.shape[1])}
	for feature in selected_features:
		q_values[feature] = 0
	return {'q_values':q_values, 'runtime':runtime}

#--------------------------------
# SSBoost
#--------------------------------
def run_ssboost(X, y, fdr_list, **kwargs):
	preselector_args = {'n_keep':kwargs['n_keep']}
	kwargs.pop('n_keep', None)
	result = ssboost(X, y, **kwargs)
	runtime = result['runtime']
	q_values = result['q_values']
	return {'q_values':q_values, 'runtime': runtime}

#--------------------------------
# Vita (R)
#--------------------------------
def run_vita(X, y, fdr_list, **kwargs):
	classifier = True if len(np.unique(y)) <= 2 else False
	r = robjects.r
	# r.source("methods.r")
	script_dir = os.path.dirname(__file__)
	r_script_path = os.path.join(script_dir, "methods.r")
	r.source(r_script_path)
	with localconverter(robjects.default_converter + numpy2ri.converter):
		X_r = robjects.conversion.py2rpy(X)
		y_r = robjects.conversion.py2rpy(y)
	run_vita = robjects.globalenv['run_vita']
	result = run_vita(X_r, y_r, classifier)
	runtime = float(result.rx2('runtime'))
	selected_features = list(result.rx2('selected_features'))
	selected_features = [i - 1 for i in selected_features]
	q_values = {feature: 1 for feature in np.arange(X.shape[1])}
	for feature in selected_features:
		q_values[feature] = 0
	return {'q_values':q_values, 'runtime':runtime}

#--------------------------------
# VSURF (R)
#--------------------------------
def run_vsurf(X, y, fdr_list, **kwargs):
	vsurf_pkg = importr('VSURF')
	classifier = True if len(np.unique(y)) <= 2 else False
	with localconverter(robjects.default_converter + numpy2ri.converter):
		X_r = robjects.conversion.py2rpy(X)
		y_r = robjects.conversion.py2rpy(y)
	if classifier:
		y_r = robjects.r('as.factor')(y_r)
	start = time.time()
	vsurf_fit = vsurf_pkg.VSURF(X_r, y_r, parallel=True, verbose=False)
	selected_features_r = vsurf_fit.rx2('varselect.pred')
	runtime = time.time() - start
	selected_features = list(robjects.conversion.rpy2py(selected_features_r))
	selected_features = [i - 1 for i in selected_features]
	q_values = {feature: 1 for feature in np.arange(X.shape[1])}
	for feature in selected_features:
		q_values[feature] = 0
	return {'q_values':q_values, 'runtime':runtime}

#--------------------------------
# Main method function
#--------------------------------
method_list = {
	'boruta':run_boruta,
	'deeppink':run_deeppink,
	'ipssgb':run_ipssgb,
	'ipssl':run_ipssl,
	'ipssrf':run_ipssrf,
	'kobt':run_kobt,
	'koglm':run_koglm,
	'kol':run_kol,
	'korf':run_korf,
	'rfegb':run_rfegb,
	'ssboost':run_ssboost,
	'vita':run_vita,
	'vsurf':run_vsurf
}

def run_method(X, y, fdr_list, method_name, kwargs={}):
	return method_list[method_name](X, y, fdr_list, **kwargs)

#--------------------------------
# Helpers
#--------------------------------
def preselect_features(X, y, **kwargs):
	preselect = kwargs.pop('preselect', False)
	preselector = kwargs.pop('preselector', 'rf')
	n_keep = kwargs.pop('n_keep', 200)
	expansion_factor = kwargs.pop('expansion_factor', 1.5)
	if preselect:
		unique_values = np.unique(y)
		binary_response = len(unique_values) == 2
		if preselector == 'gb':
			selector = 'gb_classifier' if binary_response else 'gb_regressor'
			preselector_args = {'expansion_factor':expansion_factor}
		elif preselector == 'l1':
			selector = 'logistic_regression' if binary_response else 'lasso'
			preselector_args = {'n_keep':n_keep}
		elif preselector == 'rf':
			selector = 'rf_classifier' if binary_response else 'rf_regressor'
			preselector_args = {'n_keep':n_keep}
		X, preselect_indices = preselection(X, y, selector, preselector_args=preselector_args)
	else:
		preselect_indices = np.arange(X.shape[1])
	return X, preselect_indices

