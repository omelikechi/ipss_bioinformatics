# Helper functions for the cross-validation experiments

import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
from xgboost import XGBClassifier, XGBRegressor

def compute_gb_errors(X_train, X_test, y_train, y_test, selected_features, binary_response):
	
	if len(selected_features) == 0:
		return null_model_error(y_train, y_test, binary_response)

	X_train_select = X_train[:, selected_features]
	X_test_select = X_test[:, selected_features]

	scaler = StandardScaler()
	X_train_select = scaler.fit_transform(X_train_select)
	X_test_select = scaler.transform(X_test_select)

	xgb_params = {'n_estimators': 100, 'max_depth': 3}
	xgb_params['learning_rate'] = 0.3 if X_train_select.shape[1] > 100 else 0.1
	model = XGBClassifier(**xgb_params) if binary_response else XGBRegressor(**xgb_params)

	eval_set = [(X_train_select, y_train), (X_test_select, y_test)]
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		model.fit(X_train_select, y_train, eval_set=eval_set, verbose=False)

	y_pred = model.predict(X_test_select)
	return 1 - accuracy_score(y_test, y_pred) if binary_response else mean_squared_error(y_test, y_pred)

def compute_linear_errors(X_train, X_test, y_train, y_test, selected_features, binary_response):
	
	if len(selected_features) == 0:
		return null_model_error(y_train, y_test, binary_response)

	X_train_select = X_train[:, selected_features]
	X_test_select = X_test[:, selected_features]
	n_pred, p_pred = X_train_select.shape

	scaler = StandardScaler()
	X_train_select = scaler.fit_transform(X_train_select)
	X_test_select = scaler.transform(X_test_select)

	if binary_response:
		model = LogisticRegression(max_iter=1000)
	elif p_pred > n_pred:
		model = LassoCV()
	else:
		model = LinearRegression()

	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		model.fit(X_train_select, y_train)

	y_pred = model.predict(X_test_select)
	return 1 - accuracy_score(y_test, y_pred) if binary_response else mean_squared_error(y_test, y_pred)

def compute_rf_errors(X_train, X_test, y_train, y_test, selected_features, binary_response):
	
	if len(selected_features) == 0:
		return null_model_error(y_train, y_test, binary_response)

	X_train_select = X_train[:, selected_features]
	X_test_select = X_test[:, selected_features]

	scaler = StandardScaler()
	X_train_select = scaler.fit_transform(X_train_select)
	X_test_select = scaler.transform(X_test_select)

	model = RandomForestClassifier(max_features=1/3) if binary_response else RandomForestRegressor(max_features=1/3)
	model.fit(X_train_select, y_train)

	y_pred = model.predict(X_test_select)
	return 1 - accuracy_score(y_test, y_pred) if binary_response else mean_squared_error(y_test, y_pred)

def null_model_error(y_train, y_test, binary_response):
	if binary_response:
		majority_class = np.round(np.mean(y_train)).astype(int)
		y_pred = np.full_like(y_test, majority_class)
		return 1 - accuracy_score(y_test, y_pred)
	else:
		y_pred = np.full_like(y_test, np.mean(y_train))
		return mean_squared_error(y_test, y_pred)



