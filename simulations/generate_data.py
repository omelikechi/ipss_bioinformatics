# Generate data for 

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import toeplitz, cholesky
from sklearn.preprocessing import StandardScaler

#--------------------------------
# Main data generator
#--------------------------------
def generate_data(n, p, p_true, snr, response_type, feature_matrix, standardize=True, center_response=True, 
	group_features=True, random_seed=None):

	np.random.seed(random_seed)

	# randomly select parameter from list
	if isinstance(p, list):
		p = np.random.randint(p[0], p[1] + 1)
	if isinstance(n, list):
		n = np.random.randint(n[0], n[1] + 1)
	if isinstance(p_true, list):
		p_true = np.random.randint(p_true[0], p_true[1] + 1)
	if isinstance(snr, list):
		snr = np.random.uniform(snr[0], snr[1])

	X = generate_features(n, p, feature_matrix, standardize)

	true_features = np.random.choice(np.arange(p), size=p_true, replace=False)

	y = generate_response(X, true_features, snr, response_type, center_response, group_features)

	return X, y, true_features

#--------------------------------
# Helpers
#--------------------------------
def generate_features(n, p, feature_matrix, standardize):

	if isinstance(feature_matrix, np.ndarray):
		n_full, p_full = feature_matrix.shape
		if n < n_full:
			rows = np.random.choice(n_full, size=n, replace=False)
			X = feature_matrix[rows, :]
		if p < p_full:
			cols = np.random.choice(p_full, size=p, replace=False)
			X = X[:, cols]
	elif feature_matrix == 0:
		X = np.random.normal(0, 1, size=(n, p))
	elif isinstance(feature_matrix, (int, float)) and 0 <= feature_matrix <= 1:
		X = generate_toeplitz_samples(n, p, feature_matrix)
	else:
		raise ValueError("Invalid feature_matrix parameter. Must be a valid NumPy array or a number between 0 and 1.")

	# standardize features
	if standardize:
		X = StandardScaler().fit_transform(X)

	return X

def generate_response(X, true_features, snr, response_type, center_response, group_features):
	n, p = X.shape
	p_true = len(true_features)

	# generate signal
	signal = np.zeros(n)
	if group_features:
		assignments = np.random.randint(0, p_true, size=p_true)
		groups = [[] for _ in range(p_true)]
		for i, group_id in enumerate(assignments):
			groups[group_id].append(true_features[i])
		for group in groups:
			if len(group) > 0:
				group_sum = X[:, group].sum(axis=1)
				group_sum = (group_sum - np.mean(group_sum)) / np.std(group_sum)
				signal += generate_bump_function(group_sum)
	else:
		for feature in true_features:
			signal += generate_bump_function(X[:,feature])

	# add noise
	if response_type == 'reg':
		sigma2 = np.var(signal) / snr
		y = signal + np.random.normal(0, np.sqrt(sigma2), size=n)
		y = y - np.mean(y) if center_response else y
	elif response_type == 'class':
		signal = signal - np.mean(signal)
		prob = 1 / (1 + np.exp(-snr * signal))
		y = np.random.binomial(1, prob, size=n)
	else:
		raise ValueError('Response type must be "reg" or "class"')

	return y

def generate_bump_function(x):
	alpha = np.random.uniform(1/2, 3/2)
	beta = np.random.uniform(-1, 1)
	gamma = 1 #np.random.uniform(1, 3)
	delta1 = 1 #np.random.choice([-1, 1])
	delta2 = np.random.choice([-1, 1])
	if np.random.uniform(0, 1) <= -0.5:
		y = 0.5 * delta1 * (1 + np.tanh(alpha * (delta2 * x - beta)))
	else:
		y = delta1 * np.exp(-gamma * x**2)
	# y /= np.std(y) #(y - np.mean(y)) / np.std(y)
	return y

def generate_fourier_function(X, n_terms=100, max_frequency=2):
	a_coeffs = np.random.normal(0, 1, n_terms)
	b_coeffs = np.random.normal(0, 1, n_terms)
	frequencies = np.random.randint(1, max_frequency + 1, n_terms)
	y = np.zeros_like(X)
	for a, b, freq in zip(a_coeffs, b_coeffs, frequencies):
		y += a * np.cos(freq * X) + b * np.sin(freq * X)
	y = (y - np.mean(y)) / np.std(y)
	return y

def generate_toeplitz_samples(n, p, rho):
	first_col = rho ** np.arange(p)
	Sigma = toeplitz(first_col)
	L = cholesky(Sigma, lower=True)
	Z = np.random.randn(n, p)
	return Z @ L.T



