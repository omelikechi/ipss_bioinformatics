# Functions for loading LinkedOmics data

import os

import numpy as np
import pandas as pd

def load_data(
	cancer_type, 
	feature_types, 
	responses=None,
	remove_nan=True, 
	correlation_threshold=0.999, 
	expression_threshold=None,
	variance_threshold=None,
	print_feature_names=False, 
	verbose=False
):

	if verbose:
		print(f'Loading data')
		print(f'----------------')

	base_path = os.path.join(os.path.dirname(__file__), f'./data/{cancer_type}')

	file_paths = {
		'clinical': os.path.join(base_path, 'clinical.txt'),
		'methylation': os.path.join(base_path, 'methylation.txt'),
		'mirna': os.path.join(base_path, 'mirna.txt'),
		'rnaseq': os.path.join(base_path, 'rnaseq.txt'),
		'rppa': os.path.join(base_path, 'rppa.txt')
	}

	# Initialize shared labels
	shared_labels = None

	# Process multiple responses
	Y_df_list = []  # Store response dataframes
	if responses is not None:
		for response_type, response_name in responses:
			response_df = load_dataframe(file_paths[response_type])
			if response_name == 'gender':
				response_df[response_name] = response_df[response_name].replace({'male': 0, 'female': 1})
			if response_name == 'histological_type':
				if cancer_type == 'breast':
					histological_types = {'infiltratingductalcarcinoma':0, 'infiltratinglobularcarcinoma':1}
					response_df[response_name] = response_df[response_name].map(histological_types)
			if response_name == 'pathologic_stage':
				response_df[response_name] = response_df[response_name].replace({'stagei':1, 'stageii':2, 'stageiii':3, 'stageiv':4})
			if response_name == 'pathology_M_stage':
				response_df[response_name] = response_df[response_name].replace({'m0':0, 'm1':1})
			if response_name == 'pathology_N_stage':
				response_df[response_name] = response_df[response_name].replace({'n0':0, 'n1':1, 'n2':2, 'n3':3})
			if response_name == 'pathology_T_stage':
				response_df[response_name] = response_df[response_name].replace({'t1':1, 't2':2, 't3':3, 't4':4})
			if response_name == 'radiation_therapy':
				response_df[response_name] = response_df[response_name].replace({'no': 0, 'yes': 1})
			Y = response_df[response_name].apply(pd.to_numeric, errors='coerce')
			Y_df_list.append(Y)
			if shared_labels is None:
				shared_labels = set(Y.index)
			else:
				shared_labels = shared_labels.intersection(Y.index)

	# Features
	if print_feature_names:
		for category in feature_types:
			df = load_dataframe(file_paths[category])
			print(f'{category} features: {df.columns.tolist()}')
	else:
		feature_dfs = []
		feature_names = []
		for category in feature_types:

			df = load_dataframe(file_paths[category])			

			if verbose:
				print(f' - Original {category} dimensions: {df.shape}')
			
			# Remove response variables from feature dataframes
			if responses is not None:
				for response_type, response_name in responses:
					if category == response_type:
						df = df.drop(columns=[response_name], errors='ignore')

			if shared_labels is not None:
				shared_labels = shared_labels.intersection(df.index)
			else:
				shared_labels = set(df.index)

			if category == 'rnaseq' and expression_threshold is not None:
				mean_expression = df.mean(axis=0)
				expression_threshold = np.percentile(mean_expression, expression_threshold)
				expression_mask = mean_expression > expression_threshold
				df = df.loc[:, expression_mask]
				if verbose:
					print(f' - {category} after expression threshold: {df.shape}')
					print(f'   - smallest remaining mean expression = {min(df.mean(axis=0)):.2f}')

			if category == 'rnaseq' and variance_threshold is not None:
				variances = df.var(axis=0)
				variance_threshold = np.percentile(variances, variance_threshold)
				variance_mask = variances > variance_threshold
				df = df.loc[:, variance_mask]
				if verbose:
					print(f' - {category} after variance threshold: {df.shape}')
					print(f'   - smallest remaining variance = {min(df.var(axis=0)):.4f}')

			feature_dfs.append(df)

			if category == 'mirna':
				feature_names.extend(
					mirna.replace('hsa-mir-', 'miR-').replace('hsa-let-', 'let-') 
					for mirna in df.columns.tolist()
				)
			elif category == 'rppa':
				new_rppa_features = [
					f"{name.split('|')[0]}\n({name.split('|')[1]})" if '|' in name else name 
					for name in df.columns.tolist()
				]
				if 'rnaseq' in feature_types:
					new_rppa_features = [f'{name}#' for name in new_rppa_features]
				feature_names.extend(new_rppa_features)
			else:
				feature_names.extend(df.columns.tolist())

		# Ensure there are shared labels
		if not shared_labels:
			raise ValueError("No shared labels found across the feature datasets.")

		# Keep only shared labels in all dataframes
		shared_labels = sorted(shared_labels)
		feature_dfs = [df.loc[shared_labels] for df in feature_dfs]

		# Concatenate features into X
		X = pd.concat(feature_dfs, axis=1).to_numpy().astype(float)

		# Remove columns with NaN values
		if remove_nan:
			cols_with_nan = np.isnan(X).any(axis=0)
			X = X[:, ~cols_with_nan]
			feature_names = [name for (name, keep) in zip(feature_names, ~cols_with_nan) if keep]

		if verbose:
			print(f' - After combining types and removing NaN: {X.shape}')

		# Remove highly correlated columns
		if X.shape[1] > 1:
			correlation_matrix = np.corrcoef(X, rowvar=False)
			correlated_columns = set()
			for i in range(len(correlation_matrix)):
				for j in range(i + 1, len(correlation_matrix)):
					if abs(correlation_matrix[i, j]) > correlation_threshold:
						correlated_columns.add(j)

			uncorrelated_columns = [i for i in range(X.shape[1]) if i not in correlated_columns]
			X = X[:, uncorrelated_columns]
			feature_names = [feature_names[i] for i in uncorrelated_columns]

			if verbose:
				print(f' - After removing columns with correlation > {correlation_threshold}: {X.shape}')

		# Process Y (response matrix) if there are responses
		if responses is not None:
			Y = pd.concat(Y_df_list, axis=1).loc[shared_labels].to_numpy()
			rows_with_nan = np.isnan(Y).any(axis=1)
			X = X[~rows_with_nan, :]
			Y = Y[~rows_with_nan, :]
			if verbose:
				print(f' - Final dimensions (after also removing target NaNs): {X.shape}\n')
			return {'X': X, 'Y': Y, 'feature_names': feature_names, 'responses': responses}
		else:
			if verbose:
				print(f' - Final dimensions: {X.shape}\n')
			return {'X': X, 'Y': None, 'feature_names': feature_names, 'responses': responses}

# load dataframes
def load_dataframe(file_path):
	df = pd.read_csv(file_path, sep='\t').T
	df.columns = df.iloc[0]
	df = df.drop(df.index[0])
	df.index.name = 'attrib_name'
	return df



