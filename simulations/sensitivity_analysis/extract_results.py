# Extract and save results from sensitvity simulation

import pickle

experiment = 'toeplitz_0.5_n500'
response_types = ['reg', 'class']
p_grid = [500, 2000, 5000]
method = 'ipssgb'
parameter = 'delta'

for response_type in response_types:
	for p in p_grid:
		# load results
		# with open(f'./results/{method}/{experiment}_{response_type}_{p}_{method}_{parameter}.pkl', "rb") as f:
		with open(f'{experiment}_{response_type}_{p}_{method}_{parameter}.pkl', "rb") as f:
			results = pickle.load(f)

		simulation_config = results['simulation_config']
		fdr_list = simulation_config['fdr_list']
		tpr_matrix = results['tpr_matrix']
		fdr_matrix = results['fdr_matrix']
		runtimes = results['runtimes']

		function = 'h3'
		cutoff = 0.05
		delta = 5/4 if response_type == 'reg' else 1
		specific_results = {}
		specific_results['simulation_config'] = simulation_config
		specific_results['fdr_list'] = fdr_list
		specific_results['runtimes'] = runtimes
		specific_results['tpr_matrix'] = tpr_matrix[function][cutoff][delta]
		specific_results['fdr_matrix'] = fdr_matrix[function][cutoff][delta]
		directory = '/Users/omm793/iCloud/code/research/ipss/bioinformatics/simulations'
		file_name = directory + f'/{experiment}_{response_type}_{p}_{method}.pkl'
		with open(file_name, 'wb') as f:
			pickle.dump(specific_results, f)