# Analyze results from cross-validation cancer studies

import os

import numpy as np
import pickle
import matplotlib.pyplot as plt

# save figure
save_figure = False

# random seed only for perturbing methods that select no features (DeepPINK and SSBoost)
np.random.seed(5)

# plot specifications
show_plot = True
plot_options = ['error_vs_fdr', 'error_vs_n']
fontsize = 24

method_list = ['all_features', 'vita', 'boruta', 'ssboost', 'deeppink', 'kol', 'korf', 'koglm', 'ipssl', 'ipssrf', 'ipssgb']
methods_to_remove = ['vita']
method_list = [method for method in method_list if method not in methods_to_remove]
cancer_type = 'glioma'
feature_type = 'rnaseq'
response = 'status'
figure_name = f'cv_study_{cancer_type}_{feature_type}_{response}'

fdr_min = 0
fdr_max = 0.5
lw = 3

#--------------------------------
# Method specifications
#--------------------------------
method_specs = {
	'all_features':{'name':'All features', 'color':'black', 'linestyle':'--', 'lw':lw},
	'boruta':{'name':'Boruta', 'color':'darkgray', 'linestyle':'-', 'lw':lw},
	'deeppink':{'name':'DeepPINK', 'color':'deeppink', 'linestyle':'--', 'lw':lw}, # Preselect 100 with RF preselection
	'ipssgb':{'name':'IPSSGB', 'color':'dodgerblue', 'linestyle':'-', 'lw':lw},
	'ipssl':{'name':'IPSSL1', 'color':'orange', 'linestyle':'-', 'lw':lw},
	'ipssrf':{'name':'IPSSRF', 'color':'limegreen', 'linestyle':'-', 'lw':lw},
	'koglm':{'name':'KOGLM', 'color':'deepskyblue', 'linestyle':'--', 'lw':lw}, # Preselect 200 with RF preselection
	'kol':{'name':'KOL1', 'color':'orange', 'linestyle':'--', 'lw':lw}, # Preselect 200 with L1 preselection
	'korf':{'name':'KORF', 'color':'limegreen', 'linestyle':'--', 'lw':lw}, # Preselect 200 with RF preselection
	'rfegb':{'name':'RFEGB', 'color':'darkgray', 'linestyle':'-.', 'lw':lw},
	'ssboost':{'name':'SSBoost', 'color':'gold', 'linestyle':'-', 'lw':lw},
	'vita':{'name':'Vita', 'color':'pink', 'linestyle':'--', 'lw':lw},
	'vsurf':{'name':'VSURF', 'color':'darkgray', 'linestyle':':', 'lw':lw}
}

#--------------------------------
# Analyze results
#--------------------------------
# Create subplots based on requested plots
plot_titles = {
	'n_vs_fdr': ('Target FDR', 'Number of selected features'),
	'error_vs_fdr': ('Target FDR', 'Prediction error'),
	'error_vs_n': ('Number of features selected', 'Prediction error')
}
plot_indices = {k: i for i, k in enumerate(plot_options)}
fig, ax = plt.subplots(1, len(plot_options), sharey='row', figsize=(18,6))
if len(plot_options) == 1:
	ax = [ax]  # make indexable

# Plotting loop
missing_methods = []
for method in method_list:

	filepath = f'./cv_results/{method}/{method}_{cancer_type}_{feature_type}_{response}.pkl'
	if not os.path.exists(filepath):
		missing_methods.append(f'{method} results do not exist.')
		continue
	with open(filepath, "rb") as f:
		results = pickle.load(f)

	fdr_list = results['metadata']['fdr_list']
	fdr_start = np.where(fdr_list <= fdr_min)[0][-1]
	fdr_stop = np.where(fdr_list >= fdr_max)[0][0]

	avg_runtime = np.mean(results['runtimes'])

	print(f'\n{method} ({avg_runtime:.1f} seconds)')
	print(f'--------------------------------')
	feature_names = results['metadata']['feature_names']
	Q = results['Q']
	median_q_value = np.mean(Q, axis=0)
	sorted_features = np.argsort(median_q_value)
	for feature in sorted_features:
		if median_q_value[feature] < fdr_max:
			print(f'{feature_names[feature]}: {median_q_value[feature]:.2f}')

	mean_n_selected = np.mean(results['n_selected'], axis=0)
	mean_gb_error = np.mean(results['gb_errors'], axis=0)
	mean_linear_error = np.mean(results['linear_errors'], axis=0)
	mean_rf_error = np.mean(results['rf_errors'], axis=0)
	best_error = np.minimum.reduce([mean_gb_error, mean_linear_error, mean_rf_error])

	if method in ['boruta', 'rfegb', 'vita', 'vsurf']:
		mean_n_selected[0] = mean_n_selected[1]
		best_error[0] = best_error[1]

	color = method_specs[method]['color']
	linestyle = method_specs[method]['linestyle']
	lw = method_specs[method]['lw']
	label = method_specs[method]['name']

	# Plot each requested type
	if 'n_vs_fdr' in plot_options:
		i = plot_indices['n_vs_fdr']
		ax[i].plot(fdr_list[fdr_start:fdr_stop], mean_n_selected[fdr_start:fdr_stop],
				   lw=lw, linestyle=linestyle, color=color, label=label)

	if 'error_vs_fdr' in plot_options:
		i = plot_indices['error_vs_fdr']
		if method in ['all_features', 'boruta', 'vita']:
			ax[i].axhline(best_error[0], lw=lw, linestyle=linestyle, color=color, label=label)
		else:
			ax[i].plot(fdr_list[fdr_start:fdr_stop], best_error[fdr_start:fdr_stop],
					   lw=lw, linestyle=linestyle, color=color, label=label)

	if 'error_vs_n' in plot_options:
		i = plot_indices['error_vs_n']
		if method == 'all_features':
			ax[i].axhline(best_error[0], lw=lw, linestyle=linestyle, color=color, label=label)
		elif method in ['boruta', 'vita'] or np.sum(mean_n_selected) < 5:
			ax[i].scatter(mean_n_selected[1] + np.random.uniform(0,2), best_error[1], s=400, color=color, label=label, edgecolor='black')
		else:
			ax[i].plot(mean_n_selected[fdr_start:fdr_stop], best_error[fdr_start:fdr_stop],
					   lw=lw, linestyle=linestyle, color=color, label=label)

for i, key in enumerate(plot_options):
	xlabel, ylabel = plot_titles[key]
	ax[i].set_xlabel(xlabel, fontsize=fontsize)
	if i == 0:
		ax[i].set_ylabel(ylabel, fontsize=fontsize)
	ax[i].tick_params(axis='both', labelsize=12)
	ax[i].grid(True)
	ax[i].set_axisbelow(True)
	if key == 'error_vs_n':
		label_order = list(reversed(method_list))
		handles = [plt.Line2D([], [], color=method_specs[m]['color'],
							  linestyle=method_specs[m]['linestyle'],
							  lw=4) for m in label_order]
		labels = [method_specs[m]['name'] for m in label_order]
		ax[i].legend(handles, labels, fontsize=18)

print()
print(f'Missing methods')
print(f'--------------------------------')
for string in missing_methods:
	print(string)

plt.tight_layout()
if save_figure:
	plt.savefig(f'{figure_name}.png', dpi=300)
if show_plot:
	plt.show()

print()