# Analyze Gaussian simulation results

import os

from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
import pickle

################################################################
save_figure = False
################################################################
show_figure = True
################################################################
experiment = 'toeplitz_0.5_n250'
################################################################

#--------------------------------
# Simulations to plot
#--------------------------------
response_types = ['reg', 'class']
response_names = {'reg':'Regression', 'class':'Classification'}
methods_to_plot = ['vsurf', 'vita', 'boruta', 'ssboost', 'deeppink', 'kol', 'korf', 'koglm', 'ipssl', 'ipssrf', 'ipssgb']
fdr_max = 0.5
p = 500

#--------------------------------
# Method specifications
#--------------------------------
lw_nonparam_methods_with_fdr = 3
lw_param_methods_with_fdr = 3
lw_methods_without_fdr = 3
method_specs = {
	'boruta':{'name':'Boruta', 'color':'darkgray', 'linestyle':'-', 'lw':lw_methods_without_fdr},
	'deeppink':{'name':'DeepPINK', 'color':'deeppink', 'linestyle':'--', 'lw':lw_nonparam_methods_with_fdr},
	'ipssgb':{'name':'IPSSGB', 'color':'dodgerblue', 'linestyle':'-', 'lw':lw_nonparam_methods_with_fdr},
	'ipssl':{'name':'IPSSL1', 'color':'orange', 'linestyle':'-', 'lw':lw_param_methods_with_fdr},
	'ipssrf':{'name':'IPSSRF', 'color':'limegreen', 'linestyle':'-', 'lw':lw_nonparam_methods_with_fdr},
	'koglm':{'name':'KOGLM', 'color':'deepskyblue', 'linestyle':'--', 'lw':lw_param_methods_with_fdr},
	'kol':{'name':'KOL1', 'color':'orange', 'linestyle':'--', 'lw':lw_param_methods_with_fdr},
	'korf':{'name':'KORF', 'color':'limegreen', 'linestyle':'--', 'lw':lw_nonparam_methods_with_fdr}, # Preselect 200 with RF preselection
	'rfegb':{'name':'RFEGB', 'color':'darkgray', 'linestyle':'-.', 'lw':lw_methods_without_fdr},
	'ssboost':{'name':'SSBoost', 'color':'gold', 'linestyle':'-', 'lw':lw_nonparam_methods_with_fdr},
	'ssboost_pre100':{'name':'SSBoost', 'color':'gold', 'linestyle':'-', 'lw':lw_nonparam_methods_with_fdr},
	'vita':{'name':'Vita', 'color':'darkgray', 'linestyle':'--', 'lw':lw_methods_without_fdr},
	'vsurf':{'name':'VSURF', 'color':'darkgray', 'linestyle':':', 'lw':lw_methods_without_fdr}
}

#--------------------------------
# Plot
#--------------------------------
# plot args
fontsize = 32
tpr_stdevs = 0
fdr_stdevs = 0
missing_methods = []

fig = plt.figure(figsize=(18, 9)) 
gs = GridSpec(2, len(response_types) + 1, width_ratios=[1]*len(response_types) + [0.1])
ax = np.array([[fig.add_subplot(gs[r, c]) for c in range(len(response_types))] for r in range(2)])

for i, response_type in enumerate(response_types):

	ax[0,i].set_title(f'{response_names[response_type]}', fontsize=fontsize, fontweight='bold')

	print(f'\n{response_names[response_type]}')
	print(f'--------------------------------')
	for method in methods_to_plot:

		filepath = f'./results/{method}/{experiment}_{response_type}_{p}_{method}.pkl'
		if not os.path.exists(filepath):
			missing_methods.append(f'{method} for {experiment} ({response_type}) does not exist.')
			continue
		with open(filepath, "rb") as f:
			results = pickle.load(f)

		color = method_specs[method]['color']
		linestyle = method_specs[method]['linestyle']
		lw = method_specs[method]['lw']
		name = method_specs[method]['name']

		simulation_config = results['simulation_config']
		fdr_list = simulation_config['fdr_list']
		tpr_matrix = results['tpr_matrix']
		fdr_matrix = results['fdr_matrix']
		tpr_means = np.mean(tpr_matrix, axis=0)
		fdr_means = np.mean(fdr_matrix, axis=0)

		mask = fdr_list <= fdr_max
		stop_idx = np.argmax(fdr_list * mask) + 1

		ax[0,i].plot(fdr_list[:stop_idx], tpr_means[:stop_idx], color=color, linestyle=linestyle, lw=lw, label=name)
		ax[1,i].plot(fdr_list[:stop_idx], fdr_means[:stop_idx], color=color, linestyle=linestyle, lw=lw, label=name)

		if fdr_stdevs > 0:
			tpr_stds = tpr_stdevs * np.std(tpr_matrix, axis=0)
			ax[0,i].fill_between(
				fdr_list[:stop_idx],
				(tpr_means - tpr_stds)[:stop_idx],
				(tpr_means + tpr_stds)[:stop_idx],
				color=color,
				alpha=0.2
			)

		if fdr_stdevs > 0:
			fdr_stds = fdr_stdevs * np.std(fdr_matrix, axis=0)
			ax[1,i].fill_between(
				fdr_list[:stop_idx],
				(fdr_means - fdr_stds)[:stop_idx],
				(fdr_means + fdr_stds)[:stop_idx],
				color=color,
				alpha=0.2
			)

		runtimes = results['runtimes']
		runtime_mean = np.mean(runtimes)
		runtime_std = np.std(runtimes)
		print(f'{method} runtime = {runtime_mean:.1f} ({runtime_std:.2f})')

	ax[1,i].plot(fdr_list[:stop_idx], fdr_list[:stop_idx], linestyle='--', lw=3, color='k')
	ax[0,i].grid()
	ax[1,i].grid()
	ax[0,i].tick_params(axis='both', labelsize=12)
	ax[1,i].tick_params(axis='both', labelsize=12)
	# ax[1,i].set_ylim(top=1.05)

ax[1,0].set_xlabel('Target FDR', fontsize=fontsize)
ax[1,1].set_xlabel('Target FDR', fontsize=fontsize)
ax[0,0].set_ylabel('TPR', fontsize=fontsize)
ax[1,0].set_ylabel('FDR', fontsize=fontsize)

# Create handles and labels for the legend
label_order = list(reversed(methods_to_plot))
handles = [plt.Line2D([], [], color=method_specs[m]['color'],
					  linestyle=method_specs[m]['linestyle'],
					  lw=3.5) for m in label_order]
labels = [method_specs[m]['name'] for m in label_order]

# Add legend to the extra subplot
legend_ax = fig.add_subplot(gs[:, -1])
legend_ax.legend(handles, labels, loc='center', fontsize=16)
legend_ax.axis('off')

print()
print(f'Missing methods')
print(f'--------------------------------')
for string in missing_methods:
	print(string)

plt.tight_layout()
fig.subplots_adjust(wspace=0.15)
if save_figure:
	plt.savefig(f'{experiment}.png', dpi=300)
if show_figure:
	plt.show()