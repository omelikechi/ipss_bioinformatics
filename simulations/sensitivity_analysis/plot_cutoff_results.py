# Analyze dependence on delta

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pickle

save_figure = False

response_type = 'reg'
p_grid = [500, 2000, 5000]
method = 'ipssgb'
parameter = 'cutoff'
cutoffs_to_plot = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

# plot args
colors = ['red', 'orange', 'gold', 'limegreen', 'deepskyblue', 'blue', 'magenta', 'purple']
colors = colors[::-1]
lw = 3
fontsize = 32

fig, ax = plt.subplots(2, len(p_grid), sharey='row', figsize=(18,8))
ax = np.array(ax).reshape(2,-1)

for i, p in enumerate(p_grid):

	ax[0,i].set_title(f'$\\mathbf{{p = {p}}}$', fontsize=fontsize)

	# load results
	with open(f'./results/{method}/oc_{response_type}_{p}_{method}_{parameter}.pkl', "rb") as f:
		results = pickle.load(f)

	simulation_config = results['simulation_config']
	fdr_list = simulation_config['fdr_list']
	function_grid = simulation_config['function_grid']
	cutoff_grid = simulation_config['cutoff_grid']
	delta_grid = simulation_config['delta_grid']

	if cutoffs_to_plot is None:
		cutoffs_to_plot = cutoff_grid

	color_dict = dict(zip(cutoffs_to_plot, colors))

	tpr_matrix = results['tpr_matrix']
	fdr_matrix = results['fdr_matrix']
	runtimes = results['runtimes']

	runtime_mean = np.mean(runtimes)
	print(f'Average runtime = {runtime_mean:.1f} seconds')

	for function in function_grid:
		for cutoff in cutoff_grid:
			for delta in delta_grid:
				if cutoff in cutoffs_to_plot:
					tpr_means = np.mean(tpr_matrix[function][cutoff][delta], axis=0)
					fdr_means = np.mean(fdr_matrix[function][cutoff][delta], axis=0)
					if color_dict:
						ax[0,i].plot(fdr_list, tpr_means, lw=lw, color=color_dict[cutoff], label=f'$C = {cutoff}$')
						ax[1,i].plot(fdr_list, fdr_means, lw=lw, color=color_dict[cutoff], label=f'$C = {cutoff}$')
					else:
						ax[0,i].plot(fdr_list, tpr_means, lw=lw, label=f'$C = {cutoff}$')
						ax[1,i].plot(fdr_list, fdr_means, lw=lw, label=f'$C = {cutoff}$')
	ax[1,i].plot(fdr_list, fdr_list, linestyle='--', lw=lw, color='k')
	ax[0,i].grid()
	ax[1,i].grid()
	ax[0,i].tick_params(axis='both', labelsize=12)
	ax[1,i].tick_params(axis='both', labelsize=12)

target_fdr_location = 1 if len(p_grid) == 3 else 0
ax[1,target_fdr_location].set_xlabel('Target FDR', fontsize=fontsize)
ax[0,0].set_ylabel('TPR', fontsize=fontsize)
ax[1,0].set_ylabel('FDR', fontsize=fontsize)

# legend
lines = ax[0,2].get_lines()
custom_handles = [Line2D([0], [0], color=line.get_color(), linewidth=6) for line in lines]
labels = [line.get_label() for line in lines]
ax[0,2].legend(handles=custom_handles, labels=labels, loc='best', fontsize=12)

plt.tight_layout()
if save_figure:
	plt.savefig(f'sensitivity_cutoff_{response_type}.png', dpi=300)
plt.show()