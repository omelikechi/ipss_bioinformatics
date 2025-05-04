# Analyze dependence on delta

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pickle

save_figure = False

response_type = 'reg'
p_grid = [500, 2000, 5000]
method = 'ipssgb'
B_grid = [50, 100, 150, 200]

# plot args
color = ['red', 'orange', 'limegreen', 'dodgerblue']
lw = 3
fontsize = 32

fig, ax = plt.subplots(2, len(p_grid), sharey='row', sharex='col', figsize=(18, 8))
ax = np.array(ax).reshape(2, -1)

for i, p in enumerate(p_grid):

	ax[0,i].set_title(f'$\\mathbf{{p = {p}}}$', fontsize=fontsize)

	# load results
	with open(f'./results/{method}/oc_{response_type}_{p}_{method}_B.pkl', "rb") as f:
		results = pickle.load(f)

	for j, B in enumerate(B_grid):

		simulation_config = results[B]['simulation_config']
		fdr_list = simulation_config['fdr_list']
		function_grid = simulation_config['function_grid']
		cutoff_grid = simulation_config['cutoff_grid']
		delta_grid = simulation_config['delta_grid']

		tpr_matrix = results[B]['tpr_matrix']
		fdr_matrix = results[B]['fdr_matrix']
		runtimes = results[B]['runtimes']

		runtime_mean = np.mean(runtimes)
		print(f'Average runtime, {B} subsampling steps: {runtime_mean:.1f} seconds')

		for function in function_grid:
			for cutoff in cutoff_grid:
				for delta in delta_grid:
					tpr_means = np.mean(tpr_matrix[function][cutoff][delta], axis=0)
					fdr_means = np.mean(fdr_matrix[function][cutoff][delta], axis=0)
					ax[0,i].plot(fdr_list, tpr_means, lw=lw, color=color[j], label=f'$B = {B}$')
					ax[1,i].plot(fdr_list, fdr_means, lw=lw, color=color[j], label=f'$B = {B}$')
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
lines = ax[0,len(p_grid)-1].get_lines()
custom_handles = [Line2D([0], [0], color=line.get_color(), linewidth=6) for line in lines]
labels = [line.get_label() for line in lines]
ax[0,len(p_grid)-1].legend(handles=custom_handles, labels=labels, loc='upper left', fontsize=20)

plt.tight_layout()
if save_figure:
	plt.savefig(f'sensitivity_B_{response_type}.png', dpi=300)
plt.show()