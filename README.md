# **Supplementary code for "Nonparametric IPSS: Fast, flexible feature selection with false discovery control" (*Bioinformatics*, 2025)**

A snapshot of the source code used in this paper is archived here:  
https://doi.org/10.5281/zenodo.15335289 

## **Important!**  
The full **integrated path stability selection (IPSS) package**, which includes expanded functionality and is designed for general use, is 
available here: [https://github.com/omelikechi/ipss](https://github.com/omelikechi/ipss)

## Repository structure
The `simulations` folder contains code for the simulation studies (Sections 3, S3, and S5)
- `generate_data.py`: Generate synthetic data from user-specified models
- `methods.py`: Python implementation of each feature selection method
- `methods.r`: R implementations of certain feature selection methods, wrapped in Python in `methods.py`
- `plot_gaussian_results.py`: Reproduces Figures 1 and S1 (Gaussian simulation results)
- `plot_rnaseq_results.py`: Reproduces Figures 2 and S2 (RNA-seq simulation results)
- `simulation.py`: Main script for running all simulations in Section 3
- The `data` folder contains
	- `ovarian_rnaseq.npy`: Ovarian cancer RNA-seq feature matrix
- The `KOBT` and `SSBoost` folders contain customized code for running these methods
- The `results` folder contains simulation results for each method
- The `sensitivity_analysis` folder contains code for the parameter sensitivity analyses (Section S5)
	- `sensitivity.py`: Main script for running sensitivity analyses of IPSS hyperparameters
	- `sensitivity_B.py`: Specialized sensitivity experiments for the number of subsamples B
	- `extract_results.py`: Helper to extract results from sensitivity experiments
	- `plot_B_results.py`: Reproduces Figures S16 and S17 (sensitivity to number of subsamples B)  
	- `plot_cutoff_results.py`: Reproduces Figures S12 and S13 (sensitivity to integral cutoff parameter C) 
	- `plot_delta_results.py`: Reproduces Figures S14 and S15 (sensitivity to measure parameter delta)  
	- `plot_f_results.py`: Reproduces Figures S18 and S19 (dependence on function f)
	- The `data` folder contains the same ovarian cancer RNA-seq feature matrix as the `data` folder in `simulations`, duplicated 
	in this folder for convenience
	- The `ipss_sensitivity` folder contains code for running IPSS with different hyperparameters
	- The `results` folder contains simulation results for each sensitivity experiment

The `cancer_studies` folder contains code for the cancer studies (Sections 4 and S4)
- `load_cancer_data.py`: Code for loading cancer data
- `methods.py`: Python implementations of each feature selection method
- `methods.r`: R implementations of certain feature selection methods, wrapped in Python in `methods.py`
- The `cv_study` folder contains code for the cross-validation studies (Section 4 and S4.4)
	- `cv_helpers.py`: Helper functions called in `cv_study.py`
	- `cv_study.py`: Main script for running all cross-validation studies
	- `plot_cv_results.py`: Reproduces Figure 3 and Figures S6 through S11 in the Supplement
	- The `cv_results` folder contains all of the cross-validation results for each method
- The `data` folder contains all of the ovarian cancer and glioma data considered in this work; all datasets are from The
Cancer Genome Atlas and were downloaded for free from LinkedOmics
- The `lit_search` folder contains code for the literature search results (Section 4 and Section S4.3)
	- `cancer.py`: Main script for performing feature selection on the cancer data
	- `generate_table.py`: Code for generating the LaTeX tables in Section S4.3
	- `lit_search.py`: Code for searching Europe PMC to quantify literature support for selected features
	- The `cancer_results` folder contains the features selected by each method
	- The `lit_results` folder contains the literature results for each method

## Data availability 
All datasets used in this work are included in this repository
- Ovarian cancer data from The Cancer Genome Atlas
	- Source: https://www.linkedomics.org/data_download/TCGA-OV/  
	- Files in this repository: 
		- `simulations/data/ovarian_rnaseq.npy`
		- `simulations/sensitivity_analysis/data/ovarian_rnaseq.npy` (identical to above)
		- `cancer_studies/data/ovarian`
- Glioma data from The Cancer Genome Atlas
	- Source: https://www.linkedomics.org/data_download/TCGA-GBMLGG/  
	- Files in this repository: 
		- `cancer_studies/data/glioma`
