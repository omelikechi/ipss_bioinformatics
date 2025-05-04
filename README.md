# **Supplementary code for "Nonparametric IPSS: Fast, flexible feature selection with false discovery control" (*Bioinformatics*, 2025)**

<!-- TODO: Update after Zenodo upload -->
A snapshot of this code used for the paper is permanently archived with DOI:  
https://doi.org/10.5281/zenodo.XXXXXXX  

## **Important!**  
The full **integrated path stability selection (IPSS) package**, which includes expanded functionality and is designed for general use, is 
available here: [https://github.com/omelikechi/ipss](https://github.com/omelikechi/ipss)

## Repository Structure
The `simulations` folder contains code for all simulation results (Sections 3, S3, and S5)
	- `generate_data.py`: Code for generating synthetic data from user-specified models
	- `methods.py`: Python implementations of each feature selection method
	- `methods.r`: R implementations of certain feature selection methods, which are then wrapped in Python in `methods.py`
	- `plot_gaussian_results.py`: Reproduces Figures 1 and S1 (the Gaussian simulation results; see Section 3.2)
	- `plot_rnaseq_results.py`: Reproduces Figures 2 and S2 (the simulation results using real RNA-seq data; see Section 3.3)
	- `simulation.py`: Main script for running all of the simulations in Section 3
	- The `data` folder contains
		- `ovarian_rnaseq.npy`: The ovarian cancer RNA-seq feature matrix
	- The `KOBT` and `SSBoost` folders contain customized code for running these methods
	- The `results` folder contains all of the simulation results for each method
	- The `sensitivity_analysis` folder contains code for reproducing all parameter sensitivity results (Section S5)
		- `sensitivity.py`: Main script for running sensitivity analyses over IPSS hyperparameters
		- `sensitivity_B.py`: Specialized sensitivity experiments for the number of subsamples, B
		- `extract_results.py`: Helper to extract results from different sensitivity experiments
		- `plot_B_results.py`: Reproduces Figures S16 and S17 (sensitivity to the number of subsamples B)  
		- `plot_cutoff_results.py`: Reproduces Figures S12 and S13 (sensitivity to the integral cutoff parameter C) 
		- `plot_delta_results.py`: Reproduces Figures S14 and S15 (sensitivity to the probability measure parameter delta)  
		- `plot_f_results.py`: Reproduces Figures S18 and S19 (dependence on the choice of function f)
		- The `data` folder contains the same ovarian cancer RNA-seq feature matrix as the `data` folder in `simulations`, duplicated 
		here for convenience due to relative path constraints
		- The `ipss_sensitivity` folder contains code for running IPSS with different hyperparameters
		- The `results` folder contains all of the simulation results for each sensitivity experiment
The `cancer_studies` folder contains all code for implementing the ovarian cancer and glioma case studies (Sections 4 and S4)
	- `load_cancer_data.py`: Code for loading cancer data, contained in the `data` folder
	- `methods.py`: Python implementations of each feature selection method
	- `methods.r`: R implementations of certain feature selection methods, which are then wrapped in Python in `methods.py`
	- The `cv_study` folder contains code for reproducing the cross-validation results (Section 4 and S4.4)
		- `cv_helpers.py`: Helper functions called in `cv_study.py`
		- `cv_study.py`: Main script for running all cross-validation studies
		- `plot_cv_results.py`: Reproduces Figure 3 and Figures S6 through S11 in the Supplement
		- The `cv_results` folder contains all of the cross-validation results for each method
	- The `data` folder contains all of the ovarian cancer and glioma data considered in this work; all data are from The
	Cancer Genome Atlas and were downloaded for free from LinkedOmics
	- The `lit_search` folder contains code for reproducing the literature search results (Section 4 and Section S4.3)
		- `cancer.py`: Main script for performing feature selection on the ovarian cancer and glioma datasets
		- `generate_table.py`: Code for generating the LaTeX tables in Section S4.3
		- `lit_search.py`: Code for searching Europe PMC to quantify literature support for selected features
		- The `cancer_results` folder contains the features selected by each method
		- The `lit_results` folder contains the literature results for each method

## Datasets
All of the raw data used in this work is included in this GitHub repository
- Ovarian cancer data from The Cancer Genome Atlas:
	- Source: https://www.linkedomics.org/data_download/TCGA-OV/  
	- Files in this repository: 
		- `simulations/data/ovarian_rnaseq.npy`
		- `simulations/sensitivity_analysis/data/ovarian_rnaseq.npy` (identical to above)
		- `cancer_studies/data/ovarian`
- Glioma data from The Cancer Genome Atlas:
	- Source: https://www.linkedomics.org/data_download/TCGA-GBMLGG/  
	- Files in this repository: 
		- `cancer_studies/data/glioma`
