# Simulations and analyses of the detailed kinetic model

The code in this folder implements a detailed kinetic model of mitochondrial metabolism and fuel competition built from thermodynamic constraints. It provides notebooks for model construction, parameter sampling, fuel competition simulations, and robustness analyses. The model is derived from RECON3D using thermodynamic and kinetic modeling frameworks (pyTFA and SKiMPy). To use, install the dependencies listed below and open the notebooks.

## Installation

### Using Conda (Recommended)

The easiest way to install all dependencies is using conda:

```bash
# Add conda-forge and bioconda channels
conda config --add channels bioconda
conda config --add channels conda-forge
conda config --set channel_priority strict

# Create a new environment (Python 3.8 recommended)
conda create --name skimpy-env python=3.8
conda activate skimpy-env

# Install skimpy (includes pytfa as a dependency)
conda install weilandtd::skimpy

# Install additional dependencies
conda install numpy scipy pandas matplotlib seaborn h5py tqdm pyyaml
```

**Note:** SKiMPy was developed in Python 3.8-3.9. For best compatibility, use Python 3.8 or 3.9.

### Commercial Solvers (Required)

Running these models requires a licensed version of either CPLEX or Gurobi. For the paper, CPLEX was used. Make sure the solver supports Python 3.8-3.10. Free academic licenses are available for both solvers:
- [CPLEX Academic Initiative](https://www.ibm.com/academic/technology/data-science)
- [Gurobi Academic License](https://www.gurobi.com/academia/academic-program-and-licenses/)

## Contents

| File | Description |
| --- | --- |
| redgem_eletron_transport_chain.ipynb | Notebook extracting a reduced network arround the electron transport chain model from RECON3D |
| data_integration.ipynb | Notebook integrating thermodynamic and metabolite data into the reduced model |
| build_kinetic_model.ipynb | Notebook building kinetic models from TFA model and thermodynamic sampling |
| controling_concentrations.ipynb | Notebook analyzing controlling concentrations and pathway analysis |
| parameter_sampling.py | Python script for parameter sampling with regulation |
| parameter_sampling_no_reg.py | Python script for parameter sampling without canonical enzyme regulation |
| competition_dynamics.ipynb | Notebook simulating competitive fuel utilization dynamics |
| fuel_competition.ipynb | Notebook simulating fuel competition dynamics with regulation |
| fuel_competition_no_reg.ipynb | Notebook simulating fuel competition dynamics without canonical enzyme regulation |
| plot_fuel_competition_results.ipynb | Notebook plotting and analyzing fuel competition results with regulation (related to Figure 3) |
| plot_fuel_competition_results_no_reg.ipynb | Notebook plotting and analyzing fuel competition results without canonical enzyme regulation |
| robustness_perturbations.ipynb | Notebook performing robustness analysis with regulation |
| robustness_perturbations_no_reg.ipynb | Notebook performing robustness analysis without canonical enzyme regulation |
| plot_robustness_results.ipynb | Notebook plotting and comparing robustness analysis results (related to Figure S3) |
| reduced_model_ETC_core_20250228-213124.json | Base reduced genome-scale metabolic model with MILP constraint |
| reduced_model_ETC_core_20250228-213124_continuous.json | Reduced model without MILP constraint (sinlge flux directional profile) |
| reduced_model_ETC_core_fba_only_20250226-232416.json | Reduced model for flux balance analysis only (no thermodynamics) |
| reduced_model_ETC_core_20250228-213124_kinetic_curated.yml | Kinetic model with regulation|
| reduced_model_ETC_core_20250228-213124_kinetic_curated_h_const.yml | Kinetic model with regulation constant protons |
| reduced_model_ETC_core_20250228-213124_kinetic_curated_no_reg.yml | Kinetic model without canonical enzyme regulation|
| reduced_model_ETC_core_20250228-213124_tfa_sampling.csv | Thermodynamic flux analysis sampling results |
| reduced_model_ETC_core_20250228-213124_tfa_sampling_pruned_parameters.hdf5 | Parameter population with regulation (pruned for feasibility) |
| reduced_model_ETC_core_20250228-213124_tfa_sampling_pruned_parameters_no_reg.hdf5 | Parameter population without canonical enzyme regulation (pruned for feasibility) |
| reduced_model_ETC_core_20250228-213124_tfa_sampling_robust_parameters.hdf5 | Parameter population with regulation (robust to perturbations) |
| reduced_model_ETC_core_20250228-213124_tfa_sampling_robust_parameters_no_reg.hdf5 | Parameter population without canonical enzyme regulation (robust to perturbations) |

