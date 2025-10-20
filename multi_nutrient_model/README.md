# Simulations and analyses of the multi-nutrient model

The code in this folder implements a mechanistic multi‑nutrient metabolic model of fasted metabolic homeostatsis and a set of analyses used to explore insulin action, nutrient competition, and metabolic phenotypes. It provides reusable simulation functions and Jupyter notebooks that reproduce the figures and experiments from the study. To use, install the dependcies listed below and open the notebooks or import the simulation functions from the Python module.

## Installation

Install dependencies via pip using a requirements file:

```bash
pip install -r requirements.txt
```

requirements.txt:

```text
pytfa
numpy
scipy
pandas
matplotlib
seaborn
statannotations
```

## Contents

| File | Description |
| --- | --- |
| multi_nutrient_model.py | Python file containing functions for clamp, dynamic, and sensitivity simulations |
| steady_state_analysis.ipynb | Notebook computing reference steady-state fluxes from constraints (related to Figure S1B) |
| basins_of_attraction.ipynb | Notebook simulating and plotting basins of attraction (related to Figures 1 and S1) |
| type_I_diabetes.ipynb | Notebook simulating response to insufficient insulin secretion (related to Figure 1) |
| competitive_catabolism.ipynb | Notebook comparing competitive catabolism predictions to data (related to Figure 2) |
| hyperinsulinemic_euglycemic_clamp.ipynb | Notebook comparing the multi-nutrient prediction in the hyperinsulinemic-euglycemic clamp to experimental data (related to Figure 4) |
| insulin_action.ipynb | Notebook exploring the importance of different insulin actions (related to Figures 5A–B) |
| GTT_ITT_insulin_receptor_ko.ipynb | Notebook simulating loss of muscle and adipose insulin receptors |
| obesity_simulations.ipynb | Notebook simulating obesity in the multi-nutrient model (related to Figures 6A, 6B, 6D and S6A, S6B) |
| obesity_NHANES.ipynb | Notebook comparing obesity simulation to human data (related to Figure 6C) |
| robustness_obesity_prediction.ipynb | Notebook performing robustness analysis of obesity predictions using sensitivity analysis |
| sensitivity_analysis.ipynb | Notebook simulating parameter changes and interventions for metabolic syndrome (related to Figures 7 and S6C) |

