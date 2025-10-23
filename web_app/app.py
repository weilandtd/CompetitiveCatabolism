from flask import Flask, render_template, request, jsonify
import sys
import os
import numpy as np
import pandas as pd
import json
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add the multi_nutrient_model directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'multi_nutrient_model'))

from multi_nutrient_model import (
    ref_parameters, perturbation_dynamics, perturbation_steady_state,
    insulin_clamp_dynamic, I0, PARAMETER_NAMES, PARAMETER_DESCRIPTIONS,
    PARAMETER_LATEX, steady_state, fluxes, sensitivity_analysis, change_parameters,
    TAU_INS
)

# Exclude infusion rate parameters from perturbation lists
# These are 0 by default, so fold changes don't work
INFUSION_PARAMS = ['R_insulin', 'R_lactate', 'R_glucose', 'R_fatty_acids', 'R_3HB']
PARAMETER_NAMES_NO_INFUSION = [p for p in PARAMETER_NAMES if p not in INFUSION_PARAMS]
PARAMETER_DESCRIPTIONS_NO_INFUSION = {k: v for k, v in PARAMETER_DESCRIPTIONS.items() if k not in INFUSION_PARAMS}
PARAMETER_LATEX_NO_INFUSION = {k: v for k, v in PARAMETER_LATEX.items() if k not in INFUSION_PARAMS}

# Flux scaling based on total ATP production rate
# ATP per O2 (3 per O)
PO2 = 5.0
# Whole body oxygen consumption rate ~ 2000 nmol/min/gBW
vO2 = 2000 * 0.7 
# ATP production rate
vATP = PO2 * vO2 
# Scale the oxygen fluxes by vE
vE = vATP 


app = Flask(__name__, template_folder='template', static_folder='static')

# Set matplotlib style with larger fonts
plt.rcParams.update({
    'font.size': 14,
    'font.family': 'Arial',
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
})
sns.set_style("whitegrid")

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

def plot_to_svg(fig):
    """Convert matplotlib figure to SVG string with responsive sizing"""
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    buf.seek(0)
    svg_str = buf.read().decode('utf-8')
    plt.close(fig)
    
    # Remove fixed width/height attributes to make SVG responsive
    # Replace with viewBox to maintain aspect ratio while allowing scaling
    import re
    # Extract width and height values
    width_match = re.search(r'width="([^"]+)pt"', svg_str)
    height_match = re.search(r'height="([^"]+)pt"', svg_str)
    
    if width_match and height_match:
        width = width_match.group(1)
        height = height_match.group(1)
        # Remove width and height attributes and add viewBox
        svg_str = re.sub(r'width="[^"]+pt"', '', svg_str)
        svg_str = re.sub(r'height="[^"]+pt"', '', svg_str)
        svg_str = re.sub(r'<svg', f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet"', svg_str)
    
    return svg_str

def check_simulation_warnings(data_df, scale_factors=None):
    """Check simulation results for warnings (negative values or very large values)
    
    Args:
        data_df: DataFrame with simulation results
        scale_factors: Dict of column names to their scaling factors (optional)
    
    Returns:
        List of warning messages
    """
    warnings = []
    
    # Define reasonable upper bounds for physiological variables (after scaling)
    upper_bounds = {
        'G': 500,    # Glucose > 50 mM is extremely high
        'I': 100,   # Insulin > 100 ng/mL is very high
        'F': 100,    # Fatty acids > 10 mM is very high
        'K': 100,    # Ketones > 10 mM is very high
        'L': 200,    # Lactate > 20 mM is very high
        'IA': 100,   # Insulin action should be reasonable
    }
    
    # Check each variable
    for col in data_df.columns:
        if col == 'time':
            continue
            
        values = data_df[col]
        
        # Check for neative values (< 1e-1 = 0.1)
        if (values < -1e-1).any():
            min_val = values.min()
            warnings.append(f"⚠️ Warning: {col} contains negative values (min: {min_val:.3e}). "
                          "This indicates unrealistic parameter values.")
        
        # Check for very large values
        if col in upper_bounds:
            max_val = values.max()
            if max_val > upper_bounds[col]:
                warnings.append(f"⚠️ Warning: {col} reaches very high values (max: {max_val:.2f}). "
                              "This indicates unrealistic parameter values.")
        
        # Check for NaN or Inf
        if values.isna().any():
            warnings.append(f"⚠️ Warning: {col} contains NaN (not-a-number) values. "
                          "Simulation may have failed.")
        if np.isinf(values).any():
            warnings.append(f"⚠️ Warning: {col} contains infinite values. "
                          "This indicates numerical overflow.")
    
    return warnings

def create_individual_plot(plot_func, **kwargs):
    """Create an individual plot and return as SVG string
    
    Args:
        plot_func: Function that creates and returns a matplotlib axis
        **kwargs: Arguments to pass to plot_func
    
    Returns:
        SVG string
    """
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plot_func(ax, **kwargs)
    return plot_to_svg(fig)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    """About page with project information"""
    return render_template('about.html')

@app.route('/privacy')
def privacy():
    """Privacy policy page"""
    return render_template('privacy.html')

@app.route('/terms')
def terms():
    """Terms of service page"""
    return render_template('terms.html')

@app.route('/dynamic_response')
def dynamic_response():
    """Page for Type 1 Diabetes-like dynamic response simulation"""
    return render_template('dynamic_response.html', 
                         parameter_names=PARAMETER_NAMES_NO_INFUSION,
                         parameter_descriptions=PARAMETER_DESCRIPTIONS_NO_INFUSION,
                         parameter_latex=PARAMETER_LATEX_NO_INFUSION)

@app.route('/insulin_clamp')
def insulin_clamp():
    """Page for hyperinsulinemic-euglycemic clamp simulation"""
    return render_template('insulin_clamp.html',
                         parameter_names=PARAMETER_NAMES_NO_INFUSION,
                         parameter_descriptions=PARAMETER_DESCRIPTIONS_NO_INFUSION,
                         parameter_latex=PARAMETER_LATEX_NO_INFUSION)

@app.route('/tolerance_tests')
def tolerance_tests():
    """Page for GTT/ITT simulation with receptor knockouts"""
    return render_template('tolerance_tests.html',
                         parameter_names=PARAMETER_NAMES_NO_INFUSION,
                         parameter_descriptions=PARAMETER_DESCRIPTIONS_NO_INFUSION,
                         parameter_latex=PARAMETER_LATEX_NO_INFUSION)

@app.route('/obesity')
def obesity():
    """Page for obesity simulation"""
    return render_template('obesity.html',
                         parameter_names=PARAMETER_NAMES_NO_INFUSION,
                         parameter_descriptions=PARAMETER_DESCRIPTIONS_NO_INFUSION,
                         parameter_latex=PARAMETER_LATEX_NO_INFUSION)

@app.route('/treatment')
def treatment():
    """Page for treatment simulation"""
    return render_template('treatment.html',
                         parameter_names=PARAMETER_NAMES_NO_INFUSION,
                         parameter_descriptions=PARAMETER_DESCRIPTIONS_NO_INFUSION,
                         parameter_latex=PARAMETER_LATEX_NO_INFUSION)

@app.route('/api/run_dynamic', methods=['POST'])
def run_dynamic():
    """Run dynamic response simulation (Type 1 diabetes-like)"""
    try:
        data = request.json
        
        # Get parameters
        time_max = float(data.get('time_max', 150))
        time_perturbation = float(data.get('time_perturbation', 30))
        parameter = data.get('parameter', 'Imax')
        fold_change = float(data.get('fold_change', 0))
        
        # Additional parameter perturbations (as fold changes)
        param_dict = data.get('parameters', {})
        
        # Get base parameters
        p = ref_parameters()
        
        # Apply parameter changes as fold changes
        if param_dict:
            keys = list(param_dict.keys())
            fold_changes = [float(param_dict[k]) for k in keys]
            values = [fold_changes[i] * p[PARAMETER_NAMES.index(keys[i])] for i in range(len(keys))]
            p = change_parameters(p, values, ix=keys)
        
        # Run simulation before perturbation
        time1 = np.linspace(0, time_perturbation, 50)
        X1, F1 = perturbation_dynamics(time1, 1.0, p=p)
        
        # Run simulation after perturbation
        time2 = np.linspace(time_perturbation, time_max, 100)
        param_kwargs = {parameter: fold_change * p[PARAMETER_NAMES.index(parameter)]}
        X2, F2 = perturbation_dynamics(time2, 1.0, p=p, **param_kwargs)
        
        # Concatenate results
        X = pd.concat([X1, X2], axis=0)
        
        # Scale concentrations (typical physiological values)
        X['G'] = X['G'] * 6  # 6 mM glucose (mouse fasted level)
        X['F'] = X['F'] * 0.5  # 0.5 mM FFA
        X['K'] = X['K'] * 0.5  # 0.5 mM 3HB
        X['L'] = X['L'] * 0.7  # 0.7 mM lactate

        # Scale insulin concentration to ng/mL (mouse scale)
        X['I'] = X['I'] / I0  * 0.4 # Scale to typical insulin levels in mice
        
        # Scale insulin action relative to I0
        X['IA'] = X['IA'] / I0
        
        # Create individual plots
        variables = ['G','L' ,'F', 'K', 'I']
        labels = ['Glucose (mM)','Lactate (mM)', 'Fatty acids (mM)', '3-Hydroxybutyrate (mM)', 'Insulin (ng/mL)']
        
        def plot_timeseries(ax, var, label):
            ax.plot(X['time'], X[var], linewidth=2, color='#0891b2')  # cyan-600 primary color
            ax.axvline(time_perturbation, color='#f97316', linestyle='--', alpha=0.5)  # orange-500 accent
            ax.set_xlabel('Time (min)')
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
            sns.despine(ax=ax)
        
        plots = []
        for var, label in zip(variables, labels):
            plot_data = create_individual_plot(plot_timeseries, var=var, label=label)
            plots.append({
                'id': var,
                'title': f'{label.split(" ")[0]} Response',
                'data': plot_data
            })
        
        # Check for warnings
        warnings = check_simulation_warnings(X[['time', 'G', 'F', 'K', 'L', 'I', 'IA']])
        
        # Prepare data for download with better column labels and perturbation info
        X_download = X[['time', 'G', 'F', 'K', 'L', 'I', 'IA']].copy()
        
        # Rename columns with full names and units
        X_download = X_download.rename(columns={
            'time': 'Time (min)',
            'G': 'Glucose (mM)',
            'F': 'Fatty_acids (mM)',
            'K': '3-Hydroxybutyrate (mM)',
            'L': 'Lactate (mM)',
            'I': 'Insulin (ng/mL)',
            'IA': 'Insulin_action (relative to I0)'
        })
        
        # Add perturbation information columns
        X_download['Primary_perturbation_parameter'] = parameter
        X_download['Primary_perturbation_fold_change'] = fold_change
        X_download['Perturbation_onset_time (min)'] = time_perturbation
        
        # Add additional perturbations if any
        if param_dict:
            additional_params_str = '; '.join([f"{k}={v}x" for k, v in param_dict.items()])
            X_download['Additional_perturbations'] = additional_params_str
        else:
            X_download['Additional_perturbations'] = 'None'
        
        data_csv = X_download.to_csv(index=False)
        
        return jsonify({
            'success': True,
            'plots': plots,
            'data': data_csv,
            'warnings': warnings
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/run_clamp', methods=['POST'])
def run_clamp():
    """Run insulin clamp simulation"""
    try:
        data = request.json
        
        # Get parameters
        insulin_level = data.get('insulin_level', 'low')  # 'low' or 'high'
        time_max = float(data.get('time_max', 120))
        infusion_type = data.get('infusion_type', 'none')
        infusion_amount = float(data.get('infusion_amount', 0))
        
        # Additional parameter perturbations (as fold changes)
        param_dict = data.get('parameters', {})
        
        # Get base parameters
        p = ref_parameters()
        
        # Apply parameter changes as fold changes
        if param_dict:
            keys = list(param_dict.keys())
            fold_changes = [float(param_dict[k]) for k in keys]
            values = [fold_changes[i] * p[PARAMETER_NAMES.index(keys[i])] for i in range(len(keys))]
            p = change_parameters(p, values, ix=keys)
        
        # Set insulin dose based on selection (matching hyperinsulinemic_euglycemic_clamp.ipynb)
        # vI_low = I0/TAU_INS (double basal fasted insulin levels)
        # vI_high = I0/TAU_INS * 3 (three times low dose)
        vI_low = I0 / TAU_INS
        vI_high = I0 / TAU_INS * 3
        
        insulin_dose = vI_low if insulin_level == 'low' else vI_high
        insulin_label = 'Low Dose' if insulin_level == 'low' else 'High Dose'
        
        # Run simulation
        time = np.linspace(0, time_max, 200)
        
        # Saline (no insulin, always displayed)
        X_saline, GIR_saline = insulin_clamp_dynamic(0, time, 1.0, p=p)
        X_saline['G'] = X_saline['G'] * 6  # 6 mM glucose (mouse scale)
        X_saline['F'] = X_saline['F'] * 0.5
        X_saline['K'] = X_saline['K'] * 0.5
        X_saline['L'] = X_saline['L'] * 0.7
        X_saline['I'] = X_saline['I'] / I0 * 0.4  # Scale to ng/mL (mouse scale)
        X_saline['IA'] = X_saline['IA'] / I0  # Scale relative to I0
        X_saline['condition'] = 'Saline'
        GIR_saline['condition'] = 'Saline'
        
        # Baseline (insulin clamp, no infusion)
        X_baseline, GIR_baseline = insulin_clamp_dynamic(insulin_dose, time, 1.0, p=p)
        X_baseline['G'] = X_baseline['G'] * 6  # 6 mM glucose (mouse scale)
        X_baseline['F'] = X_baseline['F'] * 0.5
        X_baseline['K'] = X_baseline['K'] * 0.5
        X_baseline['L'] = X_baseline['L'] * 0.7
        X_baseline['I'] = X_baseline['I'] / I0 * 0.4  # Scale to ng/mL (mouse scale)
        X_baseline['IA'] = X_baseline['IA'] / I0  # Scale relative to I0
        X_baseline['condition'] = 'Insulin'
        GIR_baseline['condition'] = 'Insulin'
        
        # With infusion if specified
        X_infusion = None
        GIR_infusion = None
        if infusion_type != 'none':
            # Conversion: 0.01 model units = 70 nmol/min/gBW
            # infusion_amount is already in nmol/min/gBW
            
            infusion_param_names = {
                'fatty_acids': 'R_fatty_acids',
                'lactate': 'R_lactate',
                '3HB': 'R_3HB'
            }

            scaling = {
                'fatty_acids': 1.0,
                'lactate': 200/150, 
                '3HB': 14/27
            }
            
            # Convert from nmol/min/gBW to model units (70 nmol/min/gBW = 0.01)
            # Based on total ATP production scaling
            infusion_rate_model = infusion_amount * 0.01 / 70.0 * scaling[infusion_type]

            
            # Create kwargs for insulin_clamp_dynamic
            param_name = infusion_param_names[infusion_type]
            infusion_kwargs = {param_name: infusion_rate_model}
            
            X_infusion, GIR_infusion = insulin_clamp_dynamic(insulin_dose, time, 1.0, p=p, **infusion_kwargs)
            X_infusion['G'] = X_infusion['G'] * 6  # 6 mM glucose (mouse scale)
            X_infusion['F'] = X_infusion['F'] * 0.5
            X_infusion['K'] = X_infusion['K'] * 0.5
            X_infusion['L'] = X_infusion['L'] * 0.7
            X_infusion['I'] = X_infusion['I'] / I0 * 0.4  # Scale to ng/mL (mouse scale)
            X_infusion['IA'] = X_infusion['IA'] / I0  # Scale relative to I0
            X_infusion['condition'] = f'Insulin + {infusion_type.replace("_", " ").title()}'
            GIR_infusion['condition'] = f'Insulin + {infusion_type.replace("_", " ").title()}'
        
        # Create individual plots
        plots = []
        # Convert model GIR to mg/kg/min
        # Assumptions based on model-unit conversions in this file:
        #   0.01 model units = 70 nmol / min / gBW
        # Conversion steps:
        #   model_units -> nmol/min/g: model_to_nmol = 70 / 0.01
        #   nmol -> mg: mg_per_nmol = 180 g/mol * 1e-6 mg/nmol = 180e-6 mg/nmol
        #   per g -> per kg: multiply by 1000
        model_to_nmol = 70.0 / 0.01
        mg_per_nmol = 180.0e-6
        g_to_kg = 1000.0
        gir_scale = model_to_nmol * mg_per_nmol * g_to_kg  # final factor to get mg/kg/min

        # Apply scaling to GIR time courses
        GIR_saline['GIR'] = GIR_saline['GIR'] * gir_scale
        GIR_baseline['GIR'] = GIR_baseline['GIR'] * gir_scale
        if GIR_infusion is not None:
            GIR_infusion['GIR'] = GIR_infusion['GIR'] * gir_scale

        # Combine GIR data for plotting
        gir_list = [GIR_saline, GIR_baseline]
        if GIR_infusion is not None:
            gir_list.append(GIR_infusion)
        GIR_combined = pd.concat(gir_list, ignore_index=True)
        
        # Define color palette matching website theme
        condition_colors = {
            'Saline': '#64748b',      # slate-500 (neutral baseline)
            'Insulin': '#0891b2',     # cyan-600 (primary brand color)
        }
        if GIR_infusion is not None:
            condition_colors[GIR_infusion['condition'].iloc[0]] = '#f97316'  # orange-500 (high contrast accent)
        
        # GIR time course using seaborn
        def plot_gir_timecourse(ax):
            sns.lineplot(data=GIR_combined, x='time', y='GIR', hue='condition',
                        palette=condition_colors, linewidth=2, ax=ax)
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('GIR (mg/kg/min)')
            ax.set_title(f'Glucose Infusion Rate - {insulin_label}', pad=30)
            ax.legend(title='', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False)
            sns.despine(ax=ax)
        
        plot_data = create_individual_plot(plot_gir_timecourse)
        plots.append({
            'id': 'GIR_timecourse',
            'title': 'Glucose Infusion Rate',
            'data': plot_data
        })
        
        # Calculate steady state values (last 20% of simulation)
        cutoff = time_max * 0.8
        
        # Build steady state dataframe
        metabolite_map = {'G': 'Glucose', 'F': 'FFA', 'K': '3HB', 'L': 'Lactate'}
        ss_data = []
        
        for metabolite_code, metabolite_name in metabolite_map.items():
            ss_data.append({
                'Metabolite': metabolite_name,
                'Condition': 'Saline',
                'Concentration': X_saline[X_saline['time'] > cutoff][metabolite_code].mean()
            })
            ss_data.append({
                'Metabolite': metabolite_name,
                'Condition': 'Insulin',
                'Concentration': X_baseline[X_baseline['time'] > cutoff][metabolite_code].mean()
            })
            if X_infusion is not None:
                ss_data.append({
                    'Metabolite': metabolite_name,
                    'Condition': X_infusion['condition'].iloc[0],
                    'Concentration': X_infusion[X_infusion['time'] > cutoff][metabolite_code].mean()
                })
        
        ss_df = pd.DataFrame(ss_data)
        
        # Create individual bar plot for each metabolite
        for metabolite_name in metabolite_map.values():
            metabolite_df = ss_df[ss_df['Metabolite'] == metabolite_name]
            
            def plot_metabolite(ax, met_df=metabolite_df, met_name=metabolite_name):
                sns.barplot(data=met_df, x='Condition', y='Concentration',
                           palette=condition_colors, alpha=0.7, ax=ax)
                ax.set_ylabel('Concentration (mM)')
                ax.set_xlabel('')
                ax.set_title(f'{met_name} - {insulin_label}', pad=20)
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                sns.despine(ax=ax)
            
            plot_data = create_individual_plot(plot_metabolite)
            plots.append({
                'id': f'steady_state_{metabolite_name.lower().replace(" ", "_")}',
                'title': f'{metabolite_name} Steady State',
                'data': plot_data
            })
        
        # Prepare data for download with better column labels
        if X_infusion is not None:
            X_all = pd.concat([X_saline, X_baseline, X_infusion])
        else:
            X_all = pd.concat([X_saline, X_baseline])
        
        # Rename columns with full names and units
        X_all_download = X_all.copy()
        X_all_download = X_all_download.rename(columns={
            'time': 'Time (min)',
            'G': 'Glucose (mM)',
            'F': 'Fatty_acids (mM)',
            'K': '3-Hydroxybutyrate (mM)',
            'L': 'Lactate (mM)',
            'I': 'Insulin (ng/mL)',
            'IA': 'Insulin_action (relative to I0)',
            'condition': 'Experimental_condition'
        })
        
        # Add perturbation information
        X_all_download['Insulin_dose'] = insulin_label
        X_all_download['Infusion_type'] = infusion_type.replace('_', ' ').title() if infusion_type != 'none' else 'None'
        X_all_download['Infusion_rate (nmol/min/gBW)'] = infusion_amount if infusion_type != 'none' else 0
        
        # Add additional parameter perturbations if any
        if param_dict:
            additional_params_str = '; '.join([f"{k}={v}x" for k, v in param_dict.items()])
            X_all_download['Additional_perturbations'] = additional_params_str
        else:
            X_all_download['Additional_perturbations'] = 'None'
        
        data_csv = X_all_download.to_csv(index=False)
        
        # Check for warnings in all conditions
        warnings = []
        warnings.extend(check_simulation_warnings(X_saline[['time', 'G', 'F', 'K', 'L', 'I', 'IA']]))
        warnings.extend(check_simulation_warnings(X_baseline[['time', 'G', 'F', 'K', 'L', 'I', 'IA']]))
        if X_infusion is not None:
            warnings.extend(check_simulation_warnings(X_infusion[['time', 'G', 'F', 'K', 'L', 'I', 'IA']]))
        # Remove duplicates
        warnings = list(set(warnings))
        
        return jsonify({
            'success': True,
            'plots': plots,
            'data': data_csv,
            'warnings': warnings
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/run_tolerance_tests', methods=['POST'])
def run_tolerance_tests():
    """Run GTT and ITT simulations side by side"""
    try:
        data = request.json
        
        # Get parameters
        time_max = float(data.get('time_max', 120))
        parameter = data.get('parameter', None)  # Parameter to perturb
        fold_change = float(data.get('fold_change', 1.0))  # Fold change for perturbation
        
        # Additional parameter perturbations (as fold changes)
        param_dict = data.get('parameters', {})
        
        # Get base parameters
        p_baseline = ref_parameters()
        
        # Apply additional parameter changes as fold changes
        if param_dict:
            keys = list(param_dict.keys())
            fold_changes_add = [float(param_dict[k]) for k in keys]
            values = [fold_changes_add[i] * p_baseline[PARAMETER_NAMES.index(keys[i])] for i in range(len(keys))]
            p_baseline = change_parameters(p_baseline, values, ix=keys)
        
        # Create perturbed parameters if specified
        if parameter:
            p_perturbed = change_parameters(p_baseline, 
                                           [fold_change * p_baseline[PARAMETER_NAMES.index(parameter)]], 
                                           ix=[parameter])
        else:
            p_perturbed = p_baseline
        
        # Define simulation
        time = np.linspace(0, time_max, 200)
        
        # Run GTT (Glucose Tolerance Test)
        # Simulate glucose bolus injection as in notebook: R_glucose = 0.06 for first 15 min
        time_gtt_1 = np.linspace(0, 15, 50)
        X_gtt_baseline_1, _ = perturbation_dynamics(time_gtt_1, 1.0, p=p_baseline, R_glucose=0.06)
        time_gtt_2 = np.linspace(15, time_max, 150)
        x0_gtt_baseline = X_gtt_baseline_1.iloc[-1][['L', 'G', 'F', 'K', 'I', 'IA']].values
        X_gtt_baseline_2, _ = perturbation_dynamics(time_gtt_2, 1.0, X0=x0_gtt_baseline, p=p_baseline, R_glucose=0.0)
        X_gtt_baseline = pd.concat([X_gtt_baseline_1, X_gtt_baseline_2], axis=0).reset_index(drop=True)
        
        X_gtt_perturbed_1, _ = perturbation_dynamics(time_gtt_1, 1.0, p=p_perturbed, R_glucose=0.06)
        x0_gtt_perturbed = X_gtt_perturbed_1.iloc[-1][['L', 'G', 'F', 'K', 'I', 'IA']].values
        X_gtt_perturbed_2, _ = perturbation_dynamics(time_gtt_2, 1.0, X0=x0_gtt_perturbed, p=p_perturbed, R_glucose=0.0)
        X_gtt_perturbed = pd.concat([X_gtt_perturbed_1, X_gtt_perturbed_2], axis=0).reset_index(drop=True)
        
        # Run ITT (Insulin Tolerance Test)
        # Simulate insulin bolus injection by setting initial insulin concentration as in notebook
        time_itt_1 = np.linspace(0, 1, 10)
        X_itt_baseline_1, _ = perturbation_dynamics(time_itt_1, 1.0, p=p_baseline)
        time_itt_2 = np.linspace(1, time_max, 190)
        x0_itt_baseline = steady_state(1, p_baseline)
        x0_itt_baseline[-2] = I0 * 60  # Set insulin to 60× basal
        X_itt_baseline_2, _ = perturbation_dynamics(time_itt_2, 1.0, X0=x0_itt_baseline, p=p_baseline)
        X_itt_baseline = pd.concat([X_itt_baseline_1, X_itt_baseline_2], axis=0).reset_index(drop=True)
        
        X_itt_perturbed_1, _ = perturbation_dynamics(time_itt_1, 1.0, p=p_perturbed)
        x0_itt_perturbed = steady_state(1, p_perturbed)
        x0_itt_perturbed[-2] = I0 * 60  # Set insulin to 60× basal
        X_itt_perturbed_2, _ = perturbation_dynamics(time_itt_2, 1.0, X0=x0_itt_perturbed, p=p_perturbed)
        X_itt_perturbed = pd.concat([X_itt_perturbed_1, X_itt_perturbed_2], axis=0).reset_index(drop=True)
        
        # Scale concentrations to mouse levels
        for X in [X_gtt_baseline, X_gtt_perturbed, X_itt_baseline, X_itt_perturbed]:
            X['G'] = X['G'] * 6  # 6 mM glucose (mouse scale)
        
        # Create plots
        plots = []
        
        # GTT plot - Glucose response
        def plot_gtt(ax):
            ax.plot(X_gtt_baseline['time'], X_gtt_baseline['G'], linewidth=2, 
                   color='#64748b', label='Baseline')  # slate-500
            if parameter:
                ax.plot(X_gtt_perturbed['time'], X_gtt_perturbed['G'], linewidth=2, 
                       color='#f97316', label='Perturbed', linestyle='--')  # orange-500
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Glucose (mM)')
            ax.set_title('Glucose Tolerance Test (GTT)', pad=30)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
            ax.set_ylim(bottom=0)
            sns.despine(ax=ax)
        
        plot_data = create_individual_plot(plot_gtt)
        plots.append({
            'id': 'gtt_response',
            'title': 'Glucose Tolerance Test',
            'data': plot_data
        })
        
        # ITT plot - Glucose response
        def plot_itt(ax):
            ax.plot(X_itt_baseline['time'], X_itt_baseline['G'], linewidth=2, 
                   color='#64748b', label='Baseline')  # slate-500
            if parameter:
                ax.plot(X_itt_perturbed['time'], X_itt_perturbed['G'], linewidth=2, 
                       color='#f97316', label='Perturbed', linestyle='--')  # orange-500
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Glucose (mM)')
            ax.set_title('Insulin Tolerance Test (ITT)', pad=30)
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
            ax.set_ylim(bottom=0)   
            sns.despine(ax=ax)
        
        plot_data = create_individual_plot(plot_itt)
        plots.append({
            'id': 'itt_response',
            'title': 'Insulin Tolerance Test',
            'data': plot_data
        })
        
        # Prepare data for download
        X_gtt_baseline['test'] = 'GTT'
        X_gtt_baseline['condition'] = 'Baseline'
        X_gtt_perturbed['test'] = 'GTT'
        X_gtt_perturbed['condition'] = 'Perturbed'
        X_itt_baseline['test'] = 'ITT'
        X_itt_baseline['condition'] = 'Baseline'
        X_itt_perturbed['test'] = 'ITT'
        X_itt_perturbed['condition'] = 'Perturbed'
        
        X_all = pd.concat([X_gtt_baseline, X_gtt_perturbed, X_itt_baseline, X_itt_perturbed])
        
        # Rename columns with full names and units
        X_all_download = X_all.copy()
        X_all_download = X_all_download.rename(columns={
            'time': 'Time (min)',
            'G': 'Glucose (mM)',
            'test': 'Test_type',
            'condition': 'Condition'
        })
        
        # Add perturbation information
        X_all_download['Perturbation_parameter'] = parameter if parameter else 'None'
        X_all_download['Perturbation_fold_change'] = fold_change if parameter else 1.0
        
        # Add additional parameter perturbations if any
        if param_dict:
            additional_params_str = '; '.join([f"{k}={v}x" for k, v in param_dict.items()])
            X_all_download['Additional_perturbations'] = additional_params_str
        else:
            X_all_download['Additional_perturbations'] = 'None'
        
        data_csv = X_all_download.to_csv(index=False)
        
        # Check for warnings
        warnings = []
        warnings.extend(check_simulation_warnings(X_gtt_baseline[['time', 'G']]))
        warnings.extend(check_simulation_warnings(X_gtt_perturbed[['time', 'G']]))
        warnings.extend(check_simulation_warnings(X_itt_baseline[['time', 'G']]))
        warnings.extend(check_simulation_warnings(X_itt_perturbed[['time', 'G']]))
        warnings = list(set(warnings))
        
        return jsonify({
            'success': True,
            'plots': plots,
            'data': data_csv,
            'warnings': warnings
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/run_obesity', methods=['POST'])
def run_obesity():
    """Run obesity simulation with mouse or human data overlay"""
    try:
        data = request.json
        
        # Get parameters
        species = data.get('species', 'mouse')  # 'mouse' or 'human'
        sex = data.get('sex', 'male')  # 'male' or 'female' (for human only)
        show_data = data.get('show_data', True)
        perturbation_param = data.get('perturbation_param', None)
        perturbation_value = float(data.get('perturbation_value', 1.0))
        
        # Additional parameter perturbations (as fold changes)
        param_dict = data.get('parameters', {})
        
        # Define color scheme for all plots
        PLOT_COLORS = {
            'model': '#0891b2',        # cyan-600 - main model line
            'perturbed': '#f97316',    # orange-500 - perturbed model line
            'mouse_data': '#64748b',   # slate-500 - BXD mouse data points
            'male_data': '#06b6d4',    # cyan-500 - male human data points (light teal)
            'female_data': '#fb7185'   # rose-400 - female human data points (light pink)
        }
        
        # Get base parameters
        p = ref_parameters()
        
        # Apply parameter changes as fold changes
        if param_dict:
            keys = list(param_dict.keys())
            fold_changes = [float(param_dict[k]) for k in keys]
            values = [fold_changes[i] * p[PARAMETER_NAMES.index(keys[i])] for i in range(len(keys))]
            p = change_parameters(p, values, ix=keys)
        
        # Run simulations
        if species == 'mouse':
            # Mouse: fat mass in grams (3.5g baseline for C57BL/6J)
            # Adiposity range 1 to 10 (fold change)
            A = np.linspace(1, 10, 100)
            fat_mass_g = A * 3.5  # Convert to grams
            
            results = [perturbation_steady_state(a, p=p) for a in A]
            G = np.array([r[1] for r in results]) * 90  # Scale to mg/dL
            I = np.array([r[-1] for r in results]) / I0 * 0.4  # Scale to ng/mL
            HOMA_IR = (G * I * 6) / 405  # Convert insulin to uU/mL (* 6)
            
            x_axis = fat_mass_g
            x_label = 'Fat mass (g)'
            
        else:  # human
            # Human: body fat percentage
            A = np.linspace(1.0, 3, 100)
            
            results = [perturbation_steady_state(a, p=p) for a in A]
            G = np.array([r[1] for r in results]) * 85  # Scale to mg/dL
            I = np.array([r[-1] for r in results]) * 5.0 / I0  # Scale to uU/mL
            HOMA_IR = (G * I) / 405
            
            # Convert adiposity to body fat percentage (sex-specific)
            if sex == 'male':
                HFP = 0.18  # Healthy fat percentage for men
                L = (1.0 - HFP) / HFP
                body_fat_pct = A / (A + L) * 100
            else:  # female
                HFP = 0.30  # Healthy fat percentage for women
                L = (1.0 - HFP) / HFP
                body_fat_pct = A / (A + L) * 100
            
            x_axis = body_fat_pct
            x_label = 'Body fat percentage (%)'
        
        # Run perturbed simulation if specified
        G_perturbed = None
        I_perturbed = None
        HOMA_IR_perturbed = None
        
        if perturbation_param:
            p_perturbed = change_parameters(p, [perturbation_value * p[PARAMETER_NAMES.index(perturbation_param)]], 
                                           ix=[perturbation_param])
            
            if species == 'mouse':
                A_perturbed = np.linspace(1, 10, 100)
                results_perturbed = [perturbation_steady_state(a, p=p_perturbed) for a in A_perturbed]
                G_perturbed = np.array([r[1] for r in results_perturbed]) * 90
                I_perturbed = np.array([r[-1] for r in results_perturbed]) / I0 * 0.4
                HOMA_IR_perturbed = (G_perturbed * I_perturbed * 6) / 405
            else:  # human
                A_perturbed = np.linspace(1.0, 3, 100)
                results_perturbed = [perturbation_steady_state(a, p=p_perturbed) for a in A_perturbed]
                G_perturbed = np.array([r[1] for r in results_perturbed]) * 85
                I_perturbed = np.array([r[-1] for r in results_perturbed]) * 5.0 / I0
                HOMA_IR_perturbed = (G_perturbed * I_perturbed) / 405
        
        # Load experimental data
        experimental_data = None
        if show_data:
            if species == 'mouse':
                # Load BXD mouse data
                mouse_file = os.path.join(os.path.dirname(__file__), '..', 'multi_nutrient_model', 
                                         'data', 'BXD_metabolic_traits.tsv')
                if os.path.exists(mouse_file):
                    df_mouse = pd.read_table(mouse_file, index_col=0, sep='\t')
                    # Process data as in notebook
                    df_mouse = df_mouse.drop(columns=[c for c in df_mouse.columns if c.endswith('_SE')])
                    df_mouse = df_mouse.replace('x', np.nan)
                    data_columns = df_mouse.columns[df_mouse.columns.get_loc('C57BL/6J'):]
                    df_mouse[data_columns] = df_mouse[data_columns].astype(float)
                    data_averaged = df_mouse.groupby(['Trait','Diet'])[data_columns].median().reset_index()
                    data_melt = data_averaged.melt(id_vars=['Trait','Diet'], var_name='Strain', value_name='Value')
                    data_pivot = data_melt.pivot_table(index=['Strain','Diet'], columns='Trait', values='Value').reset_index()
                    
                    experimental_data = {
                        'fat_mass': data_pivot['Fat mass [g]'].values,
                        'glucose': data_pivot['Glucose [mg/dl]'].values,
                        'insulin': data_pivot['Insulin [ng/ml]'].values,
                        'homa_ir': (data_pivot['Insulin [ng/ml]'] * 6 * data_pivot['Glucose [mg/dl]'] / 405).values
                    }
            else:  # human
                # Load NHANES data
                nhanes_file = os.path.join(os.path.dirname(__file__), '..', 'multi_nutrient_model', 
                                          'data', 'NHANES Demo Anthro Glc Ins.csv')
                if os.path.exists(nhanes_file):
                    df_nhanes = pd.read_csv(nhanes_file, index_col=0)
                    # Process as in notebook
                    df_nhanes.rename(columns={
                        'lbxglu': 'Glucose (mg/dL)',
                        'lbxin': 'Insulin (uU/mL)',
                        'bmxwaist': 'Waist circumference (cm)',
                        'bmxht': 'Height (cm)',
                        'bmxwt': 'Weight (kg)',
                        'bmxbmi': 'BMI (kg/m²)',
                        'riagendr': 'Gender',
                        'ridageyr': 'Age (years)',
                        'diq050': 'Taking insulin'
                    }, inplace=True)
                    
                    # Filter and process
                    df_nhanes = df_nhanes.dropna(subset=['Gender', 'Waist circumference (cm)', 'Height (cm)', 
                                                          'Glucose (mg/dL)', 'Insulin (uU/mL)'])
                    df_nhanes = df_nhanes[(df_nhanes['Glucose (mg/dL)'] > 40)]
                    df_nhanes = df_nhanes[(df_nhanes['Insulin (uU/mL)'] > 1)]
                    df_nhanes = df_nhanes[(df_nhanes['Age (years)'] >= 20) & (df_nhanes['Age (years)'] <= 60)]
                    df_nhanes = df_nhanes[df_nhanes['Taking insulin'] == 2]  # 2 = No
                    df_nhanes = df_nhanes[df_nhanes['phafsthr'] > 6]  # Fasting > 6 hours
                    
                    # Calculate body fat percentage
                    def body_fat_pct_waist(height, waist, gender):
                        if gender == 1:  # Male
                            return 64 - (20 * (height / waist))
                        else:  # Female
                            return 76 - (20 * (height / waist))
                    
                    df_nhanes['body_fat_percentage'] = df_nhanes.apply(
                        lambda row: body_fat_pct_waist(row['Height (cm)'], row['Waist circumference (cm)'], 
                                                       row['Gender']), axis=1
                    )
                    df_nhanes['HOMA-IR'] = (df_nhanes['Insulin (uU/mL)'] * df_nhanes['Glucose (mg/dL)']) / 405
                    
                    # Filter by sex
                    sex_code = 1 if sex == 'male' else 2
                    df_sex = df_nhanes[df_nhanes['Gender'] == sex_code]
                    
                    experimental_data = {
                        'body_fat_pct': df_sex['body_fat_percentage'].values,
                        'glucose': df_sex['Glucose (mg/dL)'].values,
                        'insulin': df_sex['Insulin (uU/mL)'].values,
                        'homa_ir': df_sex['HOMA-IR'].values
                    }
        
        # Create plots
        plots = []
        
        # Glucose plot
        def plot_glucose(ax):
            ax.plot(x_axis, G, linewidth=2, color=PLOT_COLORS['model'], label='Model')
            if G_perturbed is not None:
                ax.plot(x_axis, G_perturbed, linewidth=2, color=PLOT_COLORS['perturbed'], 
                       linestyle='--', label='Perturbed')
            
            if experimental_data and show_data:
                if species == 'mouse':
                    ax.scatter(experimental_data['fat_mass'], experimental_data['glucose'],
                              alpha=0.3, s=20, color=PLOT_COLORS['mouse_data'], label='BXD mice')
                else:
                    data_color = PLOT_COLORS['male_data'] if sex == 'male' else PLOT_COLORS['female_data']
                    ax.scatter(experimental_data['body_fat_pct'], experimental_data['glucose'],
                              alpha=0.3, s=10, color=data_color,
                              label=f'{sex.capitalize()} (NHANES)')
            
            ax.set_xlabel(x_label)
            ax.set_ylabel('Fasting glucose (mg/dL)')
            ax.set_title('Glucose vs Adiposity', pad=30)
            if show_data or G_perturbed is not None:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
            ax.set_ylim(bottom=0)
            sns.despine(ax=ax)
        
        plot_data = create_individual_plot(plot_glucose)
        plots.append({
            'id': 'glucose_adiposity',
            'title': 'Glucose vs Adiposity',
            'data': plot_data
        })
        
        # Insulin plot
        def plot_insulin(ax):
            ax.plot(x_axis, I, linewidth=2, color=PLOT_COLORS['model'], label='Model')
            if I_perturbed is not None:
                ax.plot(x_axis, I_perturbed, linewidth=2, color=PLOT_COLORS['perturbed'], 
                       linestyle='--', label='Perturbed')
            
            if experimental_data and show_data:
                if species == 'mouse':
                    ax.scatter(experimental_data['fat_mass'], experimental_data['insulin'],
                              alpha=0.3, s=20, color=PLOT_COLORS['mouse_data'], label='BXD mice')
                else:
                    data_color = PLOT_COLORS['male_data'] if sex == 'male' else PLOT_COLORS['female_data']
                    ax.scatter(experimental_data['body_fat_pct'], experimental_data['insulin'],
                              alpha=0.3, s=10, color=data_color,
                              label=f'{sex.capitalize()} (NHANES)')
            
            ax.set_xlabel(x_label)
            ylabel = 'Fasting insulin (ng/mL)' if species == 'mouse' else 'Fasting insulin (uU/mL)'
            ax.set_ylabel(ylabel)
            ax.set_title('Insulin vs Adiposity', pad=30)
            if show_data or I_perturbed is not None:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
            ax.set_ylim(bottom=0)
            sns.despine(ax=ax)
        
        plot_data = create_individual_plot(plot_insulin)
        plots.append({
            'id': 'insulin_adiposity',
            'title': 'Insulin vs Adiposity',
            'data': plot_data
        })
        
        # HOMA-IR plot
        def plot_homa_ir(ax):
            ax.plot(x_axis, HOMA_IR, linewidth=2, color=PLOT_COLORS['model'], label='Model')
            if HOMA_IR_perturbed is not None:
                ax.plot(x_axis, HOMA_IR_perturbed, linewidth=2, color=PLOT_COLORS['perturbed'], 
                       linestyle='--', label='Perturbed')
            
            if experimental_data and show_data:
                if species == 'mouse':
                    ax.scatter(experimental_data['fat_mass'], experimental_data['homa_ir'],
                              alpha=0.3, s=20, color=PLOT_COLORS['mouse_data'], label='BXD mice')
                else:
                    data_color = PLOT_COLORS['male_data'] if sex == 'male' else PLOT_COLORS['female_data']
                    ax.scatter(experimental_data['body_fat_pct'], experimental_data['homa_ir'],
                              alpha=0.3, s=10, color=data_color,
                              label=f'{sex.capitalize()} (NHANES)')
            
            ax.set_xlabel(x_label)
            ax.set_ylabel('HOMA-IR')
            ax.set_title('HOMA-IR vs Adiposity', pad=30)
            if HOMA_IR_perturbed is not None:
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
            ax.set_ylim(bottom=0)
            sns.despine(ax=ax)
        
        plot_data = create_individual_plot(plot_homa_ir)
        plots.append({
            'id': 'homa_ir_adiposity',
            'title': 'HOMA-IR vs Adiposity',
            'data': plot_data
        })
        
        # Prepare CSV data with better column labels
        x_label_units = 'Fat_mass (g)' if species == 'mouse' else 'Body_fat_percentage (%)'
        
        df_results = pd.DataFrame({
            x_label_units: x_axis,
            'Glucose_baseline (mg/dL)': G,
            'Insulin_baseline (uU/mL)': I,
            'HOMA_IR_baseline': HOMA_IR,
            'Species': species,
            'Sex': sex if species == 'human' else 'N/A'
        })
        
        if G_perturbed is not None:
            df_results['Glucose_perturbed (mg/dL)'] = G_perturbed
            df_results['Insulin_perturbed (uU/mL)'] = I_perturbed
            df_results['HOMA_IR_perturbed'] = HOMA_IR_perturbed
            df_results['Perturbation_parameter'] = perturbation_param
            df_results['Perturbation_fold_change'] = perturbation_value
        else:
            df_results['Perturbation_parameter'] = 'None'
            df_results['Perturbation_fold_change'] = 1.0
        
        # Add additional parameter perturbations if any
        if param_dict:
            additional_params_str = '; '.join([f"{k}={v}x" for k, v in param_dict.items()])
            df_results['Additional_perturbations'] = additional_params_str
        else:
            df_results['Additional_perturbations'] = 'None'
        
        data_csv = df_results.to_csv(index=False)
        
        # Check for warnings
        warnings = []
        warnings.extend(check_simulation_warnings(df_results[['glucose', 'insulin', 'homa_ir']].rename(columns={'glucose': 'G', 'insulin': 'I'})))
        if G_perturbed is not None:
            warnings.extend(check_simulation_warnings(pd.DataFrame({'G': G_perturbed, 'I': I_perturbed})))
        warnings = list(set(warnings))
        
        return jsonify({
            'success': True,
            'plots': plots,
            'data': data_csv,
            'warnings': warnings
        })
        
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 400

@app.route('/api/run_treatment', methods=['POST'])
def run_treatment():
    """Run treatment simulation with sensitivity analysis"""
    try:
        data = request.json
        
        # Get parameters
        adiposity = float(data.get('adiposity', 3.0))  # Default adiposity = 3
        disease_param = data.get('disease_param', None)  # Optional disease parameter
        disease_fold = float(data.get('disease_fold', 2.0)) if disease_param else None
        insulin_dose_u_kg_day = float(data.get('insulin_dose', 0.2))  # Insulin dose in U/kg/day (default 0.2)
        ranking_variable = data.get('ranking_variable', 'HOMA_IR')  # Variable to rank by
        treatment_params = json.loads(data.get('treatment_params', '[]'))
        treatment_folds = json.loads(data.get('treatment_folds', '[]'))
        
        # Display options
        show_glucose = data.get('show_glucose', True)
        show_insulin = data.get('show_insulin', True)
        show_homa_ir = data.get('show_homa_ir', True)
        show_fatty_acids = data.get('show_fatty_acids', False)
        show_ketones = data.get('show_ketones', False)
        show_lactate = data.get('show_lactate', False)
        
        # Additional parameter perturbations (as fold changes)
        param_dict = data.get('parameters', {})
        
        # Define color scheme for treatment plots (matching app theme)
        TREATMENT_COLORS = {
            'diseased': '#64748b',     # slate-500 - grey for diseased state
            'insulin': '#8b5cf6',      # violet-500 - muted purple for insulin
            'increase': '#f97316',     # orange-500 - orange for increased parameters (matching theme)
            'decrease': '#0891b2',     # cyan-600 - teal for decreased parameters (matching theme)
            'baseline': '#64748b'      # slate-500 - grey baseline
        }
        
        # Get base parameters
        p = ref_parameters()
        
        # Apply parameter changes as fold changes
        if param_dict:
            keys = list(param_dict.keys())
            fold_changes = [float(param_dict[k]) for k in keys]
            values = [fold_changes[i] * p[PARAMETER_NAMES.index(keys[i])] for i in range(len(keys))]
            p = change_parameters(p, values, ix=keys)
        
        # Apply disease parameter perturbation if specified
        if disease_param:
            p = change_parameters(p, [disease_fold * p[PARAMETER_NAMES.index(disease_param)]], 
                                 ix=[disease_param])
        
        # Get diseased state (adiposity + optional parameter perturbation)
        X_diseased = perturbation_steady_state(adiposity, p=p)
        
        # Scale to human values (matching obesity tab)
        G_diseased = X_diseased[1] * 85  # mg/dL
        I_diseased = X_diseased[4] * 5.0 / I0  # uU/mL
        HOMA_IR_diseased = (G_diseased * I_diseased) / 405
        F_diseased = X_diseased[2] * 0.5  # mM (fatty acids)
        K_diseased = X_diseased[3] * 0.5  # mM (ketones/3HB)
        L_diseased = X_diseased[0] * 0.7  # mM (lactate)
        
        # Run sensitivity analysis to find top perturbations
        sensitivities = {}
        for param in PARAMETER_NAMES[:17]:  # Only modifiable parameters
            try:
                S = sensitivity_analysis(param, adiposity, p=p, fold_change=2.0)
                if ranking_variable in S.index:
                    sensitivities[param] = S[ranking_variable]  # Keep sign for direction
            except:
                pass
        
        # Rank by absolute sensitivity of the ranking variable
        ranked_params = sorted(sensitivities.items(), key=lambda x: abs(x[1]), reverse=True)
        top_params = [p[0] for p in ranked_params[:10]]
        
        # If treatment params not provided, use top 3
        if not treatment_params:
            treatment_params = top_params[:3]
        
        # Test treatments
        treatment_results = []
        
        # Add diseased state
        disease_label = f'Diseased (A={adiposity}'
        if disease_param:
            disease_label += f', {disease_param}×{disease_fold}'
        disease_label += ')'
        
        treatment_results.append({
            'treatment': disease_label,
            'G': G_diseased,
            'I': I_diseased,
            'HOMA_IR': HOMA_IR_diseased,
            'F': F_diseased,
            'K': K_diseased,
            'L': L_diseased,
            'color': TREATMENT_COLORS['diseased'],
            'direction': 'baseline'
        })
        
        # Test insulin treatment with user-specified dose
        # Convert insulin dose from U/kg/day to model units (R_insulin parameter)
        # Based on sensitivity_analysis.ipynb scaling:
        # Model: I_dose = I0/TAU_INS (base insulin production rate)
        # Scaling factor: 1.25 mU/min/kg * 60 min/hr * 24 hr/day * 0.081 (mouse-to-human) = U/kg/day
        
        scaling_model_to_U_kg_day = 1.25 / 1000 * 60 * 24 * 0.081  # Model units to U/kg/day
        
        I_dose = I0 / TAU_INS  # Base insulin dose in model units
        
        # Convert user's U/kg/day to model units (R_insulin infusion rate)
        # Formula from notebook: display_value = (C/I_dose) * scaling
        # Reverse: C = display_value * I_dose / scaling
        insulin_infusion_rate = (insulin_dose_u_kg_day / scaling_model_to_U_kg_day) * I_dose
        
        # Compute steady state with insulin infusion
        X_insulin = perturbation_steady_state(adiposity, p=p, R_insulin=insulin_infusion_rate)
        
        # Scale to human values (matching obesity tab scaling)
        G_insulin = X_insulin[1] * 85  # mg/dL (human scaling)
        I_insulin = X_insulin[4] * 5.0 / I0  # uU/mL (human scaling)
        HOMA_IR_insulin = (G_insulin * I_insulin) / 405
        F_insulin = X_insulin[2] * 0.5  # mM
        K_insulin = X_insulin[3] * 0.5  # mM
        L_insulin = X_insulin[0] * 0.7  # mM
        
        # Add insulin treatment result
        treatment_results.append({
            'treatment': f'Insulin ({insulin_dose_u_kg_day:.2f} U/kg/day)',
            'G': G_insulin,
            'I': I_insulin,
            'HOMA_IR': HOMA_IR_insulin,
            'F': F_insulin,
            'K': K_insulin,
            'L': L_insulin,
            'fold_change': None,
            'sensitivity': None,
            'direction': 'insulin',
            'color': TREATMENT_COLORS['insulin']
        })
        
        # Test each treatment with 2-fold perturbation in HOMA-IR reducing direction
        for treat_param in treatment_params:
            # Get sensitivity direction
            sensitivity = sensitivities.get(treat_param, 0)
            
            # Determine fold change: if sensitivity is positive (increases HOMA-IR), 
            # we want to decrease the parameter (0.5x), and vice versa
            if sensitivity > 0:
                treat_fold = 0.5  # Decrease parameter
                direction = 'decrease'
                color = TREATMENT_COLORS['decrease']
            else:
                treat_fold = 2.0  # Increase parameter
                direction = 'increase'
                color = TREATMENT_COLORS['increase']
            
            p_treated = change_parameters(p, 
                                         [treat_fold * p[PARAMETER_NAMES.index(treat_param)]], 
                                         ix=[treat_param])
            X_treated = perturbation_steady_state(adiposity, p=p_treated)
            
            # Scale to human values
            G_treated = X_treated[1] * 85  # mg/dL
            I_treated = X_treated[4] * 5.0 / I0  # uU/mL
            HOMA_IR_treated = (G_treated * I_treated) / 405
            F_treated = X_treated[2] * 0.5  # mM
            K_treated = X_treated[3] * 0.5  # mM
            L_treated = X_treated[0] * 0.7  # mM
            
            treatment_results.append({
                'treatment': treat_param,
                'G': G_treated,
                'I': I_treated,
                'HOMA_IR': HOMA_IR_treated,
                'F': F_treated,
                'K': K_treated,
                'L': L_treated,
                'fold_change': treat_fold,
                'sensitivity': float(sensitivity),
                'direction': direction,
                'color': color
            })
        
        df_treatment = pd.DataFrame(treatment_results)
        
        # Create individual plots
        plots = []
        
        # Build list of variables to display based on user selection
        variables = []
        labels = []
        
        if show_glucose:
            variables.append('G')
            labels.append('Glucose (mg/dL)')
        
        if show_insulin:
            variables.append('I')
            labels.append('Insulin (uU/mL)')
        
        if show_homa_ir:
            variables.append('HOMA_IR')
            labels.append('HOMA-IR')
        
        if show_fatty_acids:
            variables.append('F')
            labels.append('Fatty Acids (mM)')
        
        if show_ketones:
            variables.append('K')
            labels.append('Ketones (mM)')
        
        if show_lactate:
            variables.append('L')
            labels.append('Lactate (mM)')
        
        def plot_treatment_bar(ax, var, label):
            colors = [r['color'] for r in treatment_results]
            ax.bar(range(len(df_treatment)), df_treatment[var], color=colors)
            ax.axhline(y=treatment_results[0][var], color=TREATMENT_COLORS['diseased'], 
                         linestyle='--', alpha=0.5, label='Diseased')
            ax.set_xticks(range(len(df_treatment)))
            ax.set_xticklabels(df_treatment['treatment'], rotation=45, ha='right')
            ax.set_ylabel(label)
            ax.set_title(f'{label} - Treatment Effects', pad=45)
            
            # Add legend for colors
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=TREATMENT_COLORS['diseased'], label='Diseased'),
                Patch(facecolor=TREATMENT_COLORS['insulin'], label='Insulin'),
                Patch(facecolor=TREATMENT_COLORS['increase'], label='Increased 2×'),
                Patch(facecolor=TREATMENT_COLORS['decrease'], label='Decreased 0.5×')
            ]
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.20), ncol=2, frameon=False)
            sns.despine(ax=ax)
        
        for var, label in zip(variables, labels):
            plot_data = create_individual_plot(plot_treatment_bar, var=var, label=label)
            plots.append({
                'id': f'{var}_treatment',
                'title': f'{label} - Treatment Effects',
                'data': plot_data
            })
        
        # Prepare data with better column labels
        df_treatment_download = df_treatment.copy()
        df_treatment_download = df_treatment_download.rename(columns={
            'treatment': 'Treatment_parameter',
            'G': 'Glucose (mg/dL)',
            'I': 'Insulin (uU/mL)',
            'HOMA_IR': 'HOMA_IR',
            'F': 'Fatty_acids (mM)',
            'K': '3-Hydroxybutyrate (mM)',
            'L': 'Lactate (mM)',
            'fold_change': 'Treatment_fold_change',
            'sensitivity': 'Parameter_sensitivity',
            'direction': 'Treatment_direction',
            'color': 'Plot_color'
        })
        
        # Add disease state information
        df_treatment_download['Disease_adiposity_fold'] = adiposity
        df_treatment_download['Disease_parameter'] = disease_param if disease_param else 'None'
        df_treatment_download['Disease_parameter_fold'] = disease_fold if disease_param else 1.0
        df_treatment_download['Insulin_dose (U/kg/day)'] = insulin_dose_u_kg_day
        df_treatment_download['Ranking_variable'] = ranking_variable
        
        # Add additional parameter perturbations if any
        if param_dict:
            additional_params_str = '; '.join([f"{k}={v}x" for k, v in param_dict.items()])
            df_treatment_download['Additional_perturbations'] = additional_params_str
        else:
            df_treatment_download['Additional_perturbations'] = 'None'
        
        data_csv = df_treatment_download.to_csv(index=False)
        
        # Add top perturbations info with direction and description
        top_perturbations = [
            {
                'parameter': p, 
                'description': PARAMETER_DESCRIPTIONS.get(p, p),
                'sensitivity': float(s),
                'abs_sensitivity': float(abs(s)),
                'direction': 'increase' if s < 0 else 'decrease'  # Opposite of sensitivity
            } 
            for p, s in ranked_params[:10]
        ]
        
        # Check for warnings on treatment results
        warnings = []
        if len(df_treatment) > 0:
            warnings.extend(check_simulation_warnings(df_treatment[['G', 'I', 'HOMA_IR', 'F', 'K', 'L']]))
        warnings = list(set(warnings))
        
        return jsonify({
            'success': True,
            'plots': plots,
            'data': data_csv,
            'top_perturbations': top_perturbations,
            'ranking_variable': ranking_variable,
            'adiposity': adiposity,
            'insulin_dose': insulin_dose_u_kg_day,
            'warnings': warnings
        })
        
    except Exception as e:
        import traceback
        return jsonify({'success': False, 'error': str(e), 'traceback': traceback.format_exc()}), 400
        
        return jsonify({
            'success': True,
            'plots': plots,
            'data': data_csv,
            'top_perturbations': top_perturbations
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)