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
    """Convert matplotlib figure to SVG string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    buf.seek(0)
    svg_str = buf.read().decode('utf-8')
    plt.close(fig)
    return svg_str

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

@app.route('/dynamic_response')
def dynamic_response():
    """Page for Type 1 Diabetes-like dynamic response simulation"""
    return render_template('dynamic_response.html', 
                         parameter_names=PARAMETER_NAMES,
                         parameter_descriptions=PARAMETER_DESCRIPTIONS,
                         parameter_latex=PARAMETER_LATEX)

@app.route('/insulin_clamp')
def insulin_clamp():
    """Page for hyperinsulinemic-euglycemic clamp simulation"""
    return render_template('insulin_clamp.html',
                         parameter_names=PARAMETER_NAMES,
                         parameter_descriptions=PARAMETER_DESCRIPTIONS,
                         parameter_latex=PARAMETER_LATEX)

@app.route('/tolerance_tests')
def tolerance_tests():
    """Page for GTT/ITT simulation with receptor knockouts"""
    return render_template('tolerance_tests.html',
                         parameter_names=PARAMETER_NAMES,
                         parameter_descriptions=PARAMETER_DESCRIPTIONS,
                         parameter_latex=PARAMETER_LATEX)

@app.route('/obesity')
def obesity():
    """Page for obesity simulation"""
    return render_template('obesity.html',
                         parameter_names=PARAMETER_NAMES,
                         parameter_descriptions=PARAMETER_DESCRIPTIONS,
                         parameter_latex=PARAMETER_LATEX)

@app.route('/treatment')
def treatment():
    """Page for treatment simulation"""
    return render_template('treatment.html',
                         parameter_names=PARAMETER_NAMES,
                         parameter_descriptions=PARAMETER_DESCRIPTIONS,
                         parameter_latex=PARAMETER_LATEX)

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
        
        # Additional parameter perturbations
        param_dict = data.get('parameters', {})
        
        # Get base parameters
        p = ref_parameters()
        
        # Apply parameter changes
        if param_dict:
            keys = list(param_dict.keys())
            values = [float(param_dict[k]) for k in keys]
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
        X['G'] = X['G'] * 6  # 7 mM glucose
        X['F'] = X['F'] * 0.5  # 0.5 mM FFA
        X['K'] = X['K'] * 0.5  # 0.5 mM 3HB
        X['L'] = X['L'] * 0.7  # 0.7 mM lactate

        # Scale insulin concentration
        X['I'] = X['I'] / I0  * 5 # Scale to typical insulin levels Insulin in humnas
        
        # Create individual plots
        variables = ['G', 'F', 'K', 'I']
        labels = ['Glucose (mM)', 'Fatty acids (mM)', '3-Hydroxybutyrate (mM)', 'Insulin (a.u.)']
        
        def plot_timeseries(ax, var, label):
            ax.plot(X['time'], X[var], linewidth=2, color='black')
            ax.axvline(time_perturbation, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Time (min)')
            ax.set_ylabel(label)
            ax.set_title(f'{label.split(" ")[0]} Response')
            sns.despine(ax=ax)
        
        plots = []
        for var, label in zip(variables, labels):
            plot_data = create_individual_plot(plot_timeseries, var=var, label=label)
            plots.append({
                'id': var,
                'title': f'{label.split(" ")[0]} Response',
                'data': plot_data
            })
        
        # Prepare data for download
        data_csv = X[['time', 'G', 'F', 'K', 'L', 'I', 'IA']].to_csv(index=False)
        
        return jsonify({
            'success': True,
            'plots': plots,
            'data': data_csv
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
        
        # Additional parameter perturbations
        param_dict = data.get('parameters', {})
        
        # Get base parameters
        p = ref_parameters()
        
        # Apply parameter changes
        if param_dict:
            keys = list(param_dict.keys())
            values = [float(param_dict[k]) for k in keys]
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
        X_saline['G'] = X_saline['G'] * 7
        X_saline['F'] = X_saline['F'] * 0.5
        X_saline['K'] = X_saline['K'] * 0.5
        X_saline['L'] = X_saline['L'] * 0.7
        X_saline['condition'] = 'Saline'
        GIR_saline['condition'] = 'Saline'
        
        # Baseline (insulin clamp, no infusion)
        X_baseline, GIR_baseline = insulin_clamp_dynamic(insulin_dose, time, 1.0, p=p)
        X_baseline['G'] = X_baseline['G'] * 7
        X_baseline['F'] = X_baseline['F'] * 0.5
        X_baseline['K'] = X_baseline['K'] * 0.5
        X_baseline['L'] = X_baseline['L'] * 0.7
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
            X_infusion['G'] = X_infusion['G'] * 7
            X_infusion['F'] = X_infusion['F'] * 0.5
            X_infusion['K'] = X_infusion['K'] * 0.5
            X_infusion['L'] = X_infusion['L'] * 0.7
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
        
        # Define color palette
        condition_colors = {
            'Saline': '#8E8E8E',
            'Insulin': '#C959C5',
        }
        if GIR_infusion is not None:
            condition_colors[GIR_infusion['condition'].iloc[0]] = '#4FC452'
        
        # GIR time course using seaborn
        def plot_gir_timecourse(ax):
            sns.lineplot(data=GIR_combined, x='time', y='GIR', hue='condition',
                        palette=condition_colors, linewidth=2, ax=ax)
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('GIR (mg/kg/min)')
            ax.set_title(f'Glucose Infusion Rate - {insulin_label}')
            ax.legend(title='')
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
        
        # Update color palette for infusion condition if present
        if X_infusion is not None:
            condition_colors[X_infusion['condition'].iloc[0]] = '#4FC452'
        
        # Create individual bar plot for each metabolite
        for metabolite_name in metabolite_map.values():
            metabolite_df = ss_df[ss_df['Metabolite'] == metabolite_name]
            
            def plot_metabolite(ax, met_df=metabolite_df, met_name=metabolite_name):
                sns.barplot(data=met_df, x='Condition', y='Concentration',
                           palette=condition_colors, alpha=0.7, ax=ax)
                ax.set_ylabel('Concentration (mM)')
                ax.set_xlabel('')
                ax.set_title(f'{met_name} - {insulin_label}')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                sns.despine(ax=ax)
            
            plot_data = create_individual_plot(plot_metabolite)
            plots.append({
                'id': f'steady_state_{metabolite_name.lower().replace(" ", "_")}',
                'title': f'{metabolite_name} Steady State',
                'data': plot_data
            })
        
        # Prepare data for download
        if X_infusion is not None:
            X_all = pd.concat([X_saline, X_baseline, X_infusion])
        else:
            X_all = pd.concat([X_saline, X_baseline])
        data_csv = X_all.to_csv(index=False)
        
        return jsonify({
            'success': True,
            'plots': plots,
            'data': data_csv
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/run_tolerance_tests', methods=['POST'])
def run_tolerance_tests():
    """Run GTT/ITT simulation"""
    try:
        data = request.json
        
        test_type = data.get('test_type', 'GTT')
        knockout = data.get('knockout', 'with_SI')  # Adipose insulin receptor K/O
        time_max = float(data.get('time_max', 120))
        bolus_time = float(data.get('bolus_time', 10))
        
        # Additional parameter perturbations
        param_dict = data.get('parameters', {})
        
        # Get base parameters
        p_control = ref_parameters()
        
        # Apply parameter changes to control
        if param_dict:
            keys = list(param_dict.keys())
            values = [float(param_dict[k]) for k in keys]
            p_control = change_parameters(p_control, values, ix=keys)
        
        # Create knockout parameters
        p_ko = change_parameters(p_control, [False], ix=[knockout])
        
        # Define simulation
        time = np.linspace(0, time_max, 200)
        
        if test_type == 'GTT':
            # Glucose bolus
            glucose_bolus = 2.0  # 2x normal glucose
            X_control, _ = perturbation_dynamics(time, 1.0, p=p_control, v_in_G=glucose_bolus)
            X_ko, _ = perturbation_dynamics(time, 1.0, p=p_ko, v_in_G=glucose_bolus)
        else:  # ITT
            # Insulin bolus
            insulin_bolus = 5.0  # 5x normal insulin
            X_control, _ = perturbation_dynamics(time, 1.0, p=p_control, v_in_I=insulin_bolus)
            X_ko, _ = perturbation_dynamics(time, 1.0, p=p_ko, v_in_I=insulin_bolus)
        
        # Scale concentrations
        for X in [X_control, X_ko]:
            X['G'] = X['G'] * 7
            X['I'] = X['I'] * 1.0
        
        # Create individual plots
        plots = []
        
        # Plot glucose response
        def plot_glucose_response(ax):
            ax.plot(X_control['time'], X_control['G'], linewidth=2, 
                       color='grey', label='Control')
            ax.plot(X_ko['time'], X_ko['G'], linewidth=2, 
                       color='steelblue', label='K/O', linestyle='--')
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Glucose (mM)')
            ax.set_title(f'{test_type} - Glucose Response')
            ax.legend()
            sns.despine(ax=ax)
        
        plot_data = create_individual_plot(plot_glucose_response)
        plots.append({
            'id': 'glucose_response',
            'title': f'{test_type} - Glucose Response',
            'data': plot_data
        })
        
        # Plot insulin response
        def plot_insulin_response(ax):
            ax.plot(X_control['time'], X_control['I'], linewidth=2, 
                       color='grey', label='Control')
            ax.plot(X_ko['time'], X_ko['I'], linewidth=2, 
                       color='steelblue', label='K/O', linestyle='--')
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Insulin (a.u.)')
            ax.set_title(f'{test_type} - Insulin Response')
            ax.legend()
            sns.despine(ax=ax)
        
        plot_data = create_individual_plot(plot_insulin_response)
        plots.append({
            'id': 'insulin_response',
            'title': f'{test_type} - Insulin Response',
            'data': plot_data
        })
        
        # Prepare data
        X_control['group'] = 'Control'
        X_ko['group'] = 'K/O'
        data_csv = pd.concat([X_control, X_ko]).to_csv(index=False)
        
        return jsonify({
            'success': True,
            'plots': plots,
            'data': data_csv
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/run_obesity', methods=['POST'])
def run_obesity():
    """Run obesity simulation"""
    try:
        data = request.json
        
        # Get parameters
        fat_fractions_str = data.get('fat_fractions', 
            '1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0')
        # Parse comma-separated string into list of floats
        fat_fractions = [float(x.strip()) for x in fat_fractions_str.split(',')]
        perturbation_param = data.get('perturbation_param', None)
        perturbation_value = float(data.get('perturbation_value', 1.0))
        
        # Additional parameter perturbations
        param_dict = data.get('parameters', {})
        
        # Get base parameters
        p = ref_parameters()
        
        # Apply parameter changes
        if param_dict:
            keys = list(param_dict.keys())
            values = [float(param_dict[k]) for k in keys]
            p = change_parameters(p, values, ix=keys)
        
        # Run simulations for different fat fractions
        results = []
        
        for A in fat_fractions:
            X = perturbation_steady_state(A, p=p)
            
            # Scale concentrations
            G = X[1] * 7  # glucose
            I = X[4]  # insulin
            HOMA_IR = G * I
            
            results.append({
                'fat_fraction': A,
                'glucose': G,
                'insulin': I,
                'HOMA_IR': HOMA_IR
            })
        
        df_control = pd.DataFrame(results)
        
        # If perturbation specified, run perturbed simulation
        df_perturbed = None
        if perturbation_param:
            p_perturbed = change_parameters(p, [perturbation_value], ix=[perturbation_param])
            
            results_perturbed = []
            for A in fat_fractions:
                X = perturbation_steady_state(A, p=p_perturbed)
                
                G = X[1] * 7
                I = X[4]
                HOMA_IR = G * I
                
                results_perturbed.append({
                    'fat_fraction': A,
                    'glucose': G,
                    'insulin': I,
                    'HOMA_IR': HOMA_IR
                })
            
            df_perturbed = pd.DataFrame(results_perturbed)
        
        # Load NHANES data for comparison
        nhanes_file = os.path.join(os.path.dirname(__file__), '..', 'multi_nutrient_model', 
                                   'data', 'NHANES Demo Anthro Glc Ins.csv')
        
        df_nhanes = None
        if os.path.exists(nhanes_file):
            df_nhanes = pd.read_csv(nhanes_file)
        
        # Create individual plots
        plots = []
        variables = ['glucose', 'insulin', 'HOMA_IR']
        labels = ['Glucose (mM)', 'Insulin (a.u.)', 'HOMA-IR']
        
        def plot_adiposity(ax, var, label):
            # Plot model results
            ax.plot(df_control['fat_fraction'], df_control[var], 
                       linewidth=2, color='black', label='Model')
            
            if df_perturbed is not None:
                ax.plot(df_perturbed['fat_fraction'], df_perturbed[var], 
                          linewidth=2, color='red', linestyle='--', label='Perturbed')
            
            # Plot NHANES data if available
            if df_nhanes is not None and var in df_nhanes.columns:
                # Assuming fat_fraction or BMI column exists
                if 'fat_fraction' in df_nhanes.columns:
                    # Split by sex if available
                    if 'Sex' in df_nhanes.columns:
                        for sex, marker, color in [('Male', 'o', 'blue'), ('Female', 's', 'pink')]:
                            subset = df_nhanes[df_nhanes['Sex'] == sex]
                            ax.scatter(subset['fat_fraction'], subset[var], 
                                         alpha=0.3, s=10, marker=marker, 
                                         color=color, label=f'{sex} (NHANES)')
            
            ax.set_xlabel('Fat Fraction (relative to control)')
            ax.set_ylabel(label)
            ax.set_title(f'{label} vs Adiposity')
            ax.legend()
            sns.despine(ax=ax)
        
        for var, label in zip(variables, labels):
            plot_data = create_individual_plot(plot_adiposity, var=var, label=label)
            plots.append({
                'id': f'{var}_adiposity',
                'title': f'{label} vs Adiposity',
                'data': plot_data
            })
        
        # Prepare data
        df_control['group'] = 'Control'
        if df_perturbed is not None:
            df_perturbed['group'] = 'Perturbed'
            data_csv = pd.concat([df_control, df_perturbed]).to_csv(index=False)
        else:
            data_csv = df_control.to_csv(index=False)
        
        return jsonify({
            'success': True,
            'plots': plots,
            'data': data_csv
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/run_treatment', methods=['POST'])
def run_treatment():
    """Run treatment simulation with sensitivity analysis"""
    try:
        data = request.json
        
        # Get parameters
        perturbation_param = data.get('perturbation_param', 'alpha')
        perturbation_fold = float(data.get('perturbation_fold', 2.0))
        treatment_params = json.loads(data.get('treatment_params', '[]'))
        treatment_folds = json.loads(data.get('treatment_folds', '[]'))
        
        # Additional parameter perturbations
        param_dict = data.get('parameters', {})
        
        # Get base parameters
        p = ref_parameters()
        
        # Apply parameter changes
        if param_dict:
            keys = list(param_dict.keys())
            values = [float(param_dict[k]) for k in keys]
            p = change_parameters(p, values, ix=keys)
        
        # Create perturbed condition (diseased state)
        p_perturbed = change_parameters(p, [perturbation_fold * p[PARAMETER_NAMES.index(perturbation_param)]], 
                                       ix=[perturbation_param])
        
        # Get baseline states
        A = 1.0  # Reference adiposity
        X_healthy = perturbation_steady_state(A, p=p)
        X_diseased = perturbation_steady_state(A, p=p_perturbed)
        
        # Calculate HOMA-IR for ranking
        HOMA_IR_healthy = X_healthy[1] * X_healthy[4] * 7
        HOMA_IR_diseased = X_diseased[1] * X_diseased[4] * 7
        
        # Run sensitivity analysis to find top perturbations
        sensitivities = {}
        for param in PARAMETER_NAMES[:17]:  # Only modifiable parameters
            try:
                S = sensitivity_analysis(param, A, p=p_perturbed, fold_change=2.0)
                if 'HOMA_IR' in S.index:
                    sensitivities[param] = abs(S['HOMA_IR'])
            except:
                pass
        
        # Rank by HOMA-IR sensitivity
        ranked_params = sorted(sensitivities.items(), key=lambda x: x[1], reverse=True)
        top_3_params = [p[0] for p in ranked_params[:3]]
        
        # If treatment params provided, use those; otherwise use top 3
        if not treatment_params:
            treatment_params = top_3_params
            treatment_folds = [0.5, 0.5, 0.5]  # Default: reduce by 50%
        
        # Test treatments
        treatment_results = []
        
        # Add diseased state
        treatment_results.append({
            'treatment': 'Diseased',
            'G': X_diseased[1] * 7,
            'I': X_diseased[4],
            'HOMA_IR': HOMA_IR_diseased
        })
        
        # Test each treatment
        for treat_param, treat_fold in zip(treatment_params, treatment_folds):
            p_treated = change_parameters(p_perturbed, 
                                         [treat_fold * p_perturbed[PARAMETER_NAMES.index(treat_param)]], 
                                         ix=[treat_param])
            X_treated = perturbation_steady_state(A, p=p_treated)
            
            treatment_results.append({
                'treatment': treat_param,
                'G': X_treated[1] * 7,
                'I': X_treated[4],
                'HOMA_IR': X_treated[1] * X_treated[4] * 7
            })
        
        # Add insulin treatment (increase I_max)
        p_insulin = change_parameters(p_perturbed, 
                                     [1.5 * p_perturbed[PARAMETER_NAMES.index('I_max')]], 
                                     ix=['I_max'])
        X_insulin = perturbation_steady_state(A, p=p_insulin)
        treatment_results.append({
            'treatment': 'Insulin',
            'G': X_insulin[1] * 7,
            'I': X_insulin[4],
            'HOMA_IR': X_insulin[1] * X_insulin[4] * 7
        })
        
        df_treatment = pd.DataFrame(treatment_results)
        
        # Create individual plots
        plots = []
        variables = ['G', 'I', 'HOMA_IR']
        labels = ['Glucose (mM)', 'Insulin (a.u.)', 'HOMA-IR']
        colors = ['steelblue'] * len(treatment_results)
        colors[0] = 'red'  # Diseased state in red
        
        def plot_treatment_bar(ax, var, label):
            ax.bar(range(len(df_treatment)), df_treatment[var], color=colors)
            ax.axhline(y=treatment_results[0][var], color='red', 
                         linestyle='--', alpha=0.5, label='Diseased')
            ax.set_xticks(range(len(df_treatment)))
            ax.set_xticklabels(df_treatment['treatment'], rotation=45, ha='right')
            ax.set_ylabel(label)
            ax.set_title(f'{label} - Treatment Effects')
            sns.despine(ax=ax)
        
        for var, label in zip(variables, labels):
            plot_data = create_individual_plot(plot_treatment_bar, var=var, label=label)
            plots.append({
                'id': f'{var}_treatment',
                'title': f'{label} - Treatment Effects',
                'data': plot_data
            })
        
        # Prepare data
        data_csv = df_treatment.to_csv(index=False)
        
        # Add top perturbations info
        top_perturbations = [{'parameter': p, 'sensitivity': float(s)} 
                           for p, s in ranked_params[:10]]
        
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