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
    steady_state, fluxes, sensitivity_analysis, change_parameters
)

app = Flask(__name__, template_folder='template', static_folder='static')

# Set matplotlib style
plt.rcParams.update({'font.size': 12, 'font.family': 'Arial'})
sns.set_style("whitegrid")

def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/dynamic_response')
def dynamic_response():
    """Page for Type 1 Diabetes-like dynamic response simulation"""
    return render_template('dynamic_response.html', 
                         parameter_names=PARAMETER_NAMES,
                         parameter_descriptions=PARAMETER_DESCRIPTIONS)

@app.route('/insulin_clamp')
def insulin_clamp():
    """Page for hyperinsulinemic-euglycemic clamp simulation"""
    return render_template('insulin_clamp.html',
                         parameter_names=PARAMETER_NAMES,
                         parameter_descriptions=PARAMETER_DESCRIPTIONS)

@app.route('/tolerance_tests')
def tolerance_tests():
    """Page for GTT/ITT simulation with receptor knockouts"""
    return render_template('tolerance_tests.html',
                         parameter_names=PARAMETER_NAMES,
                         parameter_descriptions=PARAMETER_DESCRIPTIONS)

@app.route('/obesity')
def obesity():
    """Page for obesity simulation"""
    return render_template('obesity.html',
                         parameter_names=PARAMETER_NAMES,
                         parameter_descriptions=PARAMETER_DESCRIPTIONS)

@app.route('/treatment')
def treatment():
    """Page for treatment simulation"""
    return render_template('treatment.html',
                         parameter_names=PARAMETER_NAMES,
                         parameter_descriptions=PARAMETER_DESCRIPTIONS)

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
        X['G'] = X['G'] * 7  # 7 mM glucose
        X['F'] = X['F'] * 0.5  # 0.5 mM FFA
        X['K'] = X['K'] * 0.5  # 0.5 mM 3HB
        X['L'] = X['L'] * 0.7  # 0.7 mM lactate
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        variables = ['G', 'F', 'K', 'I']
        labels = ['Glucose (mM)', 'Fatty acids (mM)', '3-Hydroxybutyrate (mM)', 'Insulin (a.u.)']
        
        for i, (var, label) in enumerate(zip(variables, labels)):
            axes[i].plot(X['time'], X[var], linewidth=2, color='black')
            axes[i].axvline(time_perturbation, color='red', linestyle='--', alpha=0.5)
            axes[i].set_xlabel('Time (min)')
            axes[i].set_ylabel(label)
            axes[i].set_title(f'{label.split(" ")[0]} Response')
            sns.despine(ax=axes[i])
        
        plt.tight_layout()
        
        # Convert plot to base64
        img_str = plot_to_base64(fig)
        
        # Prepare data for download
        data_csv = X[['time', 'G', 'F', 'K', 'L', 'I', 'IA']].to_csv(index=False)
        
        return jsonify({
            'success': True,
            'plot': img_str,
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
        insulin_levels_str = data.get('insulin_levels', '0, 1, 2, 5')
        # Parse comma-separated string into list of floats
        insulin_levels = [float(x.strip()) for x in insulin_levels_str.split(',')]
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
        
        # Add infusion if specified
        if infusion_type != 'none':
            infusion_param = f'v_in_{infusion_type[0].upper()}'  # v_in_L, v_in_G, v_in_F, v_in_K
            p = change_parameters(p, [infusion_amount], ix=[infusion_param])
        
        # Run simulations for each insulin level
        time = np.linspace(0, time_max, 200)
        
        results = []
        gir_results = []
        
        for ins_level in insulin_levels:
            X, GIR = insulin_clamp_dynamic(ins_level, time, 1.0, p=p)
            
            # Scale concentrations
            X['G'] = X['G'] * 7
            X['F'] = X['F'] * 0.5
            X['K'] = X['K'] * 0.5
            X['L'] = X['L'] * 0.7
            X['insulin_level'] = ins_level
            
            GIR['insulin_level'] = ins_level
            
            results.append(X)
            gir_results.append(GIR)
        
        X_all = pd.concat(results)
        GIR_all = pd.concat(gir_results)
        
        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        
        # Time course plots
        variables = ['G', 'F', 'K', 'GIR']
        labels = ['Glucose (mM)', 'FFA (mM)', '3HB (mM)', 'GIR (a.u.)']
        
        for i, (var, label) in enumerate(zip(variables[:3], labels[:3])):
            for ins_level in insulin_levels:
                subset = X_all[X_all['insulin_level'] == ins_level]
                axes[0, i].plot(subset['time'], subset[var], label=f'Insulin: {ins_level}')
            axes[0, i].set_xlabel('Time (min)')
            axes[0, i].set_ylabel(label)
            axes[0, i].set_title(f'{label.split(" ")[0]} Time Course')
            axes[0, i].legend()
            sns.despine(ax=axes[0, i])
        
        # GIR time course
        for ins_level in insulin_levels:
            subset = GIR_all[GIR_all['insulin_level'] == ins_level]
            axes[1, 0].plot(subset['time'], subset['GIR'], label=f'Insulin: {ins_level}')
        axes[1, 0].set_xlabel('Time (min)')
        axes[1, 0].set_ylabel('GIR (a.u.)')
        axes[1, 0].set_title('Glucose Infusion Rate')
        axes[1, 0].legend()
        sns.despine(ax=axes[1, 0])
        
        # Steady state bar plots
        # Calculate steady state (last 20% of simulation)
        cutoff = time_max * 0.8
        X_ss = X_all[X_all['time'] > cutoff].groupby('insulin_level').mean().reset_index()
        GIR_ss = GIR_all[GIR_all['time'] > cutoff].groupby('insulin_level').mean().reset_index()
        
        # Bar plot for glucose
        axes[1, 1].bar(range(len(insulin_levels)), X_ss['G'], color='steelblue')
        axes[1, 1].set_xticks(range(len(insulin_levels)))
        axes[1, 1].set_xticklabels([f'{x}' for x in insulin_levels])
        axes[1, 1].set_xlabel('Insulin Level')
        axes[1, 1].set_ylabel('Glucose (mM)')
        axes[1, 1].set_title('Steady State Glucose')
        sns.despine(ax=axes[1, 1])
        
        # Bar plot for GIR
        axes[1, 2].bar(range(len(insulin_levels)), GIR_ss['GIR'], color='coral')
        axes[1, 2].set_xticks(range(len(insulin_levels)))
        axes[1, 2].set_xticklabels([f'{x}' for x in insulin_levels])
        axes[1, 2].set_xlabel('Insulin Level')
        axes[1, 2].set_ylabel('GIR (a.u.)')
        axes[1, 2].set_title('Steady State GIR')
        sns.despine(ax=axes[1, 2])
        
        plt.tight_layout()
        
        # Convert plot to base64
        img_str = plot_to_base64(fig)
        
        # Prepare data
        data_csv = X_all.to_csv(index=False)
        
        return jsonify({
            'success': True,
            'plot': img_str,
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
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot glucose response
        axes[0].plot(X_control['time'], X_control['G'], linewidth=2, 
                    color='grey', label='Control')
        axes[0].plot(X_ko['time'], X_ko['G'], linewidth=2, 
                    color='steelblue', label='K/O', linestyle='--')
        axes[0].set_xlabel('Time (min)')
        axes[0].set_ylabel('Glucose (mM)')
        axes[0].set_title(f'{test_type} - Glucose Response')
        axes[0].legend()
        sns.despine(ax=axes[0])
        
        # Plot insulin response
        axes[1].plot(X_control['time'], X_control['I'], linewidth=2, 
                    color='grey', label='Control')
        axes[1].plot(X_ko['time'], X_ko['I'], linewidth=2, 
                    color='steelblue', label='K/O', linestyle='--')
        axes[1].set_xlabel('Time (min)')
        axes[1].set_ylabel('Insulin (a.u.)')
        axes[1].set_title(f'{test_type} - Insulin Response')
        axes[1].legend()
        sns.despine(ax=axes[1])
        
        plt.tight_layout()
        
        # Convert plot to base64
        img_str = plot_to_base64(fig)
        
        # Prepare data
        X_control['group'] = 'Control'
        X_ko['group'] = 'K/O'
        data_csv = pd.concat([X_control, X_ko]).to_csv(index=False)
        
        return jsonify({
            'success': True,
            'plot': img_str,
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
        
        # Create plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        variables = ['glucose', 'insulin', 'HOMA_IR']
        labels = ['Glucose (mM)', 'Insulin (a.u.)', 'HOMA-IR']
        
        for i, (var, label) in enumerate(zip(variables, labels)):
            # Plot model results
            axes[i].plot(df_control['fat_fraction'], df_control[var], 
                        linewidth=2, color='black', label='Model')
            
            if df_perturbed is not None:
                axes[i].plot(df_perturbed['fat_fraction'], df_perturbed[var], 
                           linewidth=2, color='red', linestyle='--', label='Perturbed')
            
            # Plot NHANES data if available
            if df_nhanes is not None and var in df_nhanes.columns:
                # Assuming fat_fraction or BMI column exists
                if 'fat_fraction' in df_nhanes.columns:
                    # Split by sex if available
                    if 'Sex' in df_nhanes.columns:
                        for sex, marker, color in [('Male', 'o', 'blue'), ('Female', 's', 'pink')]:
                            subset = df_nhanes[df_nhanes['Sex'] == sex]
                            axes[i].scatter(subset['fat_fraction'], subset[var], 
                                          alpha=0.3, s=10, marker=marker, 
                                          color=color, label=f'{sex} (NHANES)')
            
            axes[i].set_xlabel('Fat Fraction (relative to control)')
            axes[i].set_ylabel(label)
            axes[i].set_title(f'{label} vs Adiposity')
            axes[i].legend()
            sns.despine(ax=axes[i])
        
        plt.tight_layout()
        
        # Convert plot to base64
        img_str = plot_to_base64(fig)
        
        # Prepare data
        df_control['group'] = 'Control'
        if df_perturbed is not None:
            df_perturbed['group'] = 'Perturbed'
            data_csv = pd.concat([df_control, df_perturbed]).to_csv(index=False)
        else:
            data_csv = df_control.to_csv(index=False)
        
        return jsonify({
            'success': True,
            'plot': img_str,
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
        
        # Add insulin treatment (increase Imax)
        p_insulin = change_parameters(p_perturbed, 
                                     [1.5 * p_perturbed[PARAMETER_NAMES.index('Imax')]], 
                                     ix=['Imax'])
        X_insulin = perturbation_steady_state(A, p=p_insulin)
        treatment_results.append({
            'treatment': 'Insulin',
            'G': X_insulin[1] * 7,
            'I': X_insulin[4],
            'HOMA_IR': X_insulin[1] * X_insulin[4] * 7
        })
        
        df_treatment = pd.DataFrame(treatment_results)
        
        # Create plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        variables = ['G', 'I', 'HOMA_IR']
        labels = ['Glucose (mM)', 'Insulin (a.u.)', 'HOMA-IR']
        colors = ['steelblue'] * len(treatment_results)
        colors[0] = 'red'  # Diseased state in red
        
        for i, (var, label) in enumerate(zip(variables, labels)):
            axes[i].bar(range(len(df_treatment)), df_treatment[var], color=colors)
            axes[i].axhline(y=treatment_results[0][var], color='red', 
                          linestyle='--', alpha=0.5, label='Diseased')
            axes[i].set_xticks(range(len(df_treatment)))
            axes[i].set_xticklabels(df_treatment['treatment'], rotation=45, ha='right')
            axes[i].set_ylabel(label)
            axes[i].set_title(f'{label} - Treatment Effects')
            sns.despine(ax=axes[i])
        
        plt.tight_layout()
        
        # Convert plot to base64
        img_str = plot_to_base64(fig)
        
        # Prepare data
        data_csv = df_treatment.to_csv(index=False)
        
        # Add top perturbations info
        top_perturbations = [{'parameter': p, 'sensitivity': float(s)} 
                           for p, s in ranked_params[:10]]
        
        return jsonify({
            'success': True,
            'plot': img_str,
            'data': data_csv,
            'top_perturbations': top_perturbations
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)