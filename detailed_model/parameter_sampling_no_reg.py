from skimpy.analysis.oracle.load_pytfa_solution import load_fluxes, load_concentrations,\
    load_equilibrium_constants
from skimpy.core.parameters import ParameterValuePopulation, load_parameter_population
from pytfa.io.json import load_json_model
from skimpy.io.yaml import load_yaml_model

# Compute Nv 
from skimpy.analysis.ode.utils import make_flux_fun
from skimpy.utils.namespace import QSSA
from skimpy.utils.general import get_stoichiometry

# Parameter sampling
from skimpy.sampling.simple_parameter_sampler import SimpleParameterSampler
import seaborn as sns
import matplotlib.pyplot as plt 

from skimpy.analysis.modal import modal_matrix


import pandas as pd 
import numpy as np
import os

from tqdm import tqdm

if __name__ == '__main__':

    # Load the tfa model 
    model_file = 'reduced_model_ETC_core_20250228-213124_continuous.json'
    tmodel = load_json_model(model_file)
    #sol = tmodel.optimize()

    # Reload and prepare the model
    kmodel = load_yaml_model(model_file.replace("_continuous.json", "_kinetic_curated_no_reg.yml"))


    faraday_const = 23.061 # kcal / mol / V
    RT = tmodel.RT # kcal /mol
    delta_psi_scaled = 150/1000 * faraday_const / RT  # mV * F / RT 

    # obsolete in current version 
    kmodel.parameters.capacitance_MPDYN_m_MitoMembranePot_in.value = 1 # 600 # eq. 30 mH / min 
    kmodel.parameters.capacitance_MPDYN_m_MitoMembranePot_out.value = 1 # 600 # eq. 30 mH / min 


    # Parametrize the membrane potential modifiers
    for parameter in kmodel.parameters.values(): 
        if 'delta_psi_scaled_MPM_psi_m_c' in str(parameter.symbol):
            parameter.value = delta_psi_scaled
            print(parameter.symbol , parameter.value)

    # Parametrize the membrane potential modifiers
    # Charge export from mitochondria
    # Pos
    kmodel.parameters.charge_transport_MPM_psi_m_c_NADH2_u10mi.value = 4 # 4 pos charges from inside (Complex I)
    kmodel.parameters.charge_transport_MPM_psi_m_c_CYOR_u10mi.value = 4 # 4 H+ from insinde (Complex III)
    kmodel.parameters.charge_transport_MPM_psi_m_c_CYOOm2i.value = 4 # 4 H+ from insinde (Complex IV)

    # Dummy reactions for proton transport by other means 
    kmodel.parameters.charge_transport_MPM_psi_m_c_ATPtm.value = -1 # -1 to the outside 

    # Neg charge export from mitochondria
    kmodel.parameters.charge_transport_MPM_psi_m_c_ATPtm.value = -1 # -1 to the outside 
    
    # Charge import into mitochondria
    # Pos
    kmodel.parameters.charge_transport_MPM_psi_m_c_ASPGLUm.value = -1 # 1 H+ to the inside
    kmodel.parameters.charge_transport_MPM_psi_m_c_ATPS4mi.value = -3 # 3 H+ to the inside

    # Compile the jacobian expressions
    NCPU = 8
    kmodel.repair()
    kmodel.prepare(mca=False)
    kmodel.compile_jacobian(ncpu=NCPU)


    # Initiate a parameter sampler
    params = SimpleParameterSampler.Parameters(n_samples = 10)
    sampler = SimpleParameterSampler(params)


    # Load TFA samples 
    tfa_sample_file = 'reduced_model_ETC_core_20250228-213124_tfa_sampling.csv'
    tfa_samples = pd.read_csv(tfa_sample_file)

    # Scaling parameters
    CONCENTRATION_SCALING = 1e3 # 1 mol to 1 mmol
    TIME_SCALING = 1.0 # 1min
    DENSITY = 1200 # g/L 
    GDW_GWW_RATIO = 1.0 # Fluxes are in gWW

    # To test how close to zero the dxdt is
    flux_scaling_factor = 1e-6 / (GDW_GWW_RATIO / DENSITY) \
                          * CONCENTRATION_SCALING \
                          / TIME_SCALING
    

    # Psuedo data for the membrane potential
    tfa_samples['psi_m_c'] = delta_psi_scaled * 1e-3 # be aware of the scaling!!!
    tfa_samples['MitoMembranePot_in'] = 1000 # ~ 1000 RT/min -> at 5 RT is equivalent to about 2 min time scale for the membrane potential
    tfa_samples['MitoMembranePot_out'] = 1000

    # 30 min timescale for insulin action 
    tfa_samples['Insulin_secretion'] = 1/30 / flux_scaling_factor
    tfa_samples['Insulin_degradation'] = 1/30 / flux_scaling_factor
    tfa_samples['insulin_e'] = 1e-3 

    additional_fluxes = ['MitoMembranePot_in','MitoMembranePot_out', 'Insulin_secretion', 'Insulin_degradation',]
    additional_concentrations = ['psi_m_c', 'insulin_e']

    # Secondpass for vmax calc:
    kmodel.second_pass = ['MitoMembranePot_in','MitoMembranePot_out']

    # Get the stoichiometry matrix
    S = get_stoichiometry(kmodel, kmodel.reactants).todense()

    lambda_max_all = []
    lambda_min_all = []

    # Make a new directory for the output
    os.makedirs(tfa_sample_file.replace(".csv",'_no_reg'), exist_ok=True)

    path_for_output = './'+tfa_sample_file.replace(".csv",'_no_reg')+'/paramter_pop_{}.h5'

    flux_fun = make_flux_fun(kmodel, QSSA)


    # NOTE DW -> TRANSPORTERS SHOULD HAVE SAME KM for substrate and product pairs
    for i, sample in tqdm(tfa_samples.iterrows()):
        # Load fluxes and concentrations
        fluxes = load_fluxes(sample, tmodel, kmodel,
                                density=DENSITY,
                                ratio_gdw_gww=GDW_GWW_RATIO,
                                concentration_scaling=CONCENTRATION_SCALING,
                                time_scaling=TIME_SCALING,
                                xmol_in_flux=1e-6,
                                additional_fluxes=additional_fluxes)
        
        concentrations = load_concentrations(sample, tmodel, kmodel,
                                             concentration_scaling=CONCENTRATION_SCALING,
                                             additional_concentrations=additional_concentrations)

        ##################################################################################
        # Assume all KM are unsaturated 1 to 2 orders of magnitude above the concentration
        # Further we limit the activation and inhibition constants 0.02 / 50 fold
        ##################################################################################
        
        cofactors = [ 'atp_c', 'adp_c', 'atp_m', 'adp_m', 'amp_c',
                      'nad_c','nad_m', 
                      'nadh_m',  'nadh_c', 
                      'coa_c', 'coa_m',
                      'crn_c', 'crn_m',
                      'fad_m','fadh2_m', 'q10_m', 'q10h2_m',
                      ]
        

        for p_name,param in kmodel.parameters.items():
            if 'km' in p_name:
                concentration_hook = param.hook
                this_concentration = concentrations[concentration_hook.name]
                if concentration_hook.name in cofactors:
                    kmodel.parameters[p_name].bounds = (this_concentration*0.1, this_concentration*100)
                else:
                    kmodel.parameters[p_name].bounds = (this_concentration*10, this_concentration*100)


            if ('k_activation' in p_name) or ('k_inhibition' in p_name):
                concentration_hook = param.hook
                this_concentration = concentrations[concentration_hook.name]
                kmodel.parameters[p_name].bounds = (this_concentration*0.1, this_concentration*100)

        ##################################################################################
        # Integrate parameters from Bernda and integrate assumptions from Lenhninger 
        ##################################################################################

        # ATP dissipation mainly by myosin atpase KM 200 uM
        # https://www.med.upenn.edu/ostaplab/assets/user-content/documents/mie-reprint.pdf
        kmodel.reactions.cyt_atp2adp.parameters.km_substrate1.bounds = (0.19, 0.21) # 

        # Proton leak model (saturated doe not change with mitochondrial proton concentration)
        h_m = concentrations['h_m']
        kmodel.reactions.proton_in.parameters.km_product1.bounds = (0.01 * h_m, 0.02 * h_m) 

        # Fetch equilibrium constants
        load_equilibrium_constants(sample, tmodel, kmodel,
                                concentration_scaling=CONCENTRATION_SCALING,
                                in_place=True)
        

        # Generate sampels and fetch slowest and fastest eigenvalues
        params, lamda_max, lamda_min = sampler.sample(kmodel, fluxes, concentrations,
                                                        only_stable=False,
                                                        min_max_eigenvalues=True,
                                                        bounds_sample=(0, 1.0),
                                                        seed=100+i)
        
        # Test Nv = 0 
        params_population = ParameterValuePopulation(params, kmodel)

        # Test if the resulting sets are NV=0
        fluxes_1 = flux_fun(concentrations, parameters=params_population['0'])
        fluxes_1 = pd.Series(fluxes_1)
        dxdt = S.dot(fluxes_1[kmodel.reactions])
        if np.any(abs(dxdt) > 1e-6*flux_scaling_factor):
            raise RuntimeError('dxdt for idx {} not equal to 0'
                               .format(np.where(abs(dxdt) > 1e-6*flux_scaling_factor)))
        
        # Skip TFA sample when there is any nan in lambade_max
        if np.isnan(lamda_max).any():
            #raise RuntimeError('Nan in lambda_max for idx {}'.format(i))
            Warning('Nan in lambda_max for idx {}'.format(i))
            
        else:
            lambda_max_all.append(pd.DataFrame(lamda_max))
            lambda_min_all.append(pd.DataFrame(lamda_min))

            params_population.save(path_for_output.format(i))


    # Process df and save dataframe
    lambda_max_all = pd.concat(lambda_max_all, axis=1)
    lambda_min_all = pd.concat(lambda_min_all, axis=1)

    # Save the eigenvalue distributino
    lambda_max_all.to_csv(tfa_sample_file.replace(".csv","_lambda_max.csv"))
    lambda_min_all.to_csv(tfa_sample_file.replace(".csv","_lambda_min.csv"))

    """
    Prune parameters based on the time scales
    """


    # Build index from files in path_for_output
    output = './reduced_model_ETC_core_20250228-213124_tfa_sampling_no_reg'
    index = []
    for file in os.listdir(output):
        if file.endswith(".h5"):
            index.append(file.split('_')[-1].split('.')[0])

    lambda_max_all.columns = index

    # Prune parameter based on the maximum eigenvalue
    MAX_EIGENVALUES = -1/40 # 40 min time scale for the slowest eigenvalue 
    # -> 3 * 40 min should be enought to comeback to the steady state

    is_selected = (lambda_max_all < MAX_EIGENVALUES )

    fast_parameters = []
    fast_index = []

    for i, row in is_selected.T.iterrows():
        if any(row):
            fast_models = np.where(np.array(row))[0]
            # Load the respective solutions
            parameter_population = load_parameter_population(path_for_output.format(i))
            fast_parameters.extend([parameter_population._data[k] for k in fast_models])
            fast_index.extend(["{},{}".format(i,k) for k in fast_models])

    # Generate a parameter population file
    parameter_population = ParameterValuePopulation(fast_parameters,
                                               kmodel=kmodel,
                                               index=fast_index)
    # Save the pruned parameter population
    parameter_population.save( tfa_sample_file.replace(".csv",'_pruned_parameters_no_reg.hdf5'))



    # # Modal analysis
    # from skimpy.analysis.modal import modal_matrix
    # from skimpy.viz.modal import plot_modal_matrix
    # import random

    # # Pic a random parameter set and plot the modal matrix
    # index = random.choice(list(parameter_population._index.keys()))
    # # Print the index
    # print(f"Will perform modal analysis on index: {index}")

    # sample = tfa_samples.iloc[int(index.split(',')[0])]
    # concentrations = load_concentrations(sample, tmodel, kmodel,
    #                                             concentration_scaling=CONCENTRATION_SCALING,
    #                                             additional_concentrations=additional_concentrations)
    # parameter_values = parameter_population[index]

    # kmodel.prepare(mca=False)
    # kmodel.compile_jacobian(sim_type=QSSA,ncpu=8)
    # M = modal_matrix(kmodel,concentrations,parameter_values)

    # plot_modal_matrix(M,filename='modal_matrix.html',
    #                   width=800, height=600,
    #                   clustered=True,
    #                   backend='svg',
    #                   )
    
    # Make a histogram of the slow eigenvalues
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 240, 100)
    plt.hist( -1/np.real(lambda_max_all.values.flatten()), bins=bins)
    plt.show()
    plt.savefig(tfa_sample_file.replace(".csv", "_eigenvalue_histogram_no_reg.svg"))
