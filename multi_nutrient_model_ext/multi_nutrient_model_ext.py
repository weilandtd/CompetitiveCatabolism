import numpy as np 
import pandas as pd
from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

import warnings
warnings.filterwarnings('ignore')


######################################
# Constants and parameters
######################################

# FIXED RATIOS (Energy per molecule rel to glucose 32 ATP per glucose)
nG = 32   # Glucose
nL = 15   # Lactate
nF = 108  # Fatty acids
nK = 22.5 # Ketones (3-hydroxybutyrate)

# ATP per O2 (5 per O2, considering P/O ratio)
PO2 = 5.0
# Whole body oxygen consumption rate ~ 2000 nmol/min/gBW
vO2 = 2000
# ATP production rate (scaled by 0.75 to match base model)
vATP = PO2 * vO2 * 0.75

# Hormonal time constants
TAU_INS = 2.0  # Insulin degradation time constant (min)
TAU_INS_A = 30.0  # Insulin signaling time constant (min)
TAU_GCG = 5.0  # Glucagon degradation time constant (min)
TAU_GCG_A = 30.0  # Glucagon signaling time constant (min)

# Time constants for metabolite dynamics (from volume distribution)
TAU_L = 5.0  # Lactate (min)
TAU_F = 6.5  # Fatty acids (min)
TAU_G = 21.0  # Glucose (min)
TAU_K = 3.0  # Ketones (min)
TAU_AA = 5.0  # Amino acids (min)

# Reference insulin level (dimensionless)
# At baseline glucose G=1.0, compute insulin secretion
h_ins = 3.4  # Insulin secretion Hill coefficient
C_ins = 2.3  # Insulin secretion threshold (relative to glucose)
I0 = abs(1.0)**h_ins / (abs(1.0)**h_ins + C_ins**h_ins)

# Reference glucagon level (dimensionless)
# At baseline glucose G=1.0, compute glucagon secretion (inverse glucose dependence)
# Glucagon formula: GCG_max * (1 - G^h/(G^h + C^h))
h_gcg = 1.0  # Glucagon secretion Hill coefficient
C_gcg = 0.6  # Glucagon secretion threshold (relative to glucose ~3.5mM)
# Glucagon is inversely related to glucose: GCG_max when G→0, ~0 when G→∞
GCG0 = 1.0 * (1.0 - 1.0**h_gcg/(1.0**h_gcg + C_gcg**h_gcg))  # At G=1.0

# Amino acid list (19 amino acids, excluding tryptophan)
AMINO_ACIDS = [
    'Ala', 'Arg', 'Asn', 'Asp', 'Cys', 
    'Gln', 'Glu', 'Gly', 'His', 'Ile',
    'Leu', 'Lys', 'Met', 'Phe', 'Pro',
    'Ser', 'Thr', 'Tyr', 'Val'
]

# Amino acids that stimulate insulin secretion in a glucose-dependent manner
# Based on literature: Arg, Lys, Ala, Pro, Leu, Gln
INSULINOTROPIC_AA = ['Arg', 'Lys', 'Ala', 'Pro', 'Leu', 'Gln']

# Amino acids that stimulate glucagon secretion (liver-alpha cell axis)
# Based on literature: Ala, Arg, Cys, Pro (but NOT Gln)
# These amino acids act as acute mediators of glucagon secretion
GLUCAGONOTROPIC_AA = ['Ala', 'Arg', 'Cys', 'Pro']

# Energy yield per amino acid (ATP per amino acid)
# Based on complete oxidation accounting for oxidation state and metabolic pathways
# Values from biochemistry literature (Nelson & Cox, Lehninger Principles of Biochemistry)
# Calculations assume: NADH = 2.5 ATP, FADH2 = 1.5 ATP, GTP = 1 ATP
nAA = {
    'Ala': 12.5,   # -> Pyruvate -> Acetyl-CoA: 2.5 NADH + 1 FADH2 + 3 Acetyl-CoA(10 ATP) = 12.5 ATP
    'Arg': 30,     # -> α-Ketoglutarate (5C): ~30 ATP via TCA cycle
    'Asn': 12,     # -> Oxaloacetate: 2.5 NADH + 10 ATP (via oxaloacetate) = 12 ATP  
    'Asp': 12,     # -> Oxaloacetate: 2.5 NADH + 10 ATP = 12 ATP
    'Cys': 13.5,   # -> Pyruvate: similar to Ala, ~13.5 ATP
    'Gln': 27,     # -> α-Ketoglutarate: ~27 ATP
    'Glu': 25,     # -> α-Ketoglutarate: 5 NADH + 1 FADH2 + 1 GTP = 25 ATP
    'Gly': 11.5,   # -> Serine -> Pyruvate: ~11.5 ATP
    'His': 25,     # -> α-Ketoglutarate: ~25 ATP
    'Ile': 43,     # -> Succinyl-CoA + Acetyl-CoA: branched-chain, ~43 ATP
    'Leu': 39,     # -> Acetyl-CoA + Acetoacetate: branched-chain, ~39 ATP
    'Lys': 36,     # -> Acetoacetyl-CoA: ~36 ATP
    'Met': 29,     # -> Succinyl-CoA: ~29 ATP
    'Phe': 38,     # -> Fumarate + Acetoacetate: ~38 ATP
    'Pro': 30,     # -> α-Ketoglutarate: ~30 ATP
    'Ser': 12.5,   # -> Pyruvate: ~12.5 ATP
    'Thr': 23,     # -> Succinyl-CoA: ~23 ATP
    'Tyr': 36,     # -> Fumarate + Acetoacetate: ~36 ATP
    'Val': 32,     # -> Succinyl-CoA: branched-chain, ~32 ATP
    'Trp': 
}

# Carbon content per amino acid
cAA = {
    'Ala': 3, 'Arg': 6, 'Asn': 4, 'Asp': 4, 'Cys': 3,
    'Gln': 5, 'Glu': 5, 'Gly': 2, 'His': 6, 'Ile': 6,
    'Leu': 6, 'Lys': 6, 'Met': 5, 'Phe': 9, 'Pro': 5,
    'Ser': 3, 'Thr': 4, 'Tyr': 9, 'Val': 5,
}

# Nitrogen content per amino acid (number of N atoms)
# Only accounts for AAs making local nitrogen exclude Ala and Gln
nNAA = {
    'Ala': 0, 'Arg': 4, 'Asn': 2, 'Asp': 1, 'Cys': 1,
    'Gln': 0, 'Glu': 1, 'Gly': 1, 'His': 3, 'Ile': 1,
    'Leu': 1, 'Lys': 2, 'Met': 1, 'Phe': 1, 'Pro': 1,
    'Ser': 1, 'Thr': 1, 'Tyr': 1, 'Val': 1,
}

# Protein composition: relative abundance in mouse proteome
# Based on whole-body hydrolysate analysis (experimental data)
# Molar percentages from mouse_aa_abundance.tsv analysis
protein_composition = {
    'Ala': 0.0852340, 'Arg': 0.0482851, 'Asn': 0.0353025, 'Asp': 0.0479887, 'Cys': 0.0217910,
    'Gln': 0.0459721, 'Glu': 0.0675603, 'Gly': 0.1297861, 'His': 0.0209325, 'Ile': 0.0381187,
    'Leu': 0.0780186, 'Lys': 0.0669661, 'Met': 0.0184775, 'Phe': 0.0338044, 'Pro': 0.0779294,
    'Ser': 0.0498018, 'Thr': 0.0439362, 'Tyr': 0.0241138, 'Val': 0.0600320,
}

# Parameter names for indexing the parameter array
# Indices 0-41: Core parameters
PARAMETER_NAMES = [
    "v_energy", "k_glycolysis", "k_lactate", "k_glucose", "k_fatty_acids", "k_3HB",
    "k_lipolysis", "k_reesterification", "k_ketogenesis", "k_gluconeogenesis",
    "V_glycogenolysis", "k_protein_degradation", "V_protein_synthesis",
    "K_i_lipolysis", "K_a_glycolysis", "K_i_glycogenolysis", "K_i_ketogenesis", "K_i_protein_degradation",
    "K_a_gluconeogenesis", "K_a_glycogenolysis",
    "I_max", "h_ins", "C_ins", "GCG_max", "h_gcg", "C_gcg",
    "R_insulin", "R_glucagon", "R_lactate", "R_glucose", "R_fatty_acids", "R_3HB",
    "k_N_recycling", "alpha_N", "k_GNG_Gln", "k_GNG_Ala", "k_AL", "A_ref",
    "fN_Glu_Gln", "fN_Lac_Ala", "fN_Lac_Gln", "fN_Asp_Gln", "fN_Pro_Gln", "fN_Gly_Gln",
]

# Indices 44-62: Amino acid oxidation rate constants (19 amino acids)
PARAMETER_NAMES.extend([f"k_ox_{aa}" for aa in AMINO_ACIDS])

# Indices 63-69: Action flags (7 flags)
PARAMETER_NAMES.extend([
    "insulin_action_lipolysis",
    "insulin_action_glycolysis",
    "insulin_action_glycogenolysis",
    "insulin_action_ketogenesis",
    "insulin_action_protein_degradation",
    "glucagon_action_gluconeogenesis",
    "glucagon_action_glycogenolysis",
])


######################################
# Mass and energy balance constraints
######################################


def mass_and_energy_constraints(v, v_energy=1.0, 
                                FG = 100/vATP,   # Glucose appearance
                                FL = 150/vATP,   # Lactate appearance
                                FK = 30/vATP,    # Ketone appearance
                                FF = 150/vATP,   # Fatty acid appearance
                                FPD = 160*0.8/vATP,  # Protein degradation -> ciculation 
                                FPS = 100*0.8/vATP,   # Protein synthesis 
                                FALA = 25/vATP,  # Alanine appearance
                                FGLN = 40/vATP,  # Glutamine appearance
                                FGLU = 4/vATP,   # Glutamate appearance
                                FASP = 2.5/vATP, # Aspartate appearance
                                FPRO = 5/vATP,   # Proline appearance
                                FGLY = 14/vATP,  # Glycine appearence
                                return_dict=False,  # Return dict instead of list
                                ):
    """
    Mass and energy balance constraints for multi-nutrient model with individual amino acids.
    
    Variables (fluxes):
    -------------------
    Core metabolism:
    - vL: Lactate oxidation
    - vG: Glucose oxidation
    - vF: Fatty acid oxidation
    - vK: Ketone oxidation
    - vGL: Glycolysis (glucose -> lactate)
    - vFK: Ketogenesis (fatty acids -> ketones)
    - vLG: Gluconeogenesis from lactate
    - v0: Glycogenolysis
    - vLip: Lipolysis
    - vReest: Re-esterification
    - vCO2: CO2 production


    
    Constraints:
    ------------
    1. Mass balances for L, G, F, K (steady state: production = consumption)
    2. Mass balances for each amino acid (19 equations)
    3. CO2 balance (carbon balance)
    4. Energy balance (ATP production = expenditure)
    5. Gluconeogenesis constraint (total GNG = 1/2 * lipolysis)
    6. Re-esterification constraint (2/3 of released FFA)
    7. Protein turnover constraint (synthesis = degradation at steady state)
    8. Appearance rate constraints for G, L, F, K
    """
    
    # Unpack fluxes
    vL, vG, vF, vK, vGL, vFK, vLG, v0, vLip, vReest, vCO2 = v[0:11]
    vPD, vPS = v[11:13]
    
    # Amino acid oxidation fluxes (19)
    vOx = {aa: v[13 + i] for i, aa in enumerate(AMINO_ACIDS)}

    # GLuconeogensis from AA only for Gln and Ala (based labeling data)
    v_GNG_Gln = v[32] # Gln -> 1/2 Glc and processing of the resulting ammonia into urea
    v_GNG_Ala = v[33] # Ala -> 1/2 Glc and processing of the resulting ammonia into urea
    
    # Local nitrogen recycling glutamine and alanine production
    # These are FREE VARIABLES in the constraint system - no fixed ratios imposed here
    vN_Glu_Gln = v[34] # Glu + N -> Gln 
    vN_Asp_Gln = v[35] # Asp + N -> Gln (+ Acetyl CoA, not modeled)
    vN_Pro_Gln = v[36] # Pro + N -> Gln
    vN_Gly_Gln = v[37] # Gly + N -> Gln (+ 1C unit + Acetyl CoA, not modeled)
    vN_Lac_Gln = v[38] # 2Lac + 2N -> Gln (lactate to glutamine, consumes 2N per flux unit)
    vN_Lac_Ala = v[39] # Lac + N -> Ala (glucose-alanine cycle)
    v_Ala_Lac = v[40] # Ala -> Lac (local release of lactate and processing of ammonia into urea)

    # Constraints dictionary
    constraints = {}
    
    # ========================================
    # 1. Mass balances for core metabolites
    # ========================================
    
    # Lactate: produced by glycolysis and Ala->Lac, consumed by oxidation, gluconeogenesis, and N recycling
    dLdt = 2.0*vGL + v_Ala_Lac - 2.0*vLG - vL - vN_Lac_Ala - vN_Lac_Gln * 5/3
    constraints['dLdt'] = dLdt
    
    # Glucose: produced by glycogenolysis, GNG from lactate & AA, and lipolysis
    #          consumed by glycolysis and oxidation
    dGdt = v0 + 1/2*(vLip - vReest) + vLG - vGL - vG + 1/2*v_GNG_Gln + 1/2*v_GNG_Ala
    constraints['dGdt'] = dGdt
    
    # Fatty acids: produced by lipolysis, consumed by oxidation, ketogenesis, re-esterification
    dFdt = 3.0*(vLip - vReest) - vF - vFK
    constraints['dFdt'] = dFdt
    
    # Ketones: produced by ketogenesis, consumed by oxidation
    dKdt = 4.0*vFK - vK
    constraints['dKdt'] = dKdt
    
    # ========================================
    # 2. Mass balances for amino acids
    # ========================================
    for aa in AMINO_ACIDS:
        gamma_aa = protein_composition[aa]
        # Each AA: produced by protein degradation, consumed by protein synthesis, oxidation, and GNG
        dAAdt = gamma_aa * vPD - gamma_aa * vPS - vOx[aa]
        constraints[f'dAAdt_{aa}'] = dAAdt    

    # Glutamate: consumed by N recycling to make glutamine
    constraints['dAAdt_Glu'] -= vN_Glu_Gln
    
    # Aspartate: consumed by N recycling to make glutamine
    constraints['dAAdt_Asp'] -= vN_Asp_Gln
    
    # Proline: consumed by N recycling to make glutamine
    constraints['dAAdt_Pro'] -= vN_Pro_Gln
    
    # Glycine: consumed by N recycling to make glutamine
    constraints['dAAdt_Gly'] -= vN_Gly_Gln

    # Glutamine: produced by N recycling pathways, consumed by GNG
    constraints['dAAdt_Gln'] += vN_Glu_Gln + vN_Asp_Gln + vN_Pro_Gln + vN_Gly_Gln + vN_Lac_Gln
    constraints['dAAdt_Gln'] -= v_GNG_Gln
    
    # Alanine: produced by lactate-alanine cycle, consumed by GNG and Ala->Lac conversion
    constraints['dAAdt_Ala'] += vN_Lac_Ala
    constraints['dAAdt_Ala'] -= v_GNG_Ala
    constraints['dAAdt_Ala'] -= v_Ala_Lac
    
    # ========================================
    # 3. CO2 balance (carbon balance)
    # ========================================
    CO2_from_core = 3*vL + 6*vG + 16*vF + 4*vK
    CO2_from_AA = sum(cAA[aa] * vOx[aa] for aa in AMINO_ACIDS)
    dCO2 = CO2_from_core + CO2_from_AA - vCO2
    constraints['dCO2'] = dCO2

    # ========================================
    # 4. Nitrogen balance
    # ========================================
    # At steady state, nitrogen balance should be zero (nitrogen in = nitrogen out)
    # Nitrogen from oxidation is recycled via multiple pathways
    # Nitrogen from GNG is directly exported (not recycled locally)
    N_from_oxidation = sum(nNAA[aa] * vOx[aa] for aa in AMINO_ACIDS)
    
    # Total nitrogen consumed by recycling pathways
    # Note: vN_Lac_Gln consumes 2N per flux unit (2Lac + 2N -> Gln)
    N_consumed = vN_Glu_Gln + vN_Asp_Gln + vN_Pro_Gln + vN_Gly_Gln + vN_Lac_Ala + 2*vN_Lac_Gln
    
    # Nitrogen balance: nitrogen consumed by recycling = nitrogen from oxidation
    constraints['dN'] = N_from_oxidation - N_consumed
    
    # ========================================
    # 5. Energy balance // Competitive catabolism
    # ========================================
    energy_from_core = nL*vL + nG*vG + nF*vF + nK*vK + 2*vGL
    energy_from_AA = sum(nAA[aa] * vOx[aa] for aa in AMINO_ACIDS)
    dE = energy_from_core + energy_from_AA - v_energy
    constraints['dE'] = dE
    
    # ========================================
    # 6. ADDITIONAL CONSTRAINTS
    # ========================================
    
    
    # Re-esterification constraint (2/3 of FFA is re-esterified)
    dReest = vReest - 2/3 * vLip
    constraints['dReest'] = dReest

    #################################
    ## Appearance rate constraints ##
    #################################

    # Glucose dissapreance FG
    dFG = vGL + vG + 0.5*vLip - FG
    constraints['dFG'] = dFG 
    
    # Lactate disappearance FL
    dFL = 2*vLG + vL + vN_Lac_Ala + vN_Lac_Gln * 5/3 - FL
    constraints['dFL'] = dFL
    
    # Fatty acid dissapearance FF
    dFF = vFK + vF + 3*vReest - FF
    constraints['dFF'] = dFF
    
    # Ketone dissapearance FK
    dFK = vK - FK
    constraints['dFK'] = dFK

    # Alanine appearance FAL
    dFAL = vN_Lac_Ala + vPD * protein_composition['Ala'] - FALA
    constraints['dFAL'] = dFAL

    # Glutamine appearance FGLN
    dFGLN = (vN_Glu_Gln + vN_Asp_Gln + vN_Pro_Gln + vN_Gly_Gln + vN_Lac_Gln + 
             vPD * protein_composition['Gln']) - FGLN
    constraints['dFGLN'] = dFGLN

    # Glutamate appearance FGLU
    dFGLU = vPD * protein_composition['Glu'] - vN_Glu_Gln - FGLU
    constraints['dFGLU'] = dFGLU

    # Aspartate appearance FASP
    dFASP = vPD * protein_composition['Asp'] - vN_Asp_Gln - FASP
    constraints['dFASP'] = dFASP

    # Proline appearance FPRO
    dFPRO = vPD * protein_composition['Pro'] - vN_Pro_Gln - FPRO
    constraints['dFPRO'] = dFPRO

    # Glycine appearance FGLY
    dFGLY = vPD * protein_composition['Gly'] - vN_Gly_Gln - FGLY
    constraints['dFGLY'] = dFGLY

    # Protein degradation PD
    dPD = vPD - FPD
    constraints['dPD'] = dPD

    # Protein synthesis PS
    dPS = vPS - FPS
    constraints['dPS'] = dPS

    ##########################################
    ## Additional physiological constraints ##
    ##########################################

    # Alanine to lactate: 50% of Alanine oxidation
    dAL = v_Ala_Lac - 0.5 * vOx['Ala']
    constraints['dAL'] = dAL    

    # Carbohydrate CO2 to 20% of total CO2
    dCO2_carbo = 0.22 * vCO2 - (3*vL + 6*vG)
    constraints['dCO2_carbo'] = dCO2_carbo 

    # Amino Acid Contribution 14% of total CO2
    dCO2_AA = 0.14 * vCO2  - CO2_from_AA
    constraints['dCO2_AA'] = dCO2_AA

    # Total GNG (glucose production)
    total_GNG = v0 + 1/2 * vLip  + vLG + 1/2*v_GNG_Gln + 1/2*v_GNG_Ala
    
    # Glutamine contributes at least 2% to total GNG
    dGNG_Gln_min = 1/2*v_GNG_Gln - 0.05 * total_GNG
    constraints['dGNG_Gln_min'] = dGNG_Gln_min
    
    # Alanine contributes at least 2% to total GNG
    dGNG_Ala_min = 1/2*v_GNG_Ala - 0.05 * total_GNG
    constraints['dGNG_Ala_min'] = dGNG_Ala_min

    # Glycerol contribution of to GNG
    dGNG_GOH_min = 1/2* vLip - 0.3 * total_GNG
    constraints['dGNG_GOH_min'] = dGNG_GOH_min

    # GLycogen contribution to GNG
    dGNG_GLYC_min = v0 - 0.3 * total_GNG
    constraints['dGNG_GLYC_min'] = dGNG_GLYC_min

    # Lacate contribution to GNG
    dGNG_Lac_min = vLG - 0.3 * total_GNG
    constraints['dGNG_Lac_min'] = dGNG_Lac_min

    # 30% of alanine circulatory flux (F_Ala) should be oxidized
    # F_Ala = vN_Lac_Ala + vPD * protein_composition['Ala'] (from dFAL constraint)
    F_Ala_circ = vN_Lac_Ala + vPD * protein_composition['Ala']
    dAla_burn_min = vOx['Ala'] - 0.30 * F_Ala_circ
    constraints['dAla_burn_min'] = dAla_burn_min

    # We need to constraint some minimal oxidation for certain AAs to avoid degenerate solutions

    # Minimal oxidation rate for aspartate (at least 1% of protein degradation flux)
    dAsp_ox_min = vOx['Asp'] - 0.01 * vPD * protein_composition['Asp']
    constraints['dAsp_ox_min'] = dAsp_ox_min

    # Glycine oxidation at least 1% of protein degradation flux
    dGly_ox_min = vOx['Gly'] - 0.01 * vPD * protein_composition['Gly']
    constraints['dGly_ox_min'] = dGly_ox_min

    # Proline oxidation at least 1% of protein degradation flux
    dPro_ox_min = vOx['Pro'] - 0.01 * vPD * protein_composition['Pro']
    constraints['dPro_ox_min'] = dPro_ox_min

    # Minimal glutamate oxidation at 1 % 
    dGlu_ox_min = vOx['Glu'] - 0.01 * vPD * protein_composition['Glu']
    constraints['dGlu_ox_min'] = dGlu_ox_min

    if return_dict:
        return constraints
    else:
        return list(constraints.values())


######################################
# Reference steady state
######################################

def get_reference_steady_state():
    """
    Solve for reference steady state fluxes at baseline conditions.
    Uses constrained optimization to ensure exact satisfaction of core mass balance constraints.
    
    Returns:
    --------
    v_ss: ndarray
        Steady state flux vector that exactly satisfies all core mass balance constraints (dxdt = 0)
    """
    n_fluxes = 11 + 2 + 19 + 2 + 6 + 1  # 41 fluxes total
    v0 = np.ones(n_fluxes) * 0.1
    
    # Step 1: Get least squares solution as initial guess
    from scipy.optimize import least_squares, minimize
    res_ls = least_squares(
        mass_and_energy_constraints, 
        v0, 
        bounds=(np.zeros(n_fluxes), np.full(n_fluxes, np.inf)),
        max_nfev=10000,
        ftol=1e-12,
        xtol=1e-12
    )
    
    if not res_ls.success:
        raise ValueError("Failed to find initial least squares solution")
    
    # Store a copy of the least squares solution for the objective function
    ls_solution = res_ls.x.copy()
    
    # Step 2: Use constrained optimization to find exact steady state
    # Define objective: minimize distance from least squares solution
    def objective_distance(v):
        return np.sum((v - ls_solution)**2)
    
    # Define equality constraints: only core mass balances (dxdt = 0)
    def constraint_mass_balance(v):
        """
        Core mass balance constraints must equal zero at steady state:
        - dLdt, dGdt, dFdt, dKdt (4 metabolites)
        - dAAdt_{aa} (19 amino acids)
        - dCO2 (carbon balance)
        - dN (nitrogen balance)
        - dE (energy balance)
        Total: 26 core constraints (4 minimal oxidation constraints commented out)
        """
        # Get constraints as a dictionary
        constraints_dict = mass_and_energy_constraints(v, return_dict=True)
        
        # Extract only the core mass balance constraints
        core_constraints = [
            constraints_dict['dLdt'],
            constraints_dict['dGdt'],
            constraints_dict['dFdt'],
            constraints_dict['dKdt'],
        ]
        
        # Add amino acid mass balances
        for aa in AMINO_ACIDS:
            core_constraints.append(constraints_dict[f'dAAdt_{aa}'])
        
        # Add CO2, nitrogen, and energy balances
        core_constraints.extend([
            constraints_dict['dCO2'],
            constraints_dict['dN'],
            constraints_dict['dE'],
        ])

        # # Minimal oxidation constraints for aspartate, glutamate, glycine, and proline
        core_constraints.append(constraints_dict['dAsp_ox_min'])
        core_constraints.append(constraints_dict['dGlu_ox_min'])
        core_constraints.append(constraints_dict['dGly_ox_min'])
        core_constraints.append(constraints_dict['dPro_ox_min'])
    
        return np.array(core_constraints)
    
    
    # Set up optimization
    constraints = {
        'type': 'eq',
        'fun': constraint_mass_balance
    }
    
    bounds_opt = [(0, None) for _ in range(n_fluxes)]
    
    # Initial guess: use a copy of least squares solution
    x0_opt = ls_solution.copy()
    
    # Solve constrained optimization
    res_constrained = minimize(
        objective_distance,
        x0_opt,
        method='SLSQP',
        bounds=bounds_opt,
        constraints=constraints,
        options={'maxiter': 10000, 'ftol': 1e-12}
    )
    
    if not res_constrained.success:
        raise ValueError(f"Constrained optimization failed: {res_constrained.message}")
    
    # Verify constraint satisfaction
    residuals = constraint_mass_balance(res_constrained.x)
    max_residual = np.abs(residuals).max()
    
    if max_residual > 1e-6:
        raise ValueError(f"Core constraints not satisfied. Max residual: {max_residual:.2e}")
    
    return res_constrained.x


# Compute and store reference steady state
REF_STEADY_STATE_FLUXES = get_reference_steady_state()

vL_ref_const, vG_ref_const, vF_ref_const, vK_ref_const, vGL_ref_const, vFK_ref_const, vLG_ref_const, v0_ref_const, vLip_ref_const, vReest_ref_const, vCO2_ref_const = REF_STEADY_STATE_FLUXES[0:11]
vPD_ref_const, vPS_ref_const = REF_STEADY_STATE_FLUXES[11:13]
vOx_ref_const = {aa: REF_STEADY_STATE_FLUXES[13 + i] for i, aa in enumerate(AMINO_ACIDS)}
v_GNG_Gln_ref_const = REF_STEADY_STATE_FLUXES[32]
v_GNG_Ala_ref_const = REF_STEADY_STATE_FLUXES[33]

# Extract individual nitrogen recycling fluxes from reference steady state
vN_Glu_Gln_ref_const = REF_STEADY_STATE_FLUXES[34]
vN_Asp_Gln_ref_const = REF_STEADY_STATE_FLUXES[35]
vN_Pro_Gln_ref_const = REF_STEADY_STATE_FLUXES[36]
vN_Gly_Gln_ref_const = REF_STEADY_STATE_FLUXES[37]
vN_Lac_Gln_ref_const = REF_STEADY_STATE_FLUXES[38]
vN_Lac_Ala_ref_const = REF_STEADY_STATE_FLUXES[39]
v_Ala_Lac_ref_const = REF_STEADY_STATE_FLUXES[40]

# Reference time constants (calculated from reference turnover rates)
# These scale the dynamics to match actual metabolic turnover
# Lactate appearance: 2*vGL (glycolysis) + v_Ala_Lac (alanine to lactate)
TAU_L_ref = 1.0 / (2.0 * vGL_ref_const + v_Ala_Lac_ref_const)
# Glucose appearance: v0 (glycogenolysis) + 0.5*vLip (from glycerol) + vLG (from lactate) + 0.5*v_GNG_Gln + 0.5*v_GNG_Ala (from amino acids)
TAU_G_ref = 1.0 / (v0_ref_const + 0.5 * vLip_ref_const + vLG_ref_const + 0.5 * v_GNG_Gln_ref_const + 0.5 * v_GNG_Ala_ref_const)
# Fatty acid appearance: 3*vLip (lipolysis, 3 FFA per TAG)
TAU_F_ref = 1.0 / (3.0 * vLip_ref_const)
# Ketone appearance: 4*vFK (ketogenesis, 4 ketones per 2-carbon units)
TAU_K_ref = 1.0 / (4.0 * vFK_ref_const)

# For amino acids, calculate individual reference time constants based on appearance rates
# Each amino acid has unique turnover based on protein composition and additional pathways
TAU_AA_ref = {}
for aa in AMINO_ACIDS:
    # Base appearance from protein degradation
    gamma_aa = protein_composition[aa]
    appearance_rate = gamma_aa * vPD_ref_const
    
    # Add additional contributions for specific amino acids
    if aa == 'Gln':
        # Glutamine has additional appearance from all nitrogen recycling pathways
        appearance_rate += (vN_Glu_Gln_ref_const + vN_Asp_Gln_ref_const + 
                           vN_Pro_Gln_ref_const + vN_Gly_Gln_ref_const + vN_Lac_Gln_ref_const)
    elif aa == 'Ala':
        # Alanine has additional appearance from lactate-alanine cycle
        appearance_rate += vN_Lac_Ala_ref_const
    
    # TAU_ref = 1 / appearance_rate
    TAU_AA_ref[aa] = 1.0 / appearance_rate if appearance_rate > 0 else 1.0


######################################
# Dynamic model with hormonal regulation
######################################

def fluxes(x, p):
    """
    Compute all metabolic fluxes given state variables and parameters.
    
    State variables (x):
    --------------------
    - L: Lactate concentration
    - G: Glucose concentration
    - F: Fatty acid concentration
    - K: Ketone concentration
    - I: Insulin concentration
    - IA: Active insulin signaling
    - GCG: Glucagon concentration
    - GCGA: Active glucagon signaling
    - 19 amino acid concentrations
    
    Parameters (p):
    ---------------
    See parameter functions below for details.
    
    Returns:
    --------
    Dictionary of all flux values.
    """
    
    # Unpack state variables
    L = x[0]
    G = x[1]
    F = x[2]
    K = x[3]
    I = x[4]
    IA = x[5]
    GCG = x[6]
    GCGA = x[7]
    
    # Amino acid concentrations (19)
    AA_conc = {aa: x[8 + i] for i, aa in enumerate(AMINO_ACIDS)}
    
    # Unpack parameters
    (v_energy, k_glycolysis, k_lactate, k_glucose, k_fatty_acids, k_3HB,
     k_lipolysis, k_reesterification, k_ketogenesis, k_gluconeogenesis,
     V_glycogenolysis, k_protein_degradation, V_protein_synthesis,
     K_i_lipolysis, K_a_glycolysis, K_i_glycogenolysis, K_i_ketogenesis, K_i_protein_degradation,
     K_a_gluconeogenesis, K_a_glycogenolysis,
     I_max, h_ins_param, C_ins_param, GCG_max, h_gcg_param, C_gcg_param,
     R_insulin, R_glucagon, R_lactate, R_glucose, R_fatty_acids, R_3HB,
     k_N_recycling, alpha_N, k_GNG_Gln, k_GNG_Ala, k_AL, A_ref,
     fN_Glu_Gln, fN_Lac_Ala, fN_Lac_Gln, fN_Asp_Gln, fN_Pro_Gln, fN_Gly_Gln) = p[:44]
    
    # Amino acid oxidation rate constants (19)
    k_ox_AA = {aa: p[44 + i] for i, aa in enumerate(AMINO_ACIDS)}
    
    # Unpack action flags (indices 63-69)
    insulin_action_lipolysis = p[63] > 0.5
    insulin_action_glycolysis = p[64] > 0.5
    insulin_action_glycogenolysis = p[65] > 0.5
    insulin_action_ketogenesis = p[66] > 0.5
    insulin_action_protein_degradation = p[67] > 0.5
    glucagon_action_gluconeogenesis = p[68] > 0.5
    glucagon_action_glycogenolysis = p[69] > 0.5
    
    # ========================================
    # Hormonal regulation
    # ========================================
    
    # Insulin action (matching base model formulas)
    if IA > 0:
        f_insulin_lipolysis = (1.0 - IA / (IA + K_i_lipolysis)) if insulin_action_lipolysis else 1.0
        f_insulin_glycolysis = (1.0 + 2.0 * IA / (IA + K_a_glycolysis)) if insulin_action_glycolysis else 1.0
        f_insulin_glycogenolysis = (1.0 - IA / (IA + K_i_glycogenolysis)) if insulin_action_glycogenolysis else 1.0
        f_insulin_ketogenesis = (1.0 - IA / (IA + K_i_ketogenesis)) if insulin_action_ketogenesis else 1.0
        f_insulin_protein_deg = (1.0 - IA / (IA + K_i_protein_degradation)) if insulin_action_protein_degradation else 1.0
    else:
        f_insulin_lipolysis = 1.0
        f_insulin_glycolysis = 1.0
        f_insulin_glycogenolysis = 1.0
        f_insulin_ketogenesis = 1.0
        f_insulin_protein_deg = 1.0
    
    # Glucagon action
    if GCGA > 0:
        f_glucagon_gluconeogenesis = (1.0 + 2.0 * GCGA / (GCGA + K_a_gluconeogenesis)) if glucagon_action_gluconeogenesis else 1.0
        f_glucagon_glycogenolysis = (1.0 + 2.0 * GCGA / (GCGA + K_a_glycogenolysis)) if glucagon_action_glycogenolysis else 1.0
    else:
        f_glucagon_gluconeogenesis = 1.0
        f_glucagon_glycogenolysis = 1.0
    
    # ========================================
    # Competitive catabolism for oxidation
    # ========================================
    
    # Total energy demand from all oxidation pathways
    total_energy_supply = (
        nL * k_lactate * L +
        nG * k_glucose * G +
        nF * k_fatty_acids * F +
        nK * k_3HB * K +
        2 * k_glycolysis * G * f_insulin_glycolysis +
        sum(nAA[aa] * k_ox_AA[aa] * AA_conc[aa] for aa in AMINO_ACIDS)
    )
    
    # Competition factor M
    if total_energy_supply > 0:
        M = v_energy / total_energy_supply
    else:
        M = 0.0
    
    # Oxidation fluxes (competitive)
    vL = k_lactate * M * L
    vG = k_glucose * M * G
    vF = k_fatty_acids * M * F
    vK = k_3HB * M * K
    vGL = k_glycolysis * M * G * f_insulin_glycolysis
    
    # Amino acid oxidation (competitive)
    vOx = {aa: k_ox_AA[aa] * M * AA_conc[aa] for aa in AMINO_ACIDS}
    
    # ========================================
    # Mass action fluxes
    # ========================================
    
    # Lipolysis (regulated by insulin and glucagon)
    vLip = k_lipolysis * A_ref * f_insulin_lipolysis
    
    # Re-esterification (mass action)
    vReest = k_reesterification * F
    
    # Ketogenesis (regulated by insulin)
    vFK = k_ketogenesis * f_insulin_ketogenesis * F
    
    # Gluconeogenesis from lactate (regulated by glucagon)
    vLG = k_gluconeogenesis * f_glucagon_gluconeogenesis * L
    
    # Glycogenolysis (regulated by insulin and glucagon)
    # Note: Insulin inhibits, glucagon activates
    v0 = V_glycogenolysis * f_insulin_glycogenolysis * f_glucagon_glycogenolysis
    
    # Protein degradation (regulated by insulin)
    # Constraint on maximum degradation rate >= synthesis rate
    vPD = k_protein_degradation * f_insulin_protein_deg
    
    # Protein synthesis (constant)
    vPS = V_protein_synthesis
    
    # ========================================
    # Gluconeogenesis from amino acids
    # ========================================
    v_GNG_Gln = k_GNG_Gln * AA_conc['Gln'] * f_glucagon_gluconeogenesis
    v_GNG_Ala = k_GNG_Ala * AA_conc['Ala'] * f_glucagon_gluconeogenesis
    
    # ========================================
    # Nitrogen recycling dynamics (assumes fixes ratios)
    # ========================================
    
    # Total nitrogen production from amino acid oxidation
    N_from_oxidation = sum(nNAA[aa] * vOx[aa] for aa in AMINO_ACIDS)
    
    # Total nitrogen available for recycling or ammonia production
    # Note: Nitrogen from gluconeogenesis is NOT included in local recycling
    vN_total = N_from_oxidation
    
    # Individual pathways as fixed fractions
    # In this simple model of nitrogen balance these fluxes are limited by the overall amino acid avilability
    # e.g. vN_Glu_Gln < (V_Protein_Degradation - V_Protein_Synthesis) * protein_composition['Glu'] 

    vN_Glu_Gln = min(fN_Glu_Gln * vN_total, (vPD - vPS) * protein_composition['Glu'])
    vN_Lac_Ala = min(fN_Lac_Ala * vN_total, (vPD - vPS) * protein_composition['Ala'])
    vN_Lac_Gln = min(fN_Lac_Gln * vN_total, (vPD - vPS) * protein_composition['Gln'])
    vN_Asp_Gln = min(fN_Asp_Gln * vN_total, (vPD - vPS) * protein_composition['Asp'])
    vN_Pro_Gln = min(fN_Pro_Gln * vN_total, (vPD - vPS) * protein_composition['Pro'])
    vN_Gly_Gln = min(fN_Gly_Gln * vN_total, (vPD - vPS) * protein_composition['Gly'])    


    # Alanine to lactate conversion mass action
    v_Ala_Lac = k_AL * AA_conc['Ala']
    
    
    # ========================================
    # Return flux dictionary
    # ========================================
    
    flux_dict = {
        'vL': vL, 'vG': vG, 'vF': vF, 'vK': vK,
        'vGL': vGL, 'vFK': vFK, 'vLG': vLG, 'v0': v0,
        'vLip': vLip, 'vReest': vReest,
        'vPD': vPD, 'vPS': vPS,
        'vN_total': vN_total,
        'vN_Glu_Gln': vN_Glu_Gln, 'vN_Asp_Gln': vN_Asp_Gln, 'vN_Pro_Gln': vN_Pro_Gln,
        'vN_Gly_Gln': vN_Gly_Gln, 'vN_Lac_Gln': vN_Lac_Gln, 'vN_Lac_Ala': vN_Lac_Ala,
        'v_Ala_Lac': v_Ala_Lac,
        'v_GNG_Gln': v_GNG_Gln, 'v_GNG_Ala': v_GNG_Ala,
        'vOx': vOx,
        'M': M,
    }
    
    return flux_dict


def equation(t, x, p):
    """
    ODE system for the extended multi-nutrient model with nitrogen dynamics.
    
    State variables (27 total):
    - L, G, F, K: Core metabolites
    - I, IA: Insulin and active insulin signaling
    - GCG, GCGA: Glucagon and active glucagon signaling
    - 19 amino acids
    
    Returns:
    --------
    dx/dt: Time derivatives of all state variables
    """
    
    # Compute fluxes
    flux = fluxes(x, p)
    
    vL = flux['vL']
    vG = flux['vG']
    vF = flux['vF']
    vK = flux['vK']
    vGL = flux['vGL']
    vFK = flux['vFK']
    vLG = flux['vLG']
    v0 = flux['v0']
    vLip = flux['vLip']
    vReest = flux['vReest']
    vPD = flux['vPD']
    vPS = flux['vPS']
    vN_Glu_Gln = flux['vN_Glu_Gln']
    vN_Asp_Gln = flux['vN_Asp_Gln']
    vN_Pro_Gln = flux['vN_Pro_Gln']
    vN_Gly_Gln = flux['vN_Gly_Gln']
    vN_Lac_Gln = flux['vN_Lac_Gln']
    vN_Lac_Ala = flux['vN_Lac_Ala']
    v_Ala_Lac = flux['v_Ala_Lac']
    v_GNG_Gln = flux['v_GNG_Gln']
    v_GNG_Ala = flux['v_GNG_Ala']
    vOx = flux['vOx']
    
    # Unpack state
    L, G, F, K, I, IA, GCG, GCGA = x[0:8]
    AA_conc = {aa: x[8 + i] for i, aa in enumerate(AMINO_ACIDS)}
    
    # Unpack infusion rates from parameters
    (v_energy, k_glycolysis, k_lactate, k_glucose, k_fatty_acids, k_3HB,
     k_lipolysis, k_reesterification, k_ketogenesis, k_gluconeogenesis,
     V_glycogenolysis, k_protein_degradation, V_protein_synthesis,
     K_i_lipolysis, K_a_glycolysis, K_i_glycogenolysis, K_i_ketogenesis, K_i_protein_degradation,
     K_a_gluconeogenesis, K_a_glycogenolysis,
     I_max, h_ins_param, C_ins_param, GCG_max, h_gcg_param, C_gcg_param,
     R_insulin, R_glucagon, R_lactate, R_glucose, R_fatty_acids, R_3HB,
     k_N_recycling, alpha_N, k_GNG_Gln, k_GNG_Ala, k_AL, A_ref,
     fN_Glu_Gln, fN_Lac_Ala, fN_Lac_Gln, fN_Asp_Gln, fN_Pro_Gln, fN_Gly_Gln) = p[:44]
    
    # Note: Action flags are already unpacked in fluxes() function called below
    
    # ========================================
    # Core metabolite dynamics
    # ========================================
    
    dLdt = (2.0*vGL + v_Ala_Lac - 2.0*vLG - vL - vN_Lac_Ala - vN_Lac_Gln*5/3 + R_lactate)
    dGdt = (v0 + 1/2*(vLip - vReest) + vLG - vGL - vG + 1/2*v_GNG_Gln + 1/2*v_GNG_Ala + R_glucose)
    dFdt = (3.0*(vLip - vReest) - vF - vFK + R_fatty_acids)
    dKdt = (4.0*vFK - vK + R_3HB)
    
    # ========================================
    # Hormonal dynamics
    # ========================================
    
    # Insulin secretion (activated by glucose and amino acids)
    # Amino acids amplify glucose-stimulated insulin secretion

    # Glucose-stimulated insulin secretion
    insulin_secretion_glucose = I_max * G**h_ins_param / (G**h_ins_param + C_ins_param**h_ins_param)
    
    # Amino acid potentiation factor (linear average of insulinotropic amino acids)
    # Only active when there is glucose-stimulated secretion
    aa_potentiation = sum(AA_conc[aa] for aa in INSULINOTROPIC_AA) / len(INSULINOTROPIC_AA)
    
    # Total insulin secretion: glucose-stimulated baseline amplified by amino acids
    insulin_secretion = insulin_secretion_glucose * (1.0 + aa_potentiation)/2.0

    dIdt = (insulin_secretion - I) / TAU_INS + R_insulin
    dIAdt = (I - IA) / TAU_INS_A
    
    # Glucagon secretion (suppressed by glucose, stimulated by amino acids)
    # When G is low → glucagon is high; when G is high → glucagon is low
    # Amino acids (Ala, Arg, Cys, Pro) stimulate glucagon via liver-alpha cell axis

    # Glucose-suppressed glucagon secretion
    glucagon_secretion_glucose = GCG_max * (1.0 - G**h_gcg_param / (G**h_gcg_param + C_gcg_param**h_gcg_param))
    
    # Amino acid stimulation factor (linear average of glucagonotropic amino acids)
    # Ala, Arg, Cys, Pro but NOT Gln
    aa_stimulation = sum(AA_conc[aa] for aa in GLUCAGONOTROPIC_AA) / len(GLUCAGONOTROPIC_AA)
    
    # Total glucagon secretion: glucose-suppressed baseline amplified by amino acids
    glucagon_secretion = glucagon_secretion_glucose * (1.0 + aa_stimulation)/2.0 

    
    dGCGdt = (glucagon_secretion - GCG) / TAU_GCG + R_glucagon
    dGCGAdt = (GCG - GCGA) / TAU_GCG_A
    
    # ========================================
    # Amino acid dynamics
    # ========================================
    
    dAA = {}
    for aa in AMINO_ACIDS:
        gamma_aa = protein_composition[aa]
        dAA[aa] = (gamma_aa * vPD - gamma_aa * vPS - vOx[aa])
    
    # Special handling for amino acids involved in nitrogen recycling
    dAA['Glu'] -= vN_Glu_Gln  # Consumed by N recycling to make glutamine
    dAA['Asp'] -= vN_Asp_Gln  # Consumed by N recycling to make glutamine
    dAA['Pro'] -= vN_Pro_Gln  # Consumed by N recycling to make glutamine
    dAA['Gly'] -= vN_Gly_Gln  # Consumed by N recycling to make glutamine
    dAA['Gln'] += (vN_Glu_Gln + vN_Asp_Gln + vN_Pro_Gln + vN_Gly_Gln + vN_Lac_Gln - v_GNG_Gln)
    dAA['Ala'] += (vN_Lac_Ala - v_GNG_Ala - v_Ala_Lac)
    
    # ========================================
    # Assemble derivative vector with time constant scaling
    # ========================================
    
    # Scale metabolites by their reference time constants (as in base model)
    # This adjusts dynamics to match actual metabolic turnover rates
    dxdt = [
        dLdt / TAU_L * TAU_L_ref,
        dGdt / TAU_G * TAU_G_ref,
        dFdt / TAU_F * TAU_F_ref,
        dKdt / TAU_K * TAU_K_ref,
        dIdt,  # Hormones not scaled
        dIAdt,
        dGCGdt,
        dGCGAdt,
    ]
    
    # Scale amino acids by their individual reference time constants
    # Each amino acid has its own TAU_ref based on appearance rate
    dxdt.extend([dAA[aa] / TAU_AA * TAU_AA_ref[aa] for aa in AMINO_ACIDS])
    
    return np.array(dxdt)


######################################
# Parameter construction
######################################

def ref_parameters(
    K_i_lipolysis=1.0,
    K_a_glycolysis=10.0,
    K_i_glycogenolysis=10.0,
    K_i_ketogenesis=0.2,
    K_i_protein_degradation=20.0,
    K_a_gluconeogenesis=1.0,
    K_a_glycogenolysis=1.0,
    insulin_action_lipolysis=True,
    insulin_action_glycolysis=True,
    insulin_action_glycogenolysis=True,
    insulin_action_ketogenesis=True,
    insulin_action_protein_degradation=True,
    glucagon_action_gluconeogenesis=True,
    glucagon_action_glycogenolysis=True,
):
    """
    Construct reference parameters from steady state solution.
    
    Parameters:
    -----------
    K_i_lipolysis: float
        Insulin inhibition constant for lipolysis (scaled by I0)
    K_a_glycolysis: float
        Insulin activation constant for glycolysis (scaled by I0)
    K_i_glycogenolysis: float
        Insulin inhibition constant for glycogenolysis (scaled by I0)
    K_i_ketogenesis: float
        Insulin inhibition constant for ketogenesis (scaled by I0)
    K_i_protein_degradation: float
        Insulin inhibition constant for protein degradation (scaled by I0)
    K_a_gluconeogenesis: float
        Glucagon activation constant for gluconeogenesis (scaled by GCG0)
    K_a_glycogenolysis: float
        Glucagon activation constant for glycogenolysis (scaled by GCG0)
    insulin_action_*: bool
        Enable/disable insulin regulation for specific processes
    glucagon_action_*: bool
        Enable/disable glucagon regulation for specific processes
    
    Returns:
    --------
    p: Parameter array for dynamic model
    """
    
    # Unpack reference steady state fluxes
    vL_ref, vG_ref, vF_ref, vK_ref, vGL_ref, vFK_ref, vLG_ref, v0_ref, vLip_ref, vReest_ref, vCO2_ref = REF_STEADY_STATE_FLUXES[0:11]
    vPD_ref, vPS_ref = REF_STEADY_STATE_FLUXES[11:13]
    vOx_ref = {aa: REF_STEADY_STATE_FLUXES[13 + i] for i, aa in enumerate(AMINO_ACIDS)}
    v_GNG_Gln_ref = REF_STEADY_STATE_FLUXES[32]
    v_GNG_Ala_ref = REF_STEADY_STATE_FLUXES[33]
    
    # Extract individual nitrogen recycling fluxes from reference steady state
    vN_Glu_Gln_ref = REF_STEADY_STATE_FLUXES[34]
    vN_Asp_Gln_ref = REF_STEADY_STATE_FLUXES[35]
    vN_Pro_Gln_ref = REF_STEADY_STATE_FLUXES[36]
    vN_Gly_Gln_ref = REF_STEADY_STATE_FLUXES[37]
    vN_Lac_Gln_ref = REF_STEADY_STATE_FLUXES[38]
    vN_Lac_Ala_ref = REF_STEADY_STATE_FLUXES[39]
    v_Ala_Lac_ref = REF_STEADY_STATE_FLUXES[40]
    
    # Use the global fractions computed from reference steady state
    # (These are defined after REF_STEADY_STATE_FLUXES is computed)
    
    # Energy expenditure
    v_energy = 1.0
    
    # Compute regulatory factors at steady state (IA = I0, GCGA = GCG0)
    # These will be used to back-calculate mass action constants
    # Factors are computed conditionally based on action flags

    #Hormonal regulation constants (scaled by reference hormone levels)
    # These are defined before computing regulatory factors to allow user customization
    K_i_lipolysis_scaled = I0 * K_i_lipolysis
    K_a_glycolysis_scaled = I0 * K_a_glycolysis
    K_i_glycogenolysis_scaled = I0 * K_i_glycogenolysis
    K_i_ketogenesis_scaled = I0 * K_i_ketogenesis
    K_i_protein_degradation_scaled = I0 * K_i_protein_degradation
    
    K_a_gluconeogenesis_scaled = GCG0 * K_a_gluconeogenesis
    K_a_glycogenolysis_scaled = GCG0 * K_a_glycogenolysis
    
    # Insulin parameters
    I_max = 1.0
    h_ins_param = h_ins
    C_ins_param = C_ins
    
    # Glucagon parameters
    GCG_max = 1.0
    h_gcg_param = h_gcg
    C_gcg_param = C_gcg
    
    # Insulin inhibition factors at IA = I0
    if insulin_action_lipolysis:
        f_insulin_lipolysis_ref = 1.0 - I0 / (I0 + K_i_lipolysis_scaled)
    else:
        f_insulin_lipolysis_ref = 1.0
    
    if insulin_action_glycolysis:
        f_insulin_glycolysis_ref = 1.0 + 2.0 * I0 / (I0 + K_a_glycolysis_scaled)
    else:
        f_insulin_glycolysis_ref = 1.0
    
    if insulin_action_glycogenolysis:
        f_insulin_glycogenolysis_ref = 1.0 - I0 / (I0 + K_i_glycogenolysis_scaled)
    else:
        f_insulin_glycogenolysis_ref = 1.0
    
    if insulin_action_ketogenesis:
        f_insulin_ketogenesis_ref = 1.0 - I0 / (I0 + K_i_ketogenesis_scaled)
    else:
        f_insulin_ketogenesis_ref = 1.0
    
    if insulin_action_protein_degradation:
        f_insulin_protein_deg_ref = 1.0 - I0 / (I0 + K_i_protein_degradation_scaled)
    else:
        f_insulin_protein_deg_ref = 1.0
    
    # Glucagon activation factors at GCGA = GCG0
    if glucagon_action_gluconeogenesis:
        f_glucagon_gluconeogenesis_ref = 1.0 + 2.0 * GCG0 / (GCG0 + K_a_gluconeogenesis_scaled)
    else:
        f_glucagon_gluconeogenesis_ref = 1.0
    
    if glucagon_action_glycogenolysis:
        f_glucagon_glycogenolysis_ref = 1.0 + 2.0 * GCG0 / (GCG0 + K_a_glycogenolysis_scaled)
    else:
        f_glucagon_glycogenolysis_ref = 1.0
    
    # Calculate M at reference steady state
    M_ref = 1.0  # By construction from the energy balance constraint
    
    # Rate constants - divide reference fluxes by M_ref, regulatory factors, and concentrations
    k_glycolysis = vGL_ref / (M_ref * 1.0 * f_insulin_glycolysis_ref)  # vGL = k * M * G * f_insulin
    k_lactate = vL_ref / (M_ref * 1.0)  # vL = k * M * L
    k_glucose = vG_ref / (M_ref * 1.0)  # vG = k * M * G
    k_fatty_acids = vF_ref / (M_ref * 1.0)  # vF = k * M * F
    k_3HB = vK_ref / (M_ref * 1.0)  # vK = k * M * K
    
    # Amino acid oxidation rate constants (competitive)
    k_ox_AA = {aa: vOx_ref[aa] / (M_ref * 1.0) for aa in AMINO_ACIDS}  # vOx = k * M * AA
    
    # Mass action rate constants (non-competitive fluxes)
    k_lipolysis = vLip_ref / (1.0 * f_insulin_lipolysis_ref)  # vLip = k * A * f_insulin
    k_reesterification = vReest_ref / 1.0  # vReest = k * F
    k_ketogenesis = vFK_ref / f_insulin_ketogenesis_ref  # vFK = k * f_insulin * F
    k_gluconeogenesis = vLG_ref / f_glucagon_gluconeogenesis_ref  # vLG = k * f_glucagon * L
    V_glycogenolysis = v0_ref / (f_insulin_glycogenolysis_ref * f_glucagon_glycogenolysis_ref)  # v0 = V * f_insulin * f_glucagon
    
    k_protein_degradation = vPD_ref / f_insulin_protein_deg_ref  # vPD = k * f_insulin
    V_protein_synthesis = vPS_ref  # vPS = V (constant)

    
    # Infusion rates (zero at baseline)
    R_insulin = 0.0
    R_glucagon = 0.0
    R_lactate = 0.0
    R_glucose = 0.0
    R_fatty_acids = 0.0
    R_3HB = 0.0
    
    # Nitrogen recycling (legacy parameters for backward compatibility)
    # Map new fluxes to old naming: vN2 -> vN_Lac_Ala, vN3 -> vN_Lac_Gln
    k_N_recycling = (vN_Lac_Ala_ref + vN_Lac_Gln_ref) / 1.0  # Assumes L=1
    if vN_Lac_Ala_ref + vN_Lac_Gln_ref > 0:
        alpha_N = vN_Lac_Ala_ref / (vN_Lac_Ala_ref + vN_Lac_Gln_ref)
    else:
        alpha_N = 0.5
    
    # AA gluconeogenesis - separate rate constants for Gln and Ala
    k_GNG_Gln = v_GNG_Gln_ref / (1.0 * f_glucagon_gluconeogenesis_ref)  # Gln conc = 1, divide by f_glucagon
    k_GNG_Ala = v_GNG_Ala_ref / (1.0 * f_glucagon_gluconeogenesis_ref)  # Ala conc = 1, divide by f_glucagon
    
    # Simplified nitrogen recycling rate constants
    vN_total = sum(nNAA[aa] * vOx_ref[aa] for aa in AMINO_ACIDS)

    fN_Glu_Gln = vN_Glu_Gln_ref / vN_total 
    fN_Lac_Ala = vN_Lac_Ala_ref / vN_total 
    fN_Lac_Gln = 2 * vN_Lac_Gln_ref / vN_total 
    fN_Asp_Gln = vN_Asp_Gln_ref / vN_total 
    fN_Pro_Gln = vN_Pro_Gln_ref / vN_total 
    fN_Gly_Gln = vN_Gly_Gln_ref / vN_total 
   
    # Alanine to lactate conversion rate constant
    k_AL = v_Ala_Lac_ref / 1.0
    
    # Reference adipose mass (constant)
    A_ref = 1.0
    
    # Assemble parameter array
    p = [
        v_energy, k_glycolysis, k_lactate, k_glucose, k_fatty_acids, k_3HB,
        k_lipolysis, k_reesterification, k_ketogenesis, k_gluconeogenesis,
        V_glycogenolysis, k_protein_degradation, V_protein_synthesis,
        K_i_lipolysis_scaled, K_a_glycolysis_scaled, K_i_glycogenolysis_scaled, K_i_ketogenesis_scaled, K_i_protein_degradation_scaled,
        K_a_gluconeogenesis_scaled, K_a_glycogenolysis_scaled,
        I_max, h_ins_param, C_ins_param, GCG_max, h_gcg_param, C_gcg_param,
        R_insulin, R_glucagon, R_lactate, R_glucose, R_fatty_acids, R_3HB,
        k_N_recycling, alpha_N, k_GNG_Gln, k_GNG_Ala, k_AL, A_ref,
        fN_Glu_Gln, fN_Lac_Ala, fN_Lac_Gln, fN_Asp_Gln, fN_Pro_Gln, fN_Gly_Gln,
    ]
    
    # Add amino acid oxidation constants
    p.extend([k_ox_AA[aa] for aa in AMINO_ACIDS])
    
    # Add action flags (indices 61-67)
    p.extend([
        1.0 if insulin_action_lipolysis else 0.0,
        1.0 if insulin_action_glycolysis else 0.0,
        1.0 if insulin_action_glycogenolysis else 0.0,
        1.0 if insulin_action_ketogenesis else 0.0,
        1.0 if insulin_action_protein_degradation else 0.0,
        1.0 if glucagon_action_gluconeogenesis else 0.0,
        1.0 if glucagon_action_glycogenolysis else 0.0,
    ])
    
    return np.array(p)


def initial_state():
    """
    Return initial state vector at reference steady state (all concentrations = 1.0).
    """
    x0 = [1.0, 1.0, 1.0, 1.0, I0, I0, GCG0, GCG0]  # L, G, F, K, I, IA, GCG, GCGA
    x0.extend([1.0] * 19)  # Amino acids
    return np.array(x0)


def steady_state(p, x0=None, **kwargs):
    """
    Solve for steady state given parameters.
    
    Parameters:
    -----------
    p: Parameter array
    x0: Initial guess (if None, uses default)
    
    Returns:
    --------
    x_ss: Steady state concentrations
    """
    if x0 is None:
        x0 = initial_state()
    
    def ss_equations(x):
        return equation(0, x, p)
    
    x_ss = fsolve(ss_equations, x0, **kwargs)
    
    # Check if solution is valid
    residual = ss_equations(x_ss)
    if np.allclose(residual, 0, atol=1e-6) and np.all(x_ss >= 0):
        return x_ss
    else:
        return np.nan * np.ones_like(x0)


######################################
# Dynamic simulation utilities
######################################

def change_parameters(p, param_dict):
    """
    Change parameters by name using PARAMETER_NAMES list.
    
    Parameters:
    -----------
    p: array-like
        Parameter array
    param_dict: dict
        Dictionary of parameter names and values to change
        
    Returns:
    --------
    p_new: ndarray
        Modified parameter array
    """
    p_new = np.array(p).copy()
    
    for param_name, value in param_dict.items():
        if param_name in PARAMETER_NAMES:
            idx = PARAMETER_NAMES.index(param_name)
            p_new[idx] = value
        else:
            raise ValueError(f"Unknown parameter: {param_name}. Valid parameters: {PARAMETER_NAMES}")
    
    return p_new


def perturbation_dynamics(time, x0=None, p=None, **kwargs):
    """
    Simulate the time response to a parameter perturbation.
    
    Parameters:
    -----------
    time: array-like
        Time points for simulation
    x0: array-like, optional
        Initial state. If None, uses steady state from reference parameters
    p: array-like, optional
        Base parameters. If None, uses reference parameters
    **kwargs:
        Parameter perturbations as keyword arguments
        Example: R_glucose=0.5, K_i_lipolysis=2.0
        
    Returns:
    --------
    X: pd.DataFrame
        State variables over time (L, G, F, K, I, IA, GCG, GCGA, amino acids)
    F: pd.DataFrame
        Fluxes over time
    """
    # Unpack parameters
    if p is None:
        if kwargs == {}:
            p_ref = ref_parameters()
            p = p_ref.copy()
        else:
            p_ref = ref_parameters()
            p = change_parameters(p_ref, kwargs)
    else:
        if kwargs != {}:
            p_ref = p.copy()
            p = change_parameters(p, kwargs)
        else:
            p_ref = p.copy()
    
    # Get steady state from reference parameter
    if x0 is None:
        x0 = steady_state(p_ref)
    
    # Wrapper for solve_ivp (t, x order)
    def ode_func(t, x):
        return equation(t, x, p)
    
    # Simulate with BDF method
    sol = solve_ivp(ode_func, (time[0], time[-1]), x0, method='BDF',
                    t_eval=time, rtol=1e-6)
    sol_X = sol.y.T
    
    # Export to a pandas dataframe
    state_names = ['L', 'G', 'F', 'K', 'I', 'IA', 'GCG', 'GCGA'] + AMINO_ACIDS
    X = pd.DataFrame(sol_X, columns=state_names)
    X['time'] = time
    
    # Compute fluxes
    flux_list = [fluxes(x, p) for x in sol_X]
    
    # Extract core fluxes
    F_data = []
    for flux_dict in flux_list:
        row = {
            'vL': flux_dict['vL'],
            'vG': flux_dict['vG'],
            'vF': flux_dict['vF'],
            'vK': flux_dict['vK'],
            'vGL': flux_dict['vGL'],
            'vFK': flux_dict['vFK'],
            'vLG': flux_dict['vLG'],
            'v0': flux_dict['v0'],
            'vLip': flux_dict['vLip'],
            'vReest': flux_dict['vReest'],
            'vPD': flux_dict['vPD'],
            'vPS': flux_dict['vPS'],
            'vN_total': flux_dict['vN_total'],
            'vN_Glu_Gln': flux_dict['vN_Glu_Gln'],
            'vN_Asp_Gln': flux_dict['vN_Asp_Gln'],
            'vN_Pro_Gln': flux_dict['vN_Pro_Gln'],
            'vN_Gly_Gln': flux_dict['vN_Gly_Gln'],
            'vN_Lac_Gln': flux_dict['vN_Lac_Gln'],
            'vN_Lac_Ala': flux_dict['vN_Lac_Ala'],
            'v_Ala_Lac': flux_dict['v_Ala_Lac'],
            'v_GNG_Gln': flux_dict['v_GNG_Gln'],
            'v_GNG_Ala': flux_dict['v_GNG_Ala'],
            'M': flux_dict['M'],
        }
        F_data.append(row)
    
    F = pd.DataFrame(F_data)
    F['time'] = time
    
    return X, F


def insulin_clamp_dynamic(insulin_level, time, x0=None, p=None, **kwargs):
    """
    Simulate hyperinsulinemic-euglycemic clamp.
    
    Parameters:
    -----------
    insulin_level: float
        Insulin infusion rate (dimensionless)
    time: array-like
        Time points (minutes)
    x0: array-like, optional
        Initial state
    p: array-like, optional
        Base parameters
    **kwargs:
        Additional parameter perturbations
        
    Returns:
    --------
    X: pd.DataFrame
        State variables over time
    GIR: pd.DataFrame
        Glucose infusion rate over time
    """
    # Unpack parameters 
    if p is None:
        if kwargs == {}:
            p = ref_parameters()
        else:
            p = change_parameters(ref_parameters(), kwargs)
    else:
        if kwargs != {}:
            p = change_parameters(p, kwargs)
        
    # Get steady state from parameter perturbation
    if x0 is None:
        x0 = steady_state(p)

    # Glucose infusion rate to maintain euglycemia (P control)
    GIR_func = lambda x: np.clip(1.0 - x[1], 0.0, np.inf)
    
    # Modify equation to include glucose infusion
    def euglycemic_clamp(t, x, p):
        dxdt = equation(t, x, p)
        dxdt[1] += GIR_func(x) # Add glucose infusion to glucose equation
        return dxdt
    
    # Set insulin infusion
    p_ins = p.copy()
    p_ins[26] = insulin_level  # R_insulin

    # Solve ODE with BDF method
    sol = solve_ivp(euglycemic_clamp, (time[0], time[-1]), x0, method='BDF',
                    t_eval=time, args=(p_ins,), rtol=1e-12)
    sol_X = sol.y.T
    
    # Compute the glucose infusion as the rate needed to maintain euglycemia
    sol_GIR = np.array([GIR_func(x) for x in sol_X])

    # Export to a pandas dataframe
    state_names = ['L', 'G', 'F', 'K', 'I', 'IA', 'GCG', 'GCGA'] + AMINO_ACIDS
    X = pd.DataFrame(sol_X, columns=state_names)
    X['time'] = time

    GIR_df = pd.DataFrame(sol_GIR, columns=['GIR'])
    GIR_df['time'] = time

    return X, GIR_df
