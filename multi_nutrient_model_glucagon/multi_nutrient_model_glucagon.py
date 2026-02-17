import numpy as np 
import pandas as pd
from scipy.optimize import fsolve
from scipy.integrate import odeint

import warnings
warnings.filterwarnings('ignore')


######################################
# Constants and parameters
######################################

# FIXED RATIOS (Energy per molecule rel to glucose 30 ATP per glucose)
nG = 32
nL = 15
nF = 108
nK = 22.5

# Insulin degradation time constants
TAU_INS = 2  # Insulin degradation time constant
TAU_INS_A = 30  # Insulin signaling time constant

# Glucagon time constants
TAU_GCG = 5  # Glucagon degradation time constant
TAU_GCG_A = 30  # Glucagon signaling time constant

# Time constants from volume distribution
TAU_L = 5.0  # 5 min
TAU_F = 6.5  # 6.5 min
TAU_G = 21.0  # 21 min
TAU_K = 3.0  # 3 min

# Insulin secretion parameters
h = 3.4
C = 2.3
# Ref. insulin at G=1
I0 = abs(1.0)**h / (abs(1.0)**h + C**h)

# Glucagon secretion parameters (fitted from BIC model comparison - M2: Hill)
h_gcg = 1.95  # Glucagon secretion Hill coefficient
C_gcg = 0.41  # Glucagon secretion threshold (relative to glucose)
GCG_max = 1.0  # Glucagon secretion capacity
# Ref. glucagon at G=1 (glucagon is inversely related to glucose)
GCG0 = GCG_max * (1.0 - 1.0**h_gcg / (1.0**h_gcg + C_gcg**h_gcg))


######################################
# Steady state analysis 
######################################

# Scale the fluxes by v_energy -> ATP flux 

# ATP per O2 (3 per O)
PO2 = 5.0
# Whole body oxygen consumption rate ~ 2000 nmol/min/gBW
vO2 = 2000
# ATP production rate
vATP = PO2 * vO2 * 0.75
vE = vATP

def mass_and_energy_constraints(v, v_energy=1.0, 
                                FG = 100/vATP, 
                                FL = 150/vATP, 
                                FK = 30/vATP,
                                FF = 150/vATP,
                                ):
    vL, vG, vF, vK, vGL, vFK, vLG, v0, vA, vR, vCO2 = v

    dLdt = 2.0*vGL - 2.0*vLG - vL
    dGdt = v0 + 1/2*(vA - vR) + vLG - vGL - vG
    dFdt = 3.0*(vA-vR) - vF - vFK 
    dKdt = 4.0*vFK - vK
    # CO2 = balance 
    dCO2 = 3 * vL + 6 * vG + 16 * vF + 4 * vK - vCO2
    # Constraint energy expenditure
    dE = nL * vL + nG * vG + nF * vF + nK * vK + 2 * vGL - v_energy

    # ADDITIONAL CONSTRAINTS
    dGLY1 = vLG - 1/2 * vA  # Equal contribution of glycogen and gluconeogenesis to EGP

    # Reesterification constraint (2/3 of the FFA is reesterified 1/3 oxidized)
    dR = vR - 2/3 * vA

    # Try by constraining lactate and glucose Fcircs scaled by energy expenditure
    dDG = vGL + vG + 0.5 * vA - FG
    dDL = 2 * vLG + vL - FL

    dDF = vFK + vF + 3 * vR - FF
    dDK = vK - FK

    return [dLdt, dGdt, dFdt, dKdt, dGLY1, dR, dCO2, dE, dDF, dDL, dDK]


######################################
# Reference steady state values
######################################

# vL, vG, vF, vK, vGL, vFK, vLG, v0, vA, vR, vCO2 = v
v0 = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
REF_STEADY_STATE_VALUES = fsolve(mass_and_energy_constraints, v0)
vL_ref, vG_ref, vF_ref, vK_red, vGL_ref, vFK_ref, vLG_ref, v0_ref, vA_ref, vR_ref, vCO2 = REF_STEADY_STATE_VALUES

# Reference turnovers calculated based on appearance rates
TAU_F_ref = 1/vA_ref
TAU_G_ref = 1/(v0_ref + 1/2*vA_ref + vLG_ref)
TAU_L_ref = 1/(2*vGL_ref)
TAU_K_ref = 1/(4*vFK_ref)


######################################
# Parameter names and descriptions
######################################

PARAMETER_NAMES = [
    "v_energy", "h", "I_max", "C", 
    "K_i_lipolysis", "K_a_glycolysis", "K_i_glycogenolysis", "K_i_ketogenesis",
    "k_glycolysis", "k_lactate", "k_glucose", "k_fatty_acids", "k_3HB", 
    "k_lipolysis", "k_reesterification", "k_ketogenesis", "k_gluconeogenesis", 
    "V_glycogenolysis", 
    "R_insulin", "R_glucagon", "R_lactate", "R_glucose", "R_fatty_acids", "R_3HB", 
    "insulin_action_lipolysis", "insulin_action_glycolysis", "insulin_action_glycogenolysis", "insulin_action_ketogenesis",
    "hyperplasia",
    # Glucagon parameters
    "h_gcg", "GCG_max", "C_gcg",
    "K_a_gluconeogenesis", "K_a_glycogenolysis", "K_a_ketogenesis",
    "A_gcg_gluconeogenesis", "A_gcg_glycogenolysis", "A_gcg_ketogenesis",
    "glucagon_action_gluconeogenesis", "glucagon_action_glycogenolysis", "glucagon_action_ketogenesis",
]

PARAMETER_DESCRIPTIONS = {
    "v_energy": "Energy expenditure",
    "h": "Insulin Hill coefficient",
    "I_max": "Insulin secretion capacity",
    "C": "Insulin secretion threshold (Glucose)", 
    "K_i_lipolysis": "Insulin inhibition of lipolysis",
    "K_i_ketogenesis": "Insulin inhibition of ketogenesis",
    "K_a_glycolysis": "Insulin activation of glycolysis", 
    "K_i_glycogenolysis": "Insulin inhibition of glycogenolysis",
    "k_glycolysis": "Glycolysis", 
    "k_lactate": "Lactate oxidation", 
    "k_glucose": "Glucose oxidation", 
    "k_fatty_acids": "Fatty acid oxidation", 
    "k_3HB": "3HB oxidation", 
    "k_lipolysis": "Adipose lipolysis",
    "k_reesterification": "Reesterification",  
    "k_ketogenesis": "Ketogenesis",
    "k_gluconeogenesis": "Gluconeogenesis", 
    "V_glycogenolysis": "Glycogenolysis",
    "R_insulin": "Insulin infusion rate",
    "R_glucagon": "Glucagon infusion rate",
    "R_lactate": "Lactate infusion rate", 
    "R_glucose": "Glucose infusion rate", 
    "R_fatty_acids": "Fatty acid infusion rate", 
    "R_3HB": "3HB infusion rate", 
    "insulin_action_lipolysis": "Insulin action on lipolysis",
    "insulin_action_glycolysis": "Insulin action glycolysis",
    "insulin_action_glycogenolysis": "Insulin action glycogenolysis",
    "insulin_action_ketogenesis": "Insulin action ketogenesis",
    "hyperplasia": "Adipose hyperplasia",
    # Glucagon parameters
    "h_gcg": "Glucagon Hill coefficient",
    "GCG_max": "Glucagon secretion capacity",
    "C_gcg": "Glucagon secretion threshold (Glucose)",
    "K_a_gluconeogenesis": "Glucagon activation of gluconeogenesis",
    "K_a_glycogenolysis": "Glucagon activation of glycogenolysis",
    "K_a_ketogenesis": "Glucagon activation of ketogenesis",
    "A_gcg_gluconeogenesis": "Max glucagon activation of gluconeogenesis",
    "A_gcg_glycogenolysis": "Max glucagon activation of glycogenolysis",
    "A_gcg_ketogenesis": "Max glucagon activation of ketogenesis",
    "glucagon_action_gluconeogenesis": "Glucagon action gluconeogenesis",
    "glucagon_action_glycogenolysis": "Glucagon action glycogenolysis",
    "glucagon_action_ketogenesis": "Glucagon action ketogenesis",
}

PARAMETER_LATEX = {
    "v_energy": "v_energy",
    "h": "h",
    "I_max": "I_max",
    "C": "C",
    "K_i_lipolysis": "K_i,lipolysis",
    "K_i_ketogenesis": "K_i,ketogenesis",
    "K_a_glycolysis": "K_a,glycolysis",
    "K_i_glycogenolysis": "K_i,glycogenolysis",
    "k_glycolysis": "k_glycolysis",
    "k_lactate": "k_lactate",
    "k_glucose": "k_glucose",
    "k_fatty_acids": "k_fatty acids",
    "k_3HB": "k_3HB",
    "k_lipolysis": "k_lipolysis",
    "k_reesterification": "k_reesterification",
    "k_ketogenesis": "k_ketogenesis",
    "k_gluconeogenesis": "k_gluconeogenesis",
    "V_glycogenolysis": "V_glycogenolysis",
    "R_insulin": "R_insulin",
    "R_glucagon": "R_glucagon",
    "R_lactate": "R_lactate",
    "R_glucose": "R_glucose",
    "R_fatty_acids": "R_fatty acids",
    "R_3HB": "R_3HB",
    "insulin_action_lipolysis": "on/off",
    "insulin_action_glycolysis": "on/off",
    "insulin_action_glycogenolysis": "on/off",
    "insulin_action_ketogenesis": "on/off",
    "hyperplasia": "on/off",
    # Glucagon parameters
    "h_gcg": "h_gcg",
    "GCG_max": "GCG_max",
    "C_gcg": "C_gcg",
    "K_a_gluconeogenesis": "K_a,gluconeogenesis",
    "K_a_glycogenolysis": "K_a,glycogenolysis",
    "K_a_ketogenesis": "K_a,ketogenesis",
    "A_gcg_gluconeogenesis": "A_{gcg,GNG}",
    "A_gcg_glycogenolysis": "A_{gcg,GLY}",
    "A_gcg_ketogenesis": "A_{gcg,KG}",
    "glucagon_action_gluconeogenesis": "on/off",
    "glucagon_action_glycogenolysis": "on/off",
    "glucagon_action_ketogenesis": "on/off",
}


######################################
# Model equations
######################################

def fluxes(x, A, p):
    """
    Compute the fluxes of the model given the concentrations and parameters
    """
    
    L, G, F, K, I, IA, GCG, GCGA = x

    v_energy, h, I_max, C, \
    K_i_lipolysis, K_a_glycolysis, K_i_glycogenolysis, K_i_ketogenesis, \
    k_glycolysis, k_lactate, k_glucose, k_fatty_acids, k_3HB, \
    k_lipolysis, k_reesterification, k_ketogenesis, k_gluconeogenesis, \
    V_glycogenolysis, \
    R_insulin, R_glucagon, R_lactate, R_glucose, R_fatty_acids, R_3HB, \
    insulin_action_lipolysis, insulin_action_glycolysis, insulin_action_glycogenolysis, insulin_action_ketogenesis, \
    hyperplasia, \
    h_gcg_param, GCG_max_param, C_gcg_param, \
    K_a_gluconeogenesis, K_a_glycogenolysis, K_a_ketogenesis, \
    A_gcg_gluconeogenesis, A_gcg_glycogenolysis, A_gcg_ketogenesis, \
    glucagon_action_gluconeogenesis, glucagon_action_glycogenolysis, glucagon_action_ketogenesis = p
    
    # Insulin dynamics
    vI = (I_max * abs(G)**h / (abs(G)**h + C**h) - I) / TAU_INS + R_insulin
    vIA = (I - IA) / TAU_INS_A

    # Glucagon dynamics (inversely related to glucose)
    glucagon_secretion = GCG_max_param * (1.0 - abs(G)**h_gcg_param / (abs(G)**h_gcg_param + C_gcg_param**h_gcg_param))
    vGCG = (glucagon_secretion - GCG) / TAU_GCG + R_glucagon
    vGCGA = (GCG - GCGA) / TAU_GCG_A

    # Insulin action on lipolysis
    if insulin_action_lipolysis:
        LI = 1.0 - IA / (IA + K_i_lipolysis) 
    else:
        LI = 1.0

    # Insulin action on glucose oxidation 
    if insulin_action_glycolysis:
        SI = 1 + 2.0 * IA / (IA + K_a_glycolysis)
    else:
        SI = 1.0
        
    # Insulin action on ketone production
    if insulin_action_ketogenesis:
        FI = 1.0 - IA / (IA + K_i_ketogenesis) 
    else:
        FI = 1.0

    # Insulin action on glycogen breakdown
    if insulin_action_glycogenolysis:
        GI = 1.0 - IA / (IA + K_i_glycogenolysis) 
    else:
        GI = 1.0

    # Glucagon action on gluconeogenesis
    if glucagon_action_gluconeogenesis:
        GCG_GNG = 1.0 + A_gcg_gluconeogenesis * GCGA / (GCGA + K_a_gluconeogenesis)
    else:
        GCG_GNG = 1.0

    # Glucagon action on glycogenolysis
    if glucagon_action_glycogenolysis:
        GCG_GLY = 1.0 + A_gcg_glycogenolysis * GCGA / (GCGA + K_a_glycogenolysis)
    else:
        GCG_GLY = 1.0

    # Glucagon action on ketogenesis
    if glucagon_action_ketogenesis:
        GCG_KG = 1.0 + A_gcg_ketogenesis * GCGA / (GCGA + K_a_ketogenesis)
    else:
        GCG_KG = 1.0

    # Competitive oxidation
    M = v_energy / (nL * k_lactate * L 
                    + nG * k_glucose * G
                    + nF * k_fatty_acids * F
                    + nK * k_3HB * K
                    + 2 * k_glycolysis * G * SI)
    
    # Glycolysis
    vGL = k_glycolysis * M * G * SI

    vG = k_glucose * M * G
    vL = k_lactate * M * L
    vF = k_fatty_acids * M * F
    vK = k_3HB * M * K

    # Lipolysis - regulated by insulin
    vA = k_lipolysis * A * LI

    if hyperplasia:
        vR = k_reesterification * F * A
    else:
        vR = k_reesterification * F
    
    # Ketogenesis - regulated by insulin and glucagon
    vFK = k_ketogenesis * FI * GCG_KG * F
    
    # Gluconeogenesis - regulated by glucagon
    vLG = k_gluconeogenesis * GCG_GNG * L 
    
    # Glycogenolysis - regulated by insulin and glucagon
    v0 = V_glycogenolysis * GI * GCG_GLY

    return np.array([vL, vG, vF, vK, vGL, vFK, vLG, v0, vA, vR,
                     R_lactate, R_glucose, R_fatty_acids, R_3HB, vI, vIA, vGCG, vGCGA])
    

def equation(x, A, p):
    """
    ODE system for the multi-nutrient model with glucagon.
    
    State variables (8 total):
    - L: Lactate concentration
    - G: Glucose concentration
    - F: Fatty acid concentration
    - K: Ketone concentration
    - I: Insulin concentration
    - IA: Active insulin signaling
    - GCG: Glucagon concentration
    - GCGA: Active glucagon signaling
    """

    vL, vG, vF, vK, vGL, vFK, vLG, v0, vA, vR, \
    R_lactate, R_glucose, R_fatty_acids, R_3HB, vI, vIA, vGCG, vGCGA = fluxes(x, A, p) 

    dLdt = 2.0*vGL - 2.0*vLG - vL + R_lactate
    dGdt = v0 + 1/2*(vA - vR) + vLG - vGL - vG + R_glucose
    dFdt = 3.0*(vA - vR) - vF - vFK + R_fatty_acids
    dKdt = 4.0*vFK - vK + R_3HB
    dIdt = vI
    dIAdt = vIA
    dGCGdt = vGCG
    dGCGAdt = vGCGA
    
    # Scale the dynamic equation using the respective time constants 
    # as determined by the distribution of volume experiments
    # NOTE This only affects the time dynamics not the steady state

    return [dLdt/TAU_L*TAU_L_ref, 
            dGdt/TAU_G*TAU_G_ref, 
            dFdt/TAU_F*TAU_F_ref, 
            dKdt/TAU_K*TAU_K_ref, 
            dIdt,
            dIAdt,
            dGCGdt,
            dGCGAdt]


def initial_state():
    """
    Return initial state vector at reference steady state.
    """
    return np.array([1.0, 1.0, 1.0, 1.0, I0, I0, GCG0, GCG0])


def steady_state(A, p, x0=None, **kwargs):
    """
    Solve for steady state given parameters.
    """
    if x0 is None:
        x0 = initial_state()
    
    x = fsolve(equation, x0, args=(A, p), **kwargs) 
    if all(np.isclose(equation(x, A, p), np.zeros_like(x))):
        if all(x >= 0):
            return np.array(x)
        else:
            return np.nan * np.ones_like(x)
    else:
        return np.nan * np.ones_like(x)


# Competitive catabolism model (refactored to match `fluxes` conventions)
def competitive_oxidation(x, p):
    """
    Compute only the competitive oxidation fluxes.
    """
    # Unpack concentrations
    L, G, F, K, IA = x

    # Unpack parameters
    v_energy, h, I_max, C, \
    K_i_lipolysis, K_a_glycolysis, K_i_glycogenolysis, K_i_ketogenesis, \
    k_glycolysis, k_lactate, k_glucose, k_fatty_acids, k_3HB, \
    k_lipolysis, k_reesterification, k_ketogenesis, k_gluconeogenesis, \
    V_glycogenolysis, \
    R_insulin, R_glucagon, R_lactate, R_glucose, R_fatty_acids, R_3HB, \
    insulin_action_lipolysis, insulin_action_glycolysis, insulin_action_glycogenolysis, insulin_action_ketogenesis, \
    hyperplasia, \
    h_gcg_param, GCG_max_param, C_gcg_param, \
    K_a_gluconeogenesis, K_a_glycogenolysis, K_a_ketogenesis, \
    A_gcg_gluconeogenesis, A_gcg_glycogenolysis, A_gcg_ketogenesis, \
    glucagon_action_gluconeogenesis, glucagon_action_glycogenolysis, glucagon_action_ketogenesis = p

    # Insulin action on glucose oxidation
    if insulin_action_glycolysis:
        SI = 1.0 + 2.0 * IA / (IA + K_a_glycolysis)
    else:
        SI = 1.0

    # Competitive oxidation
    M = v_energy / (nL * k_lactate * L
                    + nG * k_glucose * G
                    + nF * k_fatty_acids * F
                    + nK * k_3HB * K
                    + 2 * k_glycolysis * G * SI)

    vG = k_glucose * M * G
    vL = k_lactate * M * L
    vF = k_fatty_acids * M * F
    vK = k_3HB * M * K
    vGL = k_glycolysis * M * G * SI

    return np.array([vL, vG, vF, vK, vGL])


######################################
# Parametrization
######################################

def ref_parameters(
        C=2.3,
        h=3.4,
        K_i_lipolysis=1.0,
        K_a_glycolysis=10.0,
        K_i_glycogenolysis=10.0,
        K_i_ketogenesis=0.2,
        insulin_action_lipolysis=True,
        insulin_action_glycolysis=True,
        insulin_action_glycogenolysis=True,
        insulin_action_ketogenesis=True,
        h_gcg_param=1.95,
        GCG_max_param=1.0,
        C_gcg_param=0.41,
        K_a_gluconeogenesis=10.0,
        K_a_glycogenolysis=10.0,
        K_a_ketogenesis=10.0,
        A_gcg_gluconeogenesis=10.0,
        A_gcg_glycogenolysis=10.0,
        A_gcg_ketogenesis=10.0,
        glucagon_action_gluconeogenesis=True,
        glucagon_action_glycogenolysis=True,
        glucagon_action_ketogenesis=True,
        steady_state_values=REF_STEADY_STATE_VALUES):
    """
    Construct reference parameters from steady state solution.
    
    Parameters:
    -----------
    C: float
        Insulin secretion threshold (relative to glucose)
    h: float
        Insulin secretion Hill coefficient
    K_i_lipolysis: float
        Insulin inhibition constant for lipolysis (scaled by I0)
    K_a_glycolysis: float
        Insulin activation constant for glycolysis (scaled by I0)
    K_i_glycogenolysis: float
        Insulin inhibition constant for glycogenolysis (scaled by I0)
    K_i_ketogenesis: float
        Insulin inhibition constant for ketogenesis (scaled by I0)
    insulin_action_*: bool
        Enable/disable insulin regulation for specific processes
    h_gcg_param: float
        Glucagon secretion Hill coefficient
    GCG_max_param: float
        Glucagon secretion capacity (default 1.0)
    C_gcg_param: float
        Glucagon secretion threshold
    K_a_gluconeogenesis: float
        Glucagon activation constant for gluconeogenesis (scaled by GCG0)
    K_a_glycogenolysis: float
        Glucagon activation constant for glycogenolysis (scaled by GCG0)
    K_a_ketogenesis: float
        Glucagon activation constant for ketogenesis (scaled by GCG0)
    A_gcg_gluconeogenesis: float
        Maximal fold-activation of gluconeogenesis by glucagon (default 2.0)
    A_gcg_glycogenolysis: float
        Maximal fold-activation of glycogenolysis by glucagon (default 2.0)
    A_gcg_ketogenesis: float
        Maximal fold-activation of ketogenesis by glucagon (default 2.0)
    glucagon_action_*: bool
        Enable/disable glucagon regulation for specific processes
        
    Returns:
    --------
    p: Parameter array for dynamic model
    """

    # Unpack steady state values
    vL, vG, vF, vK, vGL, vFK, vLG, V_glycogenolysis_ref, vA, vR, vCO2 = steady_state_values

    # Parameters 
    v_energy = 1.0
    
    # Ref. insulin
    I_max = 1.0
    I0_local = abs(1.0)**h / (abs(1.0)**h + C**h) * I_max

    # Ref. glucagon at G=1
    GCG0_local = GCG_max_param * (1.0 - 1.0**h_gcg_param / (1.0**h_gcg_param + C_gcg_param**h_gcg_param))

    # Insulin action on lipolysis
    K_i_lipolysis_scaled = I0_local * K_i_lipolysis
    if insulin_action_lipolysis:
        LI = 1.0 - I0_local / (I0_local + K_i_lipolysis_scaled)
    else:
        LI = 1.0

    # Insulin action on glucose uptake
    K_a_glycolysis_scaled = I0_local * K_a_glycolysis 
    A = 2.0
    if insulin_action_glycolysis:
        SI = 1 + A * I0_local / (I0_local + K_a_glycolysis_scaled) 
    else:
        SI = 1.0

    # Insulin action on glycogen breakdown
    K_i_glycogenolysis_scaled = I0_local * K_i_glycogenolysis
    if insulin_action_glycogenolysis:
        GI = 1.0 - I0_local / (I0_local + K_i_glycogenolysis_scaled) 
    else:
        GI = 1.0

    # Insulin action on ketone production
    K_i_ketogenesis_scaled = I0_local * K_i_ketogenesis
    if insulin_action_ketogenesis:
        FI = 1.0 - I0_local / (I0_local + K_i_ketogenesis_scaled) 
    else:
        FI = 1.0

    # Glucagon action on gluconeogenesis
    K_a_gluconeogenesis_scaled = GCG0_local * K_a_gluconeogenesis
    if glucagon_action_gluconeogenesis:
        GCG_GNG = 1.0 + A_gcg_gluconeogenesis * GCG0_local / (GCG0_local + K_a_gluconeogenesis_scaled)
    else:
        GCG_GNG = 1.0

    # Glucagon action on glycogenolysis
    K_a_glycogenolysis_scaled = GCG0_local * K_a_glycogenolysis
    if glucagon_action_glycogenolysis:
        GCG_GLY = 1.0 + A_gcg_glycogenolysis * GCG0_local / (GCG0_local + K_a_glycogenolysis_scaled)
    else:
        GCG_GLY = 1.0

    # Glucagon action on ketogenesis
    K_a_ketogenesis_scaled = GCG0_local * K_a_ketogenesis
    if glucagon_action_ketogenesis:
        GCG_KG = 1.0 + A_gcg_ketogenesis * GCG0_local / (GCG0_local + K_a_ketogenesis_scaled)
    else:
        GCG_KG = 1.0

    # Calculate parameters 
    k_glycolysis = vGL / SI
    k_lactate = vL
    k_glucose = vG
    k_fatty_acids = vF
    k_3HB = vK
    
    # Effects on lipolysis
    k_lipolysis = vA / LI 
    
    # Glycogen breakdown - accounting for both insulin and glucagon
    V_glycogenolysis = V_glycogenolysis_ref / (GI * GCG_GLY)

    # Reesterification -> more or less constant
    k_reesterification = vR / 1.0
    
    # Ketogenesis -> accounting for both insulin and glucagon
    k_ketogenesis = vFK / (FI * GCG_KG)

    # Gluconeogenesis - accounting for glucagon
    k_gluconeogenesis = vLG / GCG_GNG
    
    # Parameters to manipulate the model
    R_insulin = 0.0  # Insulin infusion
    R_glucagon = 0.0  # Glucagon infusion
    R_lactate = 0.0  # Lactate infusion
    R_glucose = 0.0  # Glucose infusion
    R_fatty_acids = 0.0  # Fatty-acid infusion
    R_3HB = 0.0  # Ketone infusion   

    # Hyperplasia
    hyperplasia = False

    return [
        v_energy, h, I_max, C, 
        K_i_lipolysis_scaled, K_a_glycolysis_scaled, K_i_glycogenolysis_scaled, K_i_ketogenesis_scaled,
        k_glycolysis, k_lactate, k_glucose, k_fatty_acids, k_3HB, 
        k_lipolysis, k_reesterification, k_ketogenesis, k_gluconeogenesis,
        V_glycogenolysis, 
        R_insulin, R_glucagon, R_lactate, R_glucose, R_fatty_acids, R_3HB, 
        insulin_action_lipolysis, insulin_action_glycolysis, insulin_action_glycogenolysis, insulin_action_ketogenesis,
        hyperplasia,
        h_gcg_param, GCG_max_param, C_gcg_param,
        K_a_gluconeogenesis_scaled, K_a_glycogenolysis_scaled, K_a_ketogenesis_scaled,
        A_gcg_gluconeogenesis, A_gcg_glycogenolysis, A_gcg_ketogenesis,
        glucagon_action_gluconeogenesis, glucagon_action_glycogenolysis, glucagon_action_ketogenesis
    ]


def change_parameters(p, e=[1.0,], ix=["v_energy",]):
    """
    Change parameters by name using PARAMETER_NAMES list.
    
    Parameters:
    -----------
    p: array-like
        Parameter array
    e: list
        Values to set
    ix: list
        Parameter names
        
    Returns:
    --------
    p_c: ndarray
        Modified parameter array
    """
    p_c = list(p).copy()
    for this_e, this_ix in zip(e, ix):
        i = PARAMETER_NAMES.index(this_ix)
        p_c[i] = this_e
        
    return p_c


######################################
# Steady state simulations
######################################

def perturbation_steady_state(A, p=None, x0=None, **kwargs):
    """
    Compute steady state with parameter perturbations.
    """
    if p is None:
        p = ref_parameters()

    if kwargs:
        (keys, values) = zip(*kwargs.items())
        p = change_parameters(p, values, ix=keys)

    if x0 is None:
        x0 = initial_state()

    X = steady_state(A, p, x0=x0)
    return X


def perturbation_steady_state_fluxes(A, p=None, x0=None, **kwargs):
    """
    Compute steady state fluxes with parameter perturbations.
    """
    if p is None:
        p = ref_parameters()

    if kwargs:
        (keys, values) = zip(*kwargs.items())
        p = change_parameters(p, values, ix=keys)

    if x0 is None:
        x0 = initial_state()

    X = steady_state(A, p, x0=x0)
    return fluxes(X, A, p)


######################################
# Dynamic simulations
######################################

def insulin_clamp_dynamic(insulin_level, time, A, p=None, **kwargs):
    """
    Simulate hyperinsulinemic-euglycemic clamp.
    
    Parameters:
    -----------
    insulin_level: float
        Insulin infusion rate (dimensionless)
    time: array-like
        Time points (minutes)
    A: float
        Adipose mass
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
        if not kwargs:
            p = ref_parameters()
        else:
            (keys, values) = zip(*kwargs.items())
            p = change_parameters(ref_parameters(), values, ix=keys)
    else:
        if kwargs:
            (keys, values) = zip(*kwargs.items())
            p = change_parameters(p, values, ix=keys)
        
    # Get steady state from parameter perturbation
    X0 = steady_state(A, p)

    # Clamp at glucose == 1 
    # Glucose infusion rate == dGdt
    # P control
    GIR_func = lambda x, A, p: np.clip(1 - x[1], 0.0, np.inf) 

    euglycemic_clamp = lambda x, t, A, p: equation(x, A, p) + np.array([0, 1, 0, 0, 0, 0, 0, 0]) * GIR_func(x, A, p)
    
    p_ins = change_parameters(p, [insulin_level,], ['R_insulin',])

    sol_X = odeint(euglycemic_clamp, X0, time, args=(A, p_ins,), rtol=1e-9, atol=1e-9)
    
    # Compute the glucose infusion as the rate needed to maintain euglycemia
    sol_GIR = np.array([GIR_func(x, A, p) for x in sol_X])

    # Export to a pandas dataframe
    X = pd.DataFrame(sol_X, columns=["L", "G", "F", "K", "I", "IA", "GCG", "GCGA"])
    X["time"] = time

    GIR_df = pd.DataFrame(sol_GIR, columns=["GIR"])
    GIR_df["time"] = time

    return X, GIR_df


def perturbation_dynamics(time, A, X0=None, p=None, **kwargs):
    """
    Simulate the time response to a parameter perturbation.
    
    Parameters:
    -----------
    time: array-like
        Time points for simulation
    A: float
        Adipose mass
    X0: array-like, optional
        Initial state. If None, uses steady state from reference parameters
    p: array-like, optional
        Base parameters. If None, uses reference parameters
    **kwargs:
        Parameter perturbations as keyword arguments
        
    Returns:
    --------
    X: pd.DataFrame
        State variables over time (L, G, F, K, I, IA, GCG, GCGA)
    F: pd.DataFrame
        Fluxes over time
    """
    # Unpack parameters
    if p is None:
        if not kwargs:
            p_ref = ref_parameters()
        else:
            (keys, values) = zip(*kwargs.items())
            p = change_parameters(ref_parameters(), values, ix=keys)
    else:
        if kwargs:
            p_ref = p.copy()
            (keys, values) = zip(*kwargs.items())
            p = change_parameters(p, values, ix=keys)
        else:
            p_ref = p.copy()
    
    # Get steady state from reference parameter
    if X0 is None:
        X0 = steady_state(A, p_ref)

    dyn_fun = lambda x, t, A, p: equation(x, A, p)

    sol_X = odeint(dyn_fun, X0, time, args=(A, p,), rtol=1e-12)

    # Export to a pandas dataframe
    X = pd.DataFrame(sol_X, columns=["L", "G", "F", "K", "I", "IA", "GCG", "GCGA"])
    X["time"] = time

    # Compute fluxes
    F = np.array([fluxes(x, A, p) for x in sol_X])
    F = pd.DataFrame(F, columns=["vL", "vG", "vF", "vK", "vGL", "vFK", "vLG",
                                  "v0", "vA", "vR", "v_in_L", "v_in_G", "v_in_F", "v_in_K", 
                                  "vI", "vIA", "vGCG", "vGCGA"])
    F["time"] = time

    return X, F


######################################
# Sensitivity analysis
######################################

def sensitivity_analysis(parameter_name, A, p=None, fold_change=2.0, **kwargs):
    """
    Compute sensitivity of steady state to parameter perturbation.
    """
    # Unpack parameters
    if p is None:
        if not kwargs:
            p = ref_parameters()
        else:
            (keys, values) = zip(*kwargs.items())
            p = change_parameters(ref_parameters(), values, ix=keys)
    else:
        if kwargs:
            p_ref = p.copy()
            (keys, values) = zip(*kwargs.items())
            p = change_parameters(p, values, ix=keys)
    
    # Get steady state from reference parameter
    X0 = steady_state(A, p)
    F0 = fluxes(X0, A, p)

    if fold_change is None:
        # Compute the derivative using central differences epsilon = 1e-3
        epsilon = 1e-4
        p_1 = list(p).copy()

        p_1[PARAMETER_NAMES.index(parameter_name)] *= (1 + epsilon)
        X_1 = steady_state(A, p_1)
        F_1 = fluxes(X_1, A, p_1)

        p_2 = list(p).copy()
        p_2[PARAMETER_NAMES.index(parameter_name)] *= (1 - epsilon)
        X_2 = steady_state(A, p_2)
        F_2 = fluxes(X_2, A, p_2)

        # compute scaled sensitivity dln(v)/dln(p) = (v1-v2)/v0 / (p1-p2)/p0
        dX = (X_1 - X_2) / X0 / (2 * epsilon)
        dF = (F_1 - F_2) / F0 / (2 * epsilon)
    
        # HOMA-IR and HOMA-B
        HOMA_IR_1 = X_1[1] * X_1[4]
        HOMA_IR_2 = X_2[1] * X_2[4]
        HOMA_IR_0 = X0[1] * X0[4]
        dHOMA_IR = (HOMA_IR_1 - HOMA_IR_2) / HOMA_IR_0 / (2 * epsilon)  
        HOMA_B_1 = X_1[4] / X_1[1]
        HOMA_B_2 = X_2[4] / X_2[1]
        HOMA_B_0 = X0[4] / X0[1]
        dHOMA_B = (HOMA_B_1 - HOMA_B_2) / HOMA_B_0 / (2 * epsilon)

    else:
        # Compute slope of F and X with respect to parameter param
        # Forward perturbation
        p_1 = list(p).copy()
        p_1[PARAMETER_NAMES.index(parameter_name)] *= fold_change
        X_1 = steady_state(A, p_1)
        F_1 = fluxes(X_1, A, p_1)

        # compute log sensitivity ln(v1/v0) / ln(p1/p0)
        dX = np.log(X_1/X0) / np.log(fold_change)
        dF = np.log(F_1/F0) / np.log(fold_change)

        dHOMA_IR = np.log(X_1[1] * X_1[4] / (X0[1] * X0[4])) / np.log(fold_change)
        dHOMA_B = np.log(X_1[4] / X_1[1] / (X0[4] / X0[1])) / np.log(fold_change)

    # Make it into a pandas series
    dX = pd.Series(dX, index=["L", "G", "F", "K", "I", "IA", "GCG", "GCGA"])
    dF = pd.Series(dF, index=["vL", "vG", "vF", "vK", "vGL", "vFK", "vLG", "v0", "vA", "vR",
                              "v_in_L", "v_in_G", "v_in_F", "v_in_K", "vI", "vIA", "vGCG", "vGCGA"])
    
    dHOMA_IR = pd.Series(dHOMA_IR, index=["HOMA_IR"])
    dHOMA_B = pd.Series(dHOMA_B, index=["HOMA_B"])
    
    # Concatenate the results
    S = pd.concat([dX, dF, dHOMA_IR, dHOMA_B], axis=0)
    return S


def jacobian(x, A, p, eps=1e-6):
    """
    Approximate Jacobian of the dynamic equations using finite differences.
    """
    x = np.array(x, dtype=float)
    f0 = np.array(equation(x, A, p), dtype=float)
    n = x.size
    m = f0.size
    J = np.zeros((m, n))
    for j in range(n):
        x_pert = x.copy()
        x_pert[j] += eps
        f1 = np.array(equation(x_pert, A, p), dtype=float)
        J[:, j] = (f1 - f0) / eps
    return J


def parallel_perturb(args):
    """
    Helper for multiprocessing: unpacks arguments and runs perturbation_steady_state.
    """
    p, X0, v = args
    vE0 = p[PARAMETER_NAMES.index("v_energy")]
    return perturbation_steady_state(1.0, p, x0=X0, v_energy=vE0 * v)
