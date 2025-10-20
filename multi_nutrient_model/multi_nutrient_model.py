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

# Insulin degrdation time constant
TAU_INS = 2 # Insulin degradation time constant
TAU_INS_A = 30 # Insulin signaling

# Time constants from volume distribution
TAU_L = 5.0 # 5 min
TAU_F = 6.5 # 6.5 min
TAU_G = 21.0 # 21 min
TAU_K = 3.0 # 3 min

# Insulin secretion
k = 3.4
C = 2.3
#Ref. insulin
I0 = abs(1.0)**k / (abs(1.0)**k + C**k)


######################################
# Sready state analysis 
######################################

# Scale the fluxes by vE -> ATP flux 

# ATP per O2 (3 per O)
PO2 = 5.0
# Whole body oxygen consumption rate ~ 2000 nmol/min/gBW
vO2 = 2000
# ATP production rate
vATP = PO2 * vO2 * 0.75

def mass_and_energy_constraints(v, vE=1.0, 
                                FG = 100/vATP, 
                                FL = 150/vATP, 
                                FK = 30/vATP ,
                                FF = 150/vATP ,
                                ):
    vL, vG, vF, vK, vGL, vFK,  vLG, v0, vA, vR, vCO2 = v


    dLdt = 2.0*vGL - 2.0*vLG - vL
    dGdt = v0 + 1/2*(vA - vR) + vLG - vGL - vG
    dFdt = 3.0*(vA-vR) - vF - vFK 
    dKdt = 4.0*vFK - vK
    # CO2 = balance 
    dCO2 = 3 * vL + 6 * vG + 16 * vF + 4 * vK - vCO2
    # Constraint energy expenditure to 
    dE = nL * vL + nG * vG + nF * vF + nK * vK + 2 * vGL - vE

    # ADDITIONAL CONSTRAINTS
    dGLY1 = vLG - 1/2 * vA # Equal contribution of glycogen and gluconeogenesis to EGP

    # Resertification constraint (2/3 of the FFA is reesterified 1/3 oxidized)
    dR = vR - 2/3 * vA

    # Direct contibution constraints 
    # DG=0.10, DF=0.50, DL=0.20, DK=0.05, 
    #dDF = DF * vCO2 - 16 * vF
    #dDL = DL * vCO2 - 3 * vL
    #dDK = DK * vCO2 - 4 * vK
    #dDG = DG * vCO2 - 6 * vG

    # Try by constraning lactate and glucose Fcircs scaled by energy expenditure
    dDG = vGL + vG + 0.5 * vA - FG
    dDL = 2* vLG + vL - FL

    dDF = vFK + vF + 3 * vR - FF
    dDK = vK - FK


    return [dLdt, dGdt, dFdt, dKdt, dGLY1, dR, dCO2, dE, dDF, dDL, dDK, ]

######################################
# Reference steady state values
######################################

# vL, vG, vF, vK, vGL, vFK,  vLG, v0, vA, vR, vCO2 = v
v0 = [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  1.0]
REF_STEADY_STATE_VALUES = fsolve(mass_and_energy_constraints, v0)
vL_ref, vG_ref, vF_ref, vK_red, vGL_ref, vFK_ref,  vLG_ref, v0_ref, vA_ref, vR_ref, vCO2 = REF_STEADY_STATE_VALUES

# Reference turnovers calcualte based on appreanc rates
TAU_F_ref = 1/vA_ref
TAU_G_ref = 1/(v0_ref + 1/2*vA_ref + vLG_ref)
TAU_L_ref = 1/(2*vGL_ref)
TAU_K_ref = 1/(4*vFK_ref)



######################################
# Parameter names and descriptions
######################################


PARAMETER_NAMES = ["vE", "k", "Imax", "C", "KI_lipo", "KA_glut4", "KI_GL", "omega", 
                   "lam", "gamma", "beta", "kappa", "alpha", "VR", "VFK", "VLG", "KI_F", 
                   "v0", "v_in_I", "v_in_L", "v_in_G", "v_in_F", "v_in_K", 
                   "with_LI", "with_SI", "with_GI", "with_FI",]


PARAMETER_DESCRIPTIONS = {"vE": "Energy expenditure",
                          "k": "Insulin hill coeffcient",
                          "f": "Fat hill coefficient",
                          "Imax": "Insulin secretion capacity",
                          "C": "Insulin secretion threshold (Glucose)", 
                          "KI_lipo": "Insulin inhibition of lipolysis",
                          "KI_F": "Insulin inhibition of ketone production",
                          "KA_glut4": "Insulin activation of glucose uptake", 
                          "KI_GL": "Insulin inhibition of glycogen breakdown",
                          "KIL": "Lactate inhibition of lipolysis",
                          "KIK": "Insulin inhibition of ketogenesis  ",
                          "KF": "Reesterification affinity fatty acids", 
                          "KG": "Reesterification affinity glucose",
                          "omega": "Glycolysis activity", 
                          "lam": "Lactate oxidation activity", 
                          "gamma": "Glucose oxidation activity", 
                          "beta": "Fatty-acid oxidation activity", 
                          "kappa": "3HB oxidation activity", 
                          "alpha": "Adipose lipolyis activity",
                          "VR": "Reseterifcation activity",  
                          "VFK": "Ketongenesis activity",
                          "KFK":"Ketogenesis affinity fatty acids" ,
                          "VLG": "Gluconeogenesis Activity ", 
                          "KL":"Gluconeogenesis affinity lactate",
                          "v0": "Liver glucose output",
                          "rho": "Fraction of regulated lipolysis",
                          "v_in_I": "Insulin infusion flux",
                          "v_in_L": "Lacate infsuion flux", 
                          "v_in_G": "Glucose infsuion flux", 
                          "v_in_F": "NEFA infsuion flux", 
                          "v_in_K": "3HB infsuion flux", 
                          "with_LI": "K/O of insulin inhibition lipolysis",
                          "with_SI": "K/O of insulin activation glucose uptake",
                          "with_GI": "K/O of insulin inhibition glycogen breakdown",
                          "with_FI": "K/O of insulin inhibition ketogenesis",

}

######################################
# Model equations
######################################

def fluxes(x,A,p):
    """
    Compute the fluxes of the model given the concentrations and parameters
    """
    
    L,G,F,K,I,IA = x
    
    vE, k, Imax, C, KI_lipo, KA_glut4, KI_GL, omega, \
    lam, gamma, beta, kappa, alpha, VR, VFK, VLG, KI_F,  \
    v0, v_in_I, v_in_L, v_in_G, v_in_F, v_in_K, \
    with_LI, with_SI, with_GI, with_FI,  hyperplasia,= p
    
    # Insulin
    vI = ( Imax * abs(G)**k / (abs(G)**k + C**k) - I ) / TAU_INS + v_in_I
    vIA = ( I - IA )/ TAU_INS_A

    # Isulin action on lipolysis
    if with_LI:
        LI = 1.0 - IA / (IA + KI_lipo) 
    else:
        LI = 1.0

    # Insulin action on glucose oxidation 
    if with_SI:
        SI =  1 + 2.0 * IA / (IA + KA_glut4 )
    else:
        SI = 1.0
        
    # Insulin action on Ketone production
    if with_FI:
        FI = 1.0 -  IA / (IA + KI_F) 
    else:
        FI = 1.0

    # Insulin action on glycogen breakdown
    if with_GI:
        GI = 1.0 -  IA / (IA + KI_GL) 
    else:
        GI = 1.0
        

    # Competitive oxidation
    M = vE/(  nL*lam*L \
            + nG*gamma*G
            + nF*beta*F
            + nK*kappa*K
            + 2*omega*G*SI
            )
    
    # Glycolysis inhibition by lactate 
    vGL = omega*M*G*SI

    vG = gamma*M*G
    vL = lam*M*L
    vF = beta*M*F
    vK = kappa*M*K

    
    # NOTE This just an idea now -> base lipid flux 
    vA = alpha * A * LI

    if hyperplasia:
        vR = VR * F * A
    else:
        vR = VR * F
    
    vFK = VFK * FI * F
    vLG = VLG * L 
    
    v0 = v0 * GI
    vLG = vLG 
    
    return np.array([vL, vG, vF, vK, vGL, vFK,  vLG, v0, vA, vR,
            v_in_L, v_in_G, v_in_F, v_in_K, vI, vIA])
    

def equation(x,A,p):

    # Relu the concentrations so that they are non-negative

    vL, vG, vF, vK, vGL, vFK, vLG, v0, vA, vR, \
    v_in_L, v_in_G, v_in_F, v_in_K, vI, vIA = fluxes(x,A,p) 

    dLdt = 2.0*vGL - 2.0*vLG - vL + v_in_L
    dGdt = v0 + 1/2*(vA -vR) + vLG - vGL - vG + v_in_G
    dFdt = 3.0*(vA-vR) - vF - vFK + v_in_F
    dKdt = 4.0*vFK - vK + v_in_K
    dIdt = vI
    dIAdt = vIA
    
    # TODO 
    # Scale the dynamic equation using the respective time constants 
    # as determined by the distributino of volume experiments

    # NOTE This only effects the time dynamics not the steady state

    return [dLdt/TAU_L*TAU_L_ref, 
            dGdt/TAU_G*TAU_G_ref, 
            dFdt/TAU_F*TAU_F_ref, 
            dKdt/TAU_K*TAU_K_ref, 
            dIdt,
            dIAdt,]



def steady_state(A,p, x0=[1.0,1.0,1.0,1.0,I0,I0], **kwargs):
    x = fsolve(equation,x0,args=(A,p),**kwargs) 
    if all(np.isclose(equation(x,A,p), np.zeros_like(x))):
        if all(x >= 0 ):
            return np.array(x)
        else:
            return np.nan * np.ones_like(x)
    else:
        return np.nan * np.ones_like(x)


# Competitive catabolism model
def competitive_oxidation(x,p,):
    # Unpack concentrations
    L,G,F,K,I = x
    
    vE, k, Imax, C, KI_lipo, KA_glut4, KI_GL, omega, \
    lam, gamma, beta, kappa, alpha, VR, VFK, VLG, KI_F,  \
    v0, v_in_I, v_in_L, v_in_G, v_in_F, v_in_K, \
    with_LI, with_SI, with_GI, with_FI,  hyperplasia,= p

    # Insulin
    I0 = abs(1.0)**k / (abs(1.0)**k + C**k)

    if with_SI:
        SI =  1 + 2.0 * I / (I + KA_glut4 )
    else:
        SI = 1.0 

    # Competitive oxidation
    M = vE/(nL*lam*L \
             + nG*gamma*G
             + nF*beta*F
             + nK*kappa*K
             + 2*omega* G * SI
             )
    
    vG = gamma*M*G
    vL = lam*M*L
    vF = beta*M*F
    vK = kappa*M*K
    vGL = omega*M*G*SI

    return np.array([vL, vG, vF, vK, vGL])
    

######################################
# Parametrization
######################################


def ref_parameters( 
        C = 2.3,
        k = 3.4,
        KI_lipo=1.0,
        KA_glut4=10.0,
        KI_GL=10.0,
        KIL=1.0,
        KIK=1.0,
        KI_F=1.0,
        with_LI = True,
        with_SI = True,
        with_GL = True,
        with_FI = True,
        steady_state=REF_STEADY_STATE_VALUES):

    # Unpack steady state values
    vL, vG, vF, vK, vGL, vFK,  vLG, v0, vA, vR, vCO2 = steady_state

    # Parameters 
    vE = 1.0
    
    #Ref. insulin
    Imax = 1.0
    I0 = abs(1.0)**k / (abs(1.0)**k + C**k) * Imax

    # Insulin action on lipolysis
    # From lactate paper
    KI_lipo = I0 * KI_lipo
    if with_LI:
        LI = 1.0 - I0 / (I0 + KI_lipo)
    else:
        LI = 1.0

    # Insulin action on glucose uptake
    # From lacate paper
    KA_glut4 = I0 * KA_glut4 
    A = 2.0
    if with_SI:
        SI =  1 + A * I0 / (I0 + KA_glut4 ) 
    else:
        SI = 1.0

    # Insulin action on glycogen breakdown
    KI_GL = I0 * KI_GL

    # Insulin action on Ketone production
    KI_F = I0 * KI_F

    if with_FI:
        FI = 1.0 - I0 / (I0 + KI_F) 
    else:
        FI = 1.0
    
    # Lactate action on lipolysis
    KIL = 1.0 * KIL
    # KETONE action on lipolysis
    KIK = 1.0 * KIK          

    # Calculate parmeters 
    omega = vGL/SI
    lam = vL
    gamma = vG
    beta = vF
    kappa = vK
    
    # Effects on lipolysis
    alpha = vA/LI 
    
    # Glycogen breakdown
    if with_GL:
        GI = 1.0 - I0 / (I0 + KI_GL) 
    else:
        GI = 1.0
    v0 = v0 / GI

    # Resterification -> more or less constant
    VR = vR / 1.0
    
    # Ketogenesis -> const. in FA dep
    VFK = vFK / FI

    # Gluconeogenesis
    VLG = vLG / 1.0
    
    # Parameters to manipulate the model
    v_in_I = 0.0 # Insulin infusion
    v_in_L = 0.0 # lactate infusion
    v_in_G = 0.0 # glucose infusion
    v_in_F = 0.0 # fatty-accid infusion
    v_in_K = 0.0 # Ketone infusion   

    # Hyperplasia
    hyperplasia = False

    return [vE, k, Imax, C, KI_lipo, KA_glut4, KI_GL, omega, \
            lam, gamma, beta, kappa, alpha, VR, VFK, VLG, KI_F, 
            v0, v_in_I, v_in_L, v_in_G, v_in_F, v_in_K, \
            with_LI, with_SI, with_GL, with_FI, hyperplasia,]


def change_parameters(p,e=[1.0,],ix=["vE",]):
    p_c = p.copy()
    for this_e, this_ix in zip(e,ix):
        i = PARAMETER_NAMES.index(this_ix)
        p_c[i] = this_e
        
    return p_c


######################################
# Steady state simulations
######################################

# Simulate perturbative simulations
def perturbation_steady_state(A, p=None, x0=[1.0,1.0,1.0,1.0,I0,I0], **kwargs):
    # Unpsack parameters 
    if p is None:
        p = ref_parameters()

    if not kwargs == {}:
        (keys,values) = zip(*kwargs.items())
        p = change_parameters(p, values, ix=keys)

    X = steady_state(A,p, x0=x0)
    return X

def perturbation_steady_state_fluxes(A,p=None, x0=[1.0,1.0,1.0,1.0,I0,I0], **kwargs):
    # Unpsack parameters 
    if p is None:
        p = ref_parameters()

    if not kwargs == {}:
        (keys,values) = zip(*kwargs.items())
        p = change_parameters(p, values, ix=keys)

    X = steady_state(A,p, x0=x0)
    return fluxes(X,A,p)


######################################
# Dynamic simulations
######################################

# Simulate perturbative simulations
def insulin_clamp_dynamic(insulin_level,time,A,p=None, **kwargs):

    # Unpack parameters 
    if p is None:
        if kwargs == {}:
            p = ref_parameters()
        else:
            (keys,values) = zip(*kwargs.items())
            p = change_parameters(ref_parameters(), values, ix=keys)
    else:
        if kwargs != {}:
            (keys,values) = zip(*kwargs.items())
            p = change_parameters(p, values, ix=keys)
        
    # Get steady state from parameter perturbation
    X0 = steady_state(A,p)

    # Clamp at glycose == 1 
    # Scale the transition function to the insulin level
    # Glucose infusion rate == dGdt
    # P control
    GIR = lambda x,A,p: np.clip(1 - x[1], 0.0, np.inf) 

    euglycemic_clamp = lambda x,t,A,p,: equation(x,A,p) + np.array([0,1,0,0,0,0]) * GIR(x,A,p)
    

    p_ins = change_parameters(p, [insulin_level,], ['v_in_I'])

    sol_X = odeint(euglycemic_clamp, X0, time, args=(A,p_ins,), rtol=1e-9, atol=1e-9)
    
    # Compute the glucose infusion as the rate needed to 
    # maintain euglycemia
    sol_GIR = np.array([GIR(x,A,p) for x in sol_X])

    # Export to a pandas dataframe
    X = pd.DataFrame(sol_X, columns=["L","G","F","K","I","IA"])
    X["time"] = time

    GIR = pd.DataFrame(sol_GIR, columns=["GIR"])
    GIR["time"] = time

    return X, GIR


# Simulate the time response to a parameter perturbation
def perturbation_dynamics(time,A,X0=None,p=None, **kwargs):

    # Unpack parameters
    if p is None:
        if kwargs == {}:
            p_ref = ref_parameters()
        else:
            (keys,values) = zip(*kwargs.items())
            p = change_parameters(ref_parameters(), values, ix=keys)
    else:
        if kwargs != {}:
            p_ref = p.copy()
            (keys,values) = zip(*kwargs.items())
            p = change_parameters(p, values, ix=keys)
        else:
            p_ref = p.copy()
    
    # Get steady state from reference parameter
    if X0 is None:
        X0 = steady_state(A,p_ref)

    dyn_fun = lambda x,t,A,p: equation(x,A,p)

    sol_X = odeint(dyn_fun, X0, time, args=(A,p,), rtol=1e-12, )

    # Export to a pandas dataframe
    X = pd.DataFrame(sol_X, columns=["L","G","F","K","I","IA"])
    X["time"] = time

    # Compute fluxes
    F = np.array([fluxes(x,A,p) for x in sol_X])
    F = pd.DataFrame(F, columns=["vL","vG","vF","vK","vGL","vFK","vLG",
                                 "v0","vA","vR","v_in_L","v_in_G","v_in_F","v_in_K","vI","vIA"])
    F["time"] = time

    return X, F


######################################
# Sensitivity analysis
######################################


def sensitivity_analysis(parameter_name,A,p=None,fold_change=2.0,**kwargs):
    
    # Unpack parameters
    if p is None:
        if kwargs == {}:
            p = ref_parameters()
        else:
            (keys,values) = zip(*kwargs.items())
            p = change_parameters(ref_parameters(), values, ix=keys)
    else:
        if kwargs != {}:
            p_ref = p.copy()
            (keys,values) = zip(*kwargs.items())
            p = change_parameters(p, values, ix=keys)
    
    # Get steady state from reference parameter
    X0 = steady_state(A,p)
    F0 = fluxes(X0,A,p)


    if fold_change is None:
        # COmpute the derivative using cenral differences epsilon = 1e-3
        epsilon = 1e-4
        p_1 = p.copy()

        p_1[PARAMETER_NAMES.index(parameter_name)] *= (1 + epsilon)
        X_1 = steady_state(A,p_1)
        F_1 = fluxes(X_1,A,p_1)

        p_2 = p.copy()
        p_2[PARAMETER_NAMES.index(parameter_name)] *= (1 - epsilon)
        X_2 = steady_state(A,p_2)
        F_2 = fluxes(X_2,A,p_2)

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
        p_1 = p.copy()
        p_1[PARAMETER_NAMES.index(parameter_name)] *= fold_change
        X_1 = steady_state(A,p_1)
        F_1 = fluxes(X_1,A,p_1)

        # compute log sensitivityt ln(v1/v0) / ln(p1/p0)
        dX = np.log(X_1/X0) / np.log(fold_change)
        dF = np.log(F_1/F0) / np.log(fold_change)


        dHOMA_IR = np.log(X_1[1] * X_1[4] / (X0[1] * X0[4])) / np.log(fold_change)
        dHOMA_B = np.log(X_1[4] / X_1[1] / (X0[4] / X0[1])) / np.log(fold_change)

    # Make it into a pandas series
    dX = pd.Series(dX, index=["L","G","F","K","I","IA"])
    dF = pd.Series(dF, index=["vL","vG","vF","vK","vGL","vFK","vLG","v0","vA","vR",
                              "v_in_L","v_in_G","v_in_F","v_in_K","vI","vIA"])
    
    dHOMA_IR = pd.Series(dHOMA_IR, index=["HOMA_IR"])
    dHOMA_B = pd.Series(dHOMA_B, index=["HOMA_B"])
    
    # Concatenate the results
    S = pd.concat([dX,dF, dHOMA_IR, dHOMA_B], axis=0)
    return S
    
    # Unpack parameters
    if p is None:
        if kwargs == {}:
            p = ref_parameters()
        else:
            (keys,values) = zip(*kwargs.items())
            p = change_parameters(ref_parameters(), values, ix=keys)
    else:
        if kwargs != {}:
            p_ref = p.copy()
            (keys,values) = zip(*kwargs.items())
            p = change_parameters(p, values, ix=keys)
    
    # Get steady state from reference parameter
    X0 = steady_state(A,p)
    F0 = fluxes(X0,A,p)

    # Compute slope of F and X with respect to parameter param
    # Forward perturbation
    p_1 = p.copy()
    p_1[PARAMETER_NAMES.index(parameter_name)] *= fold_change
    X_1 = steady_state(A,p_1)
    F_1 = fluxes(X_1,A,p_1)

    # compute log sensitivityt ln(v1/v0) / ln(p1/p0)
    dX = np.log(X_1/X0) / np.log(fold_change)
    dF = np.log(F_1/F0) / np.log(fold_change)


    dHOMA_IR = np.log(X_1[1] * X_1[4] / (X0[1] * X0[4])) / np.log(fold_change)
    dHOMA_B = np.log(X_1[4] / X_1[1] / (X0[4] / X0[1])) / np.log(fold_change)

    # Make it into a pandas series
    dX = pd.Series(dX, index=["L","G","F","K","I","IA"])
    dF = pd.Series(dF, index=["vL","vG","vF","vK","vGL","vFK","vLG","v0","vA","vR",
                              "v_in_L","v_in_G","v_in_F","v_in_K","vI","vIA"])
    
    dHOMA_IR = pd.Series(dHOMA_IR, index=["HOMA_IR"])
    dHOMA_B = pd.Series(dHOMA_B, index=["HOMA_B"])
    
    # Concatenate the results
    S = pd.concat([dX,dF, dHOMA_IR, dHOMA_B], axis=0)
    return S

def jacobian(x, A, p, eps=1e-6):
    """
    Approximate Jacobian of the dynamic equations using finite differences.
    """
    # ensure x is array
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
    vE0 = p[PARAMETER_NAMES.index("vE")]
    return perturbation_steady_state(1.0, p, x0=X0, vE=vE0 * v)




