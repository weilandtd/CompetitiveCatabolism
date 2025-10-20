#!/usr/bin/env python3
"""
Test script for the Competitive Catabolism web application
Run this to verify that the basic functionality works before starting the server
"""

import sys
import os

# Add the multi_nutrient_model directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'multi_nutrient_model'))

try:
    print("Testing imports...")
    from multi_nutrient_model import (
        ref_parameters, perturbation_dynamics, perturbation_steady_state,
        insulin_clamp_dynamic, I0, PARAMETER_NAMES, PARAMETER_DESCRIPTIONS,
        steady_state, fluxes, sensitivity_analysis, change_parameters
    )
    print("✓ Successfully imported multi_nutrient_model functions")
    
    import numpy as np
    import pandas as pd
    print("✓ Successfully imported numpy and pandas")
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("✓ Successfully imported matplotlib and seaborn")
    
    from flask import Flask
    print("✓ Successfully imported Flask")
    
    print("\nTesting basic model functionality...")
    
    # Test 1: Get reference parameters
    p = ref_parameters()
    print(f"✓ Generated reference parameters (length: {len(p)})")
    
    # Test 2: Steady state calculation
    A = 1.0
    X = steady_state(A, p)
    print(f"✓ Calculated steady state: G={X[1]:.3f}, I={X[4]:.3f}")
    
    # Test 3: Dynamic simulation
    time = np.linspace(0, 50, 30)
    X_dyn, F_dyn = perturbation_dynamics(time, A, p=p)
    print(f"✓ Ran dynamic simulation ({len(X_dyn)} time points)")
    
    # Test 4: Insulin clamp
    time_clamp = np.linspace(0, 60, 50)
    X_clamp, GIR = insulin_clamp_dynamic(2.0, time_clamp, A, p=p)
    print(f"✓ Ran insulin clamp simulation (final GIR={GIR['GIR'].iloc[-1]:.3f})")
    
    # Test 5: Sensitivity analysis
    S = sensitivity_analysis('alpha', A, p=p, fold_change=2.0)
    print(f"✓ Ran sensitivity analysis (HOMA-IR sensitivity={S['HOMA_IR']:.3f})")
    
    # Test 6: Plot generation
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.plot(X_dyn['time'], X_dyn['G'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Glucose')
    plt.close(fig)
    print("✓ Generated test plot")
    
    print("\n" + "="*60)
    print("All tests passed! ✓")
    print("="*60)
    print("\nYou can now start the web server with:")
    print("  python app.py")
    print("\nOr in production mode:")
    print("  gunicorn -w 4 -b 0.0.0.0:5000 app:app")
    print("="*60)
    
except ImportError as e:
    print(f"\n✗ Import Error: {e}")
    print("\nPlease ensure all dependencies are installed:")
    print("  pip install -r ../requirements.txt")
    sys.exit(1)
    
except Exception as e:
    print(f"\n✗ Test Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
