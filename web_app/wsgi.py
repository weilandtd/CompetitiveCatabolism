#!/usr/bin/env python3
"""
WSGI entry point for the Competitive Catabolism Web Application
This file is used by WSGI servers like Gunicorn to serve the application
"""

import sys
import os

# Add the parent directory to path for multi_nutrient_model access
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import the Flask app from current directory
from app import app as application

# For backwards compatibility
app = application

if __name__ == "__main__":
    # This allows running the WSGI file directly for testing
    application.run()
