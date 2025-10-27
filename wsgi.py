#!/usr/bin/env python3
"""
WSGI entry point for the Competitive Catabolism Web Application
This file is used by WSGI servers like Gunicorn to serve the application
"""

import sys
import os

# Add the web_app directory to the Python path
web_app_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web_app')
sys.path.insert(0, web_app_dir)

# Import the Flask app
from app import app as application

# For backwards compatibility
app = application

if __name__ == "__main__":
    # This allows running the WSGI file directly for testing
    application.run()
