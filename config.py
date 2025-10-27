#!/usr/bin/env python3
"""
Configuration file for the Competitive Catabolism Web Application
Loads configuration from environment variables (typically set via .env file)
"""

import os
from pathlib import Path

# Load .env file if it exists
def load_env_file():
    """Load environment variables from .env file if it exists"""
    env_path = Path(__file__).parent / '.env'
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # Only set if not already in environment
                    if key not in os.environ:
                        os.environ[key] = value

# Load .env on import
load_env_file()


class Config:
    """Base configuration - loads from environment variables"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Application settings
    DEBUG = False
    TESTING = False
    
    # Server settings (from .env file)
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # WSGI settings (from .env file)
    WORKERS = int(os.environ.get('WORKERS', 4))
    WORKER_CLASS = os.environ.get('WORKER_CLASS', 'sync')
    TIMEOUT = int(os.environ.get('TIMEOUT', 120))
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    WEB_APP_DIR = os.path.join(BASE_DIR, 'web_app')
    MULTI_NUTRIENT_MODEL_DIR = os.path.join(BASE_DIR, 'multi_nutrient_model')
    

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    HOST = '127.0.0.1'


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    # Ensure SECRET_KEY is set in production
    if not os.environ.get('SECRET_KEY'):
        import warnings
        warnings.warn('SECRET_KEY not set! Using default key. Set SECRET_KEY in .env file.')
    

class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    DEBUG = True


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'dev': DevelopmentConfig,
    'production': ProductionConfig,
    'prod': ProductionConfig,
    'testing': TestingConfig,
    'test': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env=None):
    """Get configuration object based on environment"""
    if env is None:
        env = os.environ.get('MODE') or os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])
