#!/usr/bin/env python3
"""
Launcher script for the Competitive Catabolism Web Application
Provides a simple interface to start the server
"""

import sys
import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(
        description='Launch the Competitive Catabolism Web Application'
    )
    parser.add_argument(
        '--mode',
        choices=['dev', 'prod', 'test'],
        default='dev',
        help='Run mode: dev (Flask debug), prod (Gunicorn), or test (run tests)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=5000,
        help='Port to run the server on (default: 5000)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Number of Gunicorn workers in prod mode (default: 4)'
    )
    parser.add_argument(
        '--host',
        default='localhost',
        help='Host to bind to (default: localhost, use 0.0.0.0 for all interfaces)'
    )
    
    args = parser.parse_args()
    
    # Change to the web_app directory
    web_app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(web_app_dir)
    
    if args.mode == 'test':
        print("Running tests...")
        subprocess.run([sys.executable, 'test_app.py'])
    elif args.mode == 'dev':
        print(f"Starting development server on http://{args.host}:{args.port}")
        print("Press Ctrl+C to stop")
        print("-" * 60)
        # Set environment variables for Flask
        env = os.environ.copy()
        env['FLASK_APP'] = 'app.py'
        env['FLASK_ENV'] = 'development'
        
        # Import and run Flask directly with custom host and port
        sys.path.insert(0, web_app_dir)
        from app import app
        app.run(debug=True, host=args.host, port=args.port)
    elif args.mode == 'prod':
        print(f"Starting production server on http://{args.host}:{args.port}")
        print(f"Using {args.workers} workers")
        print("Press Ctrl+C to stop")
        print("-" * 60)
        subprocess.run([
            'gunicorn',
            '-w', str(args.workers),
            '-b', f'{args.host}:{args.port}',
            'app:app'
        ])

if __name__ == '__main__':
    main()
