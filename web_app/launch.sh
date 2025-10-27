#!/bin/bash
# Launch script for the Competitive Catabolism Web Application
# This script starts the application using Gunicorn WSGI server

set -e  # Exit on error

# Script directory (web_app)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Load configuration from .env file if it exists (check current and parent dir)
if [ -f "$SCRIPT_DIR/.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/.env" | xargs)
elif [ -f "$SCRIPT_DIR/../.env" ]; then
    export $(grep -v '^#' "$SCRIPT_DIR/../.env" | xargs)
fi

cd "$SCRIPT_DIR"

# Default values (used if not set in .env or environment)
MODE="${MODE:-production}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-5000}"
WORKERS="${WORKERS:-4}"
TIMEOUT="${TIMEOUT:-120}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_message() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Help message
show_help() {
    cat << EOF
Usage: ./launch.sh [MODE]

Launch the Competitive Catabolism Web Application

ARGUMENTS:
    MODE    Run mode: development, production, or test (optional)
            If not provided, reads from .env file or defaults to production

CONFIGURATION:
    Edit the .env file to configure the application settings.
    If .env doesn't exist, copy .env.example:
        cp .env.example .env

EXAMPLES:
    ./launch.sh                 # Start with settings from .env
    ./launch.sh development     # Start in development mode
    ./launch.sh production      # Start in production mode
    ./launch.sh test            # Run tests

EOF
}

# Parse arguments
if [[ $# -gt 0 ]]; then
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        development|dev|production|prod|test)
            MODE="$1"
            ;;
        *)
            print_error "Invalid mode: $1"
            show_help
            exit 1
            ;;
    esac
fi

# Check if .env file exists, if not suggest creating it
if [ ! -f .env ]; then
    print_warning ".env file not found. Using default configuration."
    print_message "To customize settings, copy .env.example to .env:"
    print_message "  cp .env.example .env"
    echo ""
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if requirements are installed
if ! python3 -c "import flask" 2>/dev/null; then
    print_warning "Flask is not installed. Installing requirements..."
    pip install -r requirements.txt
fi

# Export environment variables
export FLASK_ENV="${MODE}"
export HOST="${HOST}"
export PORT="${PORT}"
export WORKERS="${WORKERS}"
export TIMEOUT="${TIMEOUT}"

print_message "==================================================="
print_message "Competitive Catabolism Web Application"
print_message "==================================================="
print_message "Mode: ${MODE}"
print_message "Host: ${HOST}"
print_message "Port: ${PORT}"

# Check if auto-scaling is enabled
AUTO_SCALE="${AUTO_SCALE:-true}"

case $MODE in
    development|dev)
        print_message "Starting development server..."
        print_message "Access at: http://localhost:${PORT}"
        print_message "Press Ctrl+C to stop"
        print_message "==================================================="
        python3 start_server.py --mode development --host "${HOST}" --port "${PORT}"
        ;;
    
    production|prod)
        # Check if gunicorn is installed
        if ! command -v gunicorn &> /dev/null; then
            print_error "Gunicorn is not installed"
            print_message "Installing gunicorn..."
            pip install gunicorn
        fi
        
        # Build command for start_server.py
        CMD_ARGS="--mode production --host ${HOST} --port ${PORT} --timeout ${TIMEOUT}"
        
        # Add auto-scaling or fixed worker configuration
        if [ "${AUTO_SCALE}" = "true" ]; then
            print_message "Worker Scaling: Auto (dynamic based on CPU and load)"
            if [ ! -z "${MIN_WORKERS}" ]; then
                CMD_ARGS="${CMD_ARGS} --min-workers ${MIN_WORKERS}"
            fi
            if [ ! -z "${MAX_WORKERS}" ]; then
                CMD_ARGS="${CMD_ARGS} --max-workers ${MAX_WORKERS}"
            fi
        else
            print_message "Workers: ${WORKERS} (fixed)"
            CMD_ARGS="${CMD_ARGS} --no-autoscale --workers ${WORKERS}"
        fi
        
        print_message "Press Ctrl+C to stop"
        print_message "==================================================="
        
        # Start with dynamic scaling
        python3 start_server.py ${CMD_ARGS}
        ;;
    
    test)
        print_message "Running tests..."
        print_message "==================================================="
        cd web_app
        python3 test_app.py
        ;;
    
    *)
        print_error "Invalid mode: ${MODE}"
        print_message "Valid modes: development, production, test"
        exit 1
        ;;
esac
