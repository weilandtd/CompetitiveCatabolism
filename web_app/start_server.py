#!/usr/bin/env python3
"""
Dynamic WSGI server launcher with auto-scaling workers
Automatically adjusts worker count based on CPU cores and system load
"""

import os
import sys
import psutil
import multiprocessing
import argparse
import subprocess
import signal
from pathlib import Path


def load_env_file():
    """Load environment variables from .env file if it exists"""
    # Check current directory first, then parent directory
    env_paths = [
        Path(__file__).parent / '.env',
        Path(__file__).parent.parent / '.env'
    ]
    
    for env_path in env_paths:
        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        if key not in os.environ:
                            os.environ[key] = value
            break


def calculate_workers(auto_scale=True, min_workers=2, max_workers=None):
    """
    Calculate optimal number of workers based on system resources
    
    Formula: (2 * CPU_CORES) + 1 (recommended by Gunicorn docs)
    Adjusted by current system load
    
    Args:
        auto_scale: If True, use automatic calculation
        min_workers: Minimum number of workers
        max_workers: Maximum number of workers (None = no limit)
    
    Returns:
        int: Number of workers to use
    """
    if not auto_scale:
        return int(os.environ.get('WORKERS', 4))
    
    # Get CPU count
    cpu_count = multiprocessing.cpu_count()
    
    # Recommended formula: (2 * CPU_CORES) + 1
    recommended = (2 * cpu_count) + 1
    
    # Adjust based on current load average (on Unix-like systems)
    try:
        load_avg = psutil.getloadavg()[0]  # 1-minute load average
        load_factor = load_avg / cpu_count
        
        # If load is high, use fewer workers to avoid overload
        if load_factor > 0.7:
            recommended = max(cpu_count, min_workers)
        elif load_factor < 0.3:
            # If load is low, can use more workers
            recommended = min(recommended, (3 * cpu_count))
    except (AttributeError, OSError):
        # getloadavg not available on Windows
        pass
    
    # Apply min/max constraints
    workers = max(min_workers, recommended)
    if max_workers:
        workers = min(workers, max_workers)
    
    return workers


def get_memory_limit():
    """Calculate worker memory limit based on available RAM"""
    mem = psutil.virtual_memory()
    # Reserve 20% for system, divide rest by worker count
    available_gb = (mem.total * 0.8) / (1024**3)
    return int(available_gb)


def print_system_info(workers):
    """Print system information and worker configuration"""
    cpu_count = multiprocessing.cpu_count()
    mem = psutil.virtual_memory()
    
    print("=" * 60)
    print("System Information:")
    print(f"  CPU Cores: {cpu_count}")
    print(f"  Total Memory: {mem.total / (1024**3):.1f} GB")
    print(f"  Available Memory: {mem.available / (1024**3):.1f} GB")
    
    try:
        load_avg = psutil.getloadavg()
        print(f"  Load Average: {load_avg[0]:.2f}, {load_avg[1]:.2f}, {load_avg[2]:.2f}")
    except (AttributeError, OSError):
        print("  Load Average: Not available on this system")
    
    print(f"\nWorker Configuration:")
    print(f"  Workers: {workers}")
    print(f"  Estimated Memory per Worker: {mem.available / (1024**2) / workers:.0f} MB")
    print("=" * 60)


def start_gunicorn(mode='production', host='0.0.0.0', port=5000, 
                  workers=4, timeout=120, auto_scale=False):
    """Start Gunicorn with dynamic worker configuration"""
    
    # Calculate workers if auto-scaling is enabled
    if auto_scale:
        workers = calculate_workers(auto_scale=True, min_workers=2, max_workers=16)
        print(f"Auto-scaling enabled: Using {workers} workers")
    
    print_system_info(workers)
    
    # Gunicorn command
    cmd = [
        'gunicorn',
        '--bind', f'{host}:{port}',
        '--workers', str(workers),
        '--timeout', str(timeout),
        '--worker-class', os.environ.get('WORKER_CLASS', 'sync'),
        '--access-logfile', '-',
        '--error-logfile', '-',
        '--log-level', 'info',
        # Worker lifecycle settings
        '--max-requests', '1000',  # Restart worker after 1000 requests (prevent memory leaks)
        '--max-requests-jitter', '100',  # Add jitter to avoid all workers restarting at once
        '--preload',  # Load application before forking workers
    ]
    
    # Add worker restart settings for memory management
    mem_limit_mb = get_memory_limit() * 1024 // workers
    if mem_limit_mb > 0:
        # This is advisory - actual enforcement depends on system
        print(f"  Memory limit per worker: ~{mem_limit_mb} MB")
    
    cmd.append('wsgi:application')
    
    print(f"\nStarting Gunicorn server on http://{host}:{port}")
    print(f"Command: {' '.join(cmd)}")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down server...")
        sys.exit(0)


def start_dev_server(host='127.0.0.1', port=5000):
    """Start Flask development server"""
    os.environ['FLASK_APP'] = 'wsgi.py'
    os.environ['FLASK_ENV'] = 'development'
    
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from wsgi import application
    
    print("=" * 60)
    print(f"Starting development server on http://{host}:{port}")
    print("Auto-reload enabled")
    print("Press Ctrl+C to stop")
    print("=" * 60)
    
    application.run(debug=True, host=host, port=port)


def main():
    parser = argparse.ArgumentParser(
        description='Dynamic WSGI Server Launcher with Auto-scaling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Production with auto-scaling
  %(prog)s --no-autoscale          # Production with fixed workers from .env
  %(prog)s --mode development      # Development mode
  %(prog)s --workers 8             # Production with 8 workers
  %(prog)s --autoscale --max-workers 12  # Auto-scale up to 12 workers
        """
    )
    
    parser.add_argument(
        '--mode',
        choices=['development', 'dev', 'production', 'prod'],
        default='production',
        help='Run mode (default: production)'
    )
    
    parser.add_argument(
        '--host',
        help='Host to bind to (default: from .env or 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        help='Port to run on (default: from .env or 5000)'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        help='Number of workers (default: auto-calculated or from .env)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        help='Worker timeout in seconds (default: from .env or 120)'
    )
    
    parser.add_argument(
        '--autoscale',
        action='store_true',
        default=True,
        help='Enable auto-scaling of workers (default: enabled)'
    )
    
    parser.add_argument(
        '--no-autoscale',
        action='store_true',
        help='Disable auto-scaling (use fixed worker count)'
    )
    
    parser.add_argument(
        '--max-workers',
        type=int,
        default=16,
        help='Maximum workers when auto-scaling (default: 16)'
    )
    
    parser.add_argument(
        '--min-workers',
        type=int,
        default=2,
        help='Minimum workers when auto-scaling (default: 2)'
    )
    
    args = parser.parse_args()
    
    # Load .env file
    load_env_file()
    
    # Determine auto-scaling
    auto_scale = args.autoscale and not args.no_autoscale
    
    # Get configuration from args or .env
    host = args.host or os.environ.get('HOST', '0.0.0.0')
    port = args.port or int(os.environ.get('PORT', 5000))
    timeout = args.timeout or int(os.environ.get('TIMEOUT', 120))
    
    # Workers can be from args, .env, or auto-calculated
    if args.workers:
        workers = args.workers
        auto_scale = False  # Explicit worker count disables auto-scaling
    else:
        workers = calculate_workers(auto_scale, args.min_workers, args.max_workers)
    
    # Check dependencies
    if args.mode in ['production', 'prod']:
        try:
            import gunicorn
        except ImportError:
            print("ERROR: Gunicorn is not installed")
            print("Install with: pip install gunicorn")
            sys.exit(1)
    
    # Start server
    if args.mode in ['development', 'dev']:
        start_dev_server(host, port)
    else:
        start_gunicorn(
            mode=args.mode,
            host=host,
            port=port,
            workers=workers,
            timeout=timeout,
            auto_scale=auto_scale
        )


if __name__ == '__main__':
    main()
