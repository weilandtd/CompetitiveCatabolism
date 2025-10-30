"""
Utility functions for the Competitive Catabolism web application
"""

import json
import hashlib
from pathlib import Path
from threading import Lock
from datetime import datetime
from flask import request

# Visitor tracking file paths
VISITOR_IPS_FILE = Path(__file__).parent / 'visitor_ips.json'
ACTIVE_VISITORS_FILE = Path(__file__).parent / 'active_visitors.json'
visitor_lock = Lock()
ACTIVE_TIMEOUT = 300  # 5 minutes in seconds


def hash_ip(ip_address):
    """
    Hash an IP address using SHA-256 for privacy.
    This is one-way encryption - the original IP cannot be recovered.
    """
    if not ip_address:
        return None
    # Use SHA-256 to create a one-way hash of the IP
    return hashlib.sha256(ip_address.encode('utf-8')).hexdigest()


def get_client_ip():
    """Get the client's IP address, handling proxies"""
    # Check for X-Forwarded-For header (for proxies/load balancers)
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0].strip()
    # Check for X-Real-IP header
    elif request.headers.get('X-Real-IP'):
        return request.headers.get('X-Real-IP')
    # Fallback to remote_addr
    return request.remote_addr


def get_visitor_data():
    """Get both total unique visitors and active visitor counts"""
    with visitor_lock:
        unique_ip_hashes = set()
        active_ip_hashes = {}
        
        # Get unique visitor IP hashes
        try:
            if VISITOR_IPS_FILE.exists():
                with open(VISITOR_IPS_FILE, 'r') as f:
                    data = json.load(f)
                    unique_ip_hashes = set(data.get('ip_hashes', []))
        except Exception:
            pass
        
        # Get active visitors (clean up stale entries)
        try:
            if ACTIVE_VISITORS_FILE.exists():
                with open(ACTIVE_VISITORS_FILE, 'r') as f:
                    active_ip_hashes = json.load(f)
                
                # Remove stale sessions (older than ACTIVE_TIMEOUT)
                now = datetime.now().timestamp()
                active_ip_hashes = {
                    ip_hash: last_seen for ip_hash, last_seen in active_ip_hashes.items()
                    if isinstance(last_seen, (int, float)) and now - last_seen < ACTIVE_TIMEOUT
                }
                
                # Filter out any non-hashed IPs (cleanup from old format)
                # Hashed IPs are 64 characters (SHA-256 hex)
                active_ip_hashes = {
                    ip_hash: last_seen for ip_hash, last_seen in active_ip_hashes.items()
                    if len(ip_hash) == 64
                }
                
                # Save cleaned up sessions
                with open(ACTIVE_VISITORS_FILE, 'w') as f:
                    json.dump(active_ip_hashes, f, indent=2)
        except Exception:
            pass
        
        return len(unique_ip_hashes), len(active_ip_hashes)


def track_unique_visitor(ip_address):
    """Track a unique visitor by hashed IP and return if it's a new visitor"""
    ip_hash = hash_ip(ip_address)
    if not ip_hash:
        return False, 0
    
    with visitor_lock:
        unique_ip_hashes = set()
        is_new = False
        
        try:
            if VISITOR_IPS_FILE.exists():
                with open(VISITOR_IPS_FILE, 'r') as f:
                    data = json.load(f)
                    # Handle both old format ('ips') and new format ('ip_hashes')
                    unique_ip_hashes = set(data.get('ip_hashes', data.get('ips', [])))
                    # Filter to only keep hashed values (64 characters)
                    unique_ip_hashes = {h for h in unique_ip_hashes if len(h) == 64}
        except Exception:
            pass
        
        # Check if this is a new visitor
        if ip_hash not in unique_ip_hashes:
            is_new = True
            unique_ip_hashes.add(ip_hash)
            
            try:
                with open(VISITOR_IPS_FILE, 'w') as f:
                    json.dump({'ip_hashes': list(unique_ip_hashes)}, f, indent=2)
            except Exception:
                pass
        
        return is_new, len(unique_ip_hashes)


def update_active_visitor(ip_address):
    """Update the last seen time for an active visitor using hashed IP"""
    ip_hash = hash_ip(ip_address)
    if not ip_hash:
        return
    
    with visitor_lock:
        active_ip_hashes = {}
        
        try:
            if ACTIVE_VISITORS_FILE.exists():
                with open(ACTIVE_VISITORS_FILE, 'r') as f:
                    active_ip_hashes = json.load(f)
        except Exception:
            pass
        
        # Filter out any old non-hashed entries (64 chars for SHA-256)
        active_ip_hashes = {
            k: v for k, v in active_ip_hashes.items()
            if len(k) == 64 and isinstance(v, (int, float))
        }
        
        # Update this IP hash's timestamp
        active_ip_hashes[ip_hash] = datetime.now().timestamp()
        
        # Clean up stale sessions
        now = datetime.now().timestamp()
        active_ip_hashes = {
            ip_hash: last_seen for ip_hash, last_seen in active_ip_hashes.items()
            if now - last_seen < ACTIVE_TIMEOUT
        }
        
        try:
            with open(ACTIVE_VISITORS_FILE, 'w') as f:
                json.dump(active_ip_hashes, f, indent=2)
        except Exception:
            pass
