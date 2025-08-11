#!/usr/bin/env python3
"""
Quick start script for Data Sync Server
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_sync_server import run_server

if __name__ == "__main__":
    # Get host and port from command line or use defaults
    host = sys.argv[1] if len(sys.argv) > 1 else '0.0.0.0'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
    
    print("\n" + "="*60)
    print("🚀 FOREX ML DATA SYNC SERVER")
    print("="*60)
    print(f"Starting server on {host}:{port}")
    print("Ready to receive data from MT5 EA...")
    print("\nPress Ctrl+C to stop")
    print("="*60 + "\n")
    
    run_server(host, port)