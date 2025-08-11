#!/usr/bin/env python3
"""
Script لبدء خادم الجسر MT5
"""

import sys
import os
from loguru import logger

# إضافة المسار
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """بدء الخادم"""
    try:
        logger.info("Starting MT5 Bridge Server...")
        
        # استيراد وتشغيل الخادم
        from src.mt5_bridge_server import run_server
        
        # يمكن تغيير المنفذ من متغير البيئة
        port = int(os.getenv('BRIDGE_PORT', 5000))
        host = os.getenv('BRIDGE_HOST', '0.0.0.0')
        
        logger.info(f"Server will run on {host}:{port}")
        run_server(host=host, port=port)
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()