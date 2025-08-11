#!/usr/bin/env python3
"""
Script لبدء خادم الجسر MT5
"""

import sys
import os
import platform
from loguru import logger

# إضافة المسار
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """بدء الخادم"""
    try:
        # التحقق من نظام التشغيل
        is_linux = platform.system() == 'Linux'
        
        if is_linux:
            logger.info("Detected Linux system - using simplified bridge server")
            logger.info("Starting Linux MT5 Bridge Server...")
            
            # استخدام النسخة المبسطة على Linux
            from src.mt5_bridge_server_linux import run_server
        else:
            logger.info("Starting full MT5 Bridge Server...")
            
            # محاولة استخدام النسخة الكاملة
            try:
                from src.mt5_bridge_server import run_server
            except ImportError as e:
                logger.warning(f"Cannot import full server: {e}")
                logger.info("Falling back to Linux version")
                from src.mt5_bridge_server_linux import run_server
        
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