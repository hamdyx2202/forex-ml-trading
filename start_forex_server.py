#!/usr/bin/env python3
"""
🚀 Forex ML Trading Server - COMPLETE SYSTEM
📊 Full features: Training + Continuous Learning + API
🌐 Server: 69.62.121.53:5000
"""

import os
import sys
import logging
import subprocess
import time
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('forex_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_server_files():
    """إنشاء ملفات السيرفر المعدلة للعمل على Linux"""
    logger.info("📝 Creating server files for Linux...")
    
    # 1. Create modified unified_trading_learning_system for server
    if os.path.exists('unified_trading_learning_system.py'):
        with open('unified_trading_learning_system.py', 'r') as f:
            content = f.read()
        
        # Comment out MT5 parts for server
        content = content.replace('import MetaTrader5 as mt5', '# import MetaTrader5 as mt5  # Not needed on server')
        content = content.replace('if mt5.initialize():', 'if False:  # MT5 not available on server')
        content = content.replace('mt5.', '# mt5.')
        
        # Save server version
        with open('unified_trading_learning_system_server.py', 'w') as f:
            f.write(content)
            
    # 2. Create server version of prediction server
    if os.path.exists('unified_prediction_server.py'):
        with open('unified_prediction_server.py', 'r') as f:
            content = f.read()
        
        # Use server version of system
        content = content.replace(
            'from unified_trading_learning_system import UnifiedTradingLearningSystem',
            'from unified_trading_learning_system_server import UnifiedTradingLearningSystem'
        )
        
        # Bind to all interfaces
        content = content.replace("app.run(host='0.0.0.0', port=5000, debug=False)", 
                                "app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)")
        
        with open('unified_prediction_server_linux.py', 'w') as f:
            f.write(content)
            
    logger.info("✅ Server files created")

def check_system():
    """فحص النظام والمتطلبات"""
    logger.info("🔍 Checking system requirements...")
    
    # Check database
    if not os.path.exists('./data/forex_ml.db'):
        logger.error("❌ Database not found at ./data/forex_ml.db")
        logger.info("   Please upload the database file")
        return False
        
    # Check Python packages
    required = ['pandas', 'numpy', 'sklearn', 'lightgbm', 'xgboost', 'flask']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
            
    if missing:
        logger.error(f"❌ Missing packages: {', '.join(missing)}")
        logger.info("   Run: pip install " + ' '.join(missing))
        return False
        
    # Check models directory
    if not os.path.exists('unified_models'):
        os.makedirs('unified_models')
        logger.info("📁 Created unified_models directory")
        
    logger.info("✅ All requirements satisfied")
    return True

def initial_training():
    """التدريب الأولي على البيانات التاريخية"""
    logger.info("\n" + "="*60)
    logger.info("🧠 Initial Training Phase")
    logger.info("="*60)
    
    # Check for existing models
    model_count = len([f for f in os.listdir('unified_models') if f.endswith('.pkl')])
    
    if model_count > 0:
        logger.info(f"✅ Found {model_count} existing models")
        return True
        
    logger.info("📊 No models found. Starting initial training...")
    
    # Create training script
    training_code = '''
import logging
from unified_trading_learning_system_server import UnifiedTradingLearningSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_initial_models():
    """Train models using historical data"""
    system = UnifiedTradingLearningSystem()
    
    # Main currency pairs
    pairs = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm']
    timeframes = ['M15', 'H1']
    
    for symbol in pairs:
        for timeframe in timeframes:
            try:
                logger.info(f"Training {symbol} {timeframe}...")
                system.train_unified_model(symbol, timeframe)
            except Exception as e:
                logger.error(f"Error training {symbol} {timeframe}: {str(e)}")
                
    logger.info("✅ Initial training completed")

if __name__ == "__main__":
    train_initial_models()
'''
    
    with open('train_initial.py', 'w') as f:
        f.write(training_code)
        
    # Run training
    try:
        result = subprocess.run(
            [sys.executable, 'train_initial.py'],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            logger.info("✅ Training completed successfully")
        else:
            logger.warning("⚠️ Training had some issues but continuing...")
            if result.stderr:
                logger.error(result.stderr)
                
    except subprocess.TimeoutExpired:
        logger.warning("⚠️ Training timeout - continuing with partial models")
    except Exception as e:
        logger.error(f"⚠️ Training error: {str(e)}")
        
    return True

def start_server():
    """تشغيل السيرفر الرئيسي"""
    logger.info("\n" + "="*60)
    logger.info("🚀 Starting Main Server")
    logger.info("="*60)
    
    try:
        # Import and run Flask app
        logger.info("📡 Starting Flask server on port 5000...")
        
        # Run server directly in this process
        import unified_prediction_server_linux
        
    except Exception as e:
        logger.error(f"❌ Server error: {str(e)}")
        
        # Try to run as subprocess
        logger.info("Trying to run server as subprocess...")
        server_process = subprocess.Popen(
            [sys.executable, 'unified_prediction_server_linux.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Monitor server
        monitor_server(server_process)

def monitor_server(server_process=None):
    """مراقبة السيرفر"""
    logger.info("\n" + "="*60)
    logger.info("🔍 Server Monitoring Active")
    logger.info("="*60)
    
    # Get server IP
    import socket
    hostname = socket.gethostname()
    
    # Get external IP
    try:
        import requests
        external_ip = requests.get('https://api.ipify.org').text
    except:
        external_ip = "69.62.121.53"
        
    logger.info(f"🌐 Server IP: {external_ip}")
    logger.info(f"📱 Configure MT5 EA with: http://{external_ip}:5000")
    logger.info(f"")
    logger.info(f"📊 Endpoints:")
    logger.info(f"   POST /predict - Send 200 candles, get signal")
    logger.info(f"   POST /trade_result - Report trade results")
    logger.info(f"   GET  /status - Check server status")
    logger.info(f"")
    logger.info(f"🛑 Press Ctrl+C to stop")
    logger.info("="*60 + "\n")
    
    try:
        while True:
            if server_process and server_process.poll() is not None:
                logger.error("❌ Server crashed! Restarting...")
                server_process = subprocess.Popen(
                    [sys.executable, 'unified_prediction_server_linux.py'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
            logger.info(f"💚 Server active [{datetime.now().strftime('%H:%M:%S')}]")
            time.sleep(300)  # Check every 5 minutes
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down server...")
        if server_process:
            server_process.terminate()
        logger.info("✅ Server stopped")

def main():
    """البرنامج الرئيسي"""
    logger.info("\n" + "="*80)
    logger.info("🚀 FOREX ML TRADING SYSTEM - COMPLETE VERSION")
    logger.info("📊 All Features: Historical Training + Live Learning + API")
    logger.info("🌐 Server: 69.62.121.53:5000")
    logger.info("🤖 6 ML Models | 200+ Features | Dynamic SL/TP")
    logger.info("="*80 + "\n")
    
    # Create server files
    create_server_files()
    
    # Check system
    if not check_system():
        logger.error("❌ System check failed")
        sys.exit(1)
        
    # Initial training
    initial_training()
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()