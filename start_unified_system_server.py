#!/usr/bin/env python3
"""
🚀 تشغيل النظام الموحد على السيرفر (بدون MT5)
📊 يشغل السيرفر فقط لاستقبال الطلبات من MT5
"""

import subprocess
import time
import os
import sys
import logging
from datetime import datetime

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('unified_system_startup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_requirements():
    """التحقق من المتطلبات"""
    logger.info("🔍 Checking requirements...")
    
    # Check if database exists
    if not os.path.exists('./data/forex_ml.db'):
        logger.error("❌ Database not found at ./data/forex_ml.db")
        return False
        
    # Check Python packages (بدون MetaTrader5)
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'lightgbm', 
        'xgboost', 'flask'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
            
    if missing:
        logger.error(f"❌ Missing packages: {', '.join(missing)}")
        logger.info("Install with: pip install " + ' '.join(missing))
        return False
        
    logger.info("✅ All requirements satisfied")
    return True

def start_unified_server():
    """تشغيل السيرفر الموحد"""
    logger.info("\n" + "="*60)
    logger.info("🚀 Starting Unified Prediction Server...")
    logger.info("="*60)
    
    # First modify the server files to work without MT5
    modify_server_files()
    
    try:
        # Start server in a subprocess
        server_process = subprocess.Popen(
            [sys.executable, 'unified_prediction_server_no_mt5.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # Wait for server to start
        time.sleep(5)
        
        # Check if server is running
        if server_process.poll() is None:
            logger.info("✅ Server started successfully")
            logger.info("📊 Server endpoints:")
            logger.info("   - POST /predict - Get trading signals")
            logger.info("   - POST /trade_result - Report trade results")
            logger.info("   - GET  /status - Server status")
            logger.info("   - POST /retrain - Force model retrain")
            logger.info("\n🌐 Server URL: http://0.0.0.0:5000")
            logger.info("📱 Configure your MT5 EA to use: http://YOUR_SERVER_IP:5000")
            return server_process
        else:
            logger.error("❌ Server failed to start")
            stdout, stderr = server_process.communicate()
            logger.error(f"Error: {stderr}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to start server: {str(e)}")
        return None

def modify_server_files():
    """تعديل ملفات السيرفر للعمل بدون MT5"""
    logger.info("📝 Creating server version without MT5...")
    
    # Create modified version of unified_trading_learning_system
    if os.path.exists('unified_trading_learning_system.py'):
        with open('unified_trading_learning_system.py', 'r') as f:
            content = f.read()
        
        # Remove MT5 imports and usage
        content = content.replace('import MetaTrader5 as mt5', '# import MetaTrader5 as mt5')
        content = content.replace('mt5.initialize()', 'False  # MT5 not available on server')
        content = content.replace('mt5.', '# mt5.')
        
        with open('unified_trading_learning_system_no_mt5.py', 'w') as f:
            f.write(content)
    
    # Create modified version of unified_prediction_server
    if os.path.exists('unified_prediction_server.py'):
        with open('unified_prediction_server.py', 'r') as f:
            content = f.read()
        
        # Import the modified system
        content = content.replace(
            'from unified_trading_learning_system import UnifiedTradingLearningSystem',
            'from unified_trading_learning_system_no_mt5 import UnifiedTradingLearningSystem'
        )
        
        with open('unified_prediction_server_no_mt5.py', 'w') as f:
            f.write(content)
    
    logger.info("✅ Server files modified for server environment")

def initial_training():
    """التدريب الأولي إذا لزم الأمر"""
    logger.info("\n" + "="*60)
    logger.info("🧠 Checking for existing models...")
    logger.info("="*60)
    
    model_dir = 'unified_models'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    if len(os.listdir(model_dir)) == 0:
        logger.info("No existing models found. Starting initial training...")
        
        # Create and run initial training script
        training_script = '''#!/usr/bin/env python3
"""Initial training using historical data only"""
from unified_trading_learning_system_no_mt5 import UnifiedTradingLearningSystem
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting initial model training using historical data...")
    
    system = UnifiedTradingLearningSystem()
    
    # Train main pairs from database
    pairs = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm']
    timeframes = ['M15', 'H1']
    
    # Map timeframes for database
    tf_map = {
        'M15': 'PERIOD_M15',
        'H1': 'PERIOD_H1'
    }
    
    for symbol in pairs:
        for tf, db_tf in tf_map.items():
            logger.info(f"Training {symbol} {tf}...")
            # The system will use historical data only since MT5 is not available
            system.train_unified_model(symbol, tf)
            
    logger.info("✅ Initial training completed")

if __name__ == "__main__":
    main()
'''
        
        with open('train_initial_models_server.py', 'w') as f:
            f.write(training_script)
            
        try:
            # Run initial training
            result = subprocess.run(
                [sys.executable, 'train_initial_models_server.py'],
                capture_output=True,
                text=True,
                timeout=3600  # 60 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Initial training completed")
            else:
                logger.error("❌ Initial training failed")
                logger.error(result.stderr)
                # Continue anyway - models can be trained later
                
        except subprocess.TimeoutExpired:
            logger.error("⚠️ Training timeout - continuing anyway")
        except Exception as e:
            logger.error(f"⚠️ Training error: {str(e)} - continuing anyway")
    else:
        logger.info(f"✅ Found {len(os.listdir(model_dir))} existing models")
        
    return True

def monitor_system(server_process):
    """مراقبة النظام"""
    logger.info("\n" + "="*60)
    logger.info("🔍 System Monitoring Started")
    logger.info("📊 Server is ready to receive requests from MT5")
    logger.info("🛑 Press Ctrl+C to stop")
    logger.info("="*60 + "\n")
    
    # Get server IP
    import socket
    hostname = socket.gethostname()
    
    # Try to get external IP
    try:
        # Get all IPs
        ips = subprocess.check_output(['hostname', '-I']).decode().strip().split()
        # Filter out local IPs
        external_ips = [ip for ip in ips if not ip.startswith('127.') and not ip.startswith('192.168.')]
        if external_ips:
            ip_address = external_ips[0]
        else:
            ip_address = ips[0] if ips else socket.gethostbyname(hostname)
    except:
        ip_address = socket.gethostbyname(hostname)
    
    logger.info(f"🌐 Server IP: {ip_address}")
    logger.info(f"📱 In MT5 EA, set ServerURL to: http://{ip_address}:5000")
    logger.info("")
    
    try:
        while True:
            # Check server health
            if server_process and server_process.poll() is not None:
                logger.error("❌ Server crashed! Restarting...")
                server_process = start_unified_server()
                
            # Display status every 5 minutes
            logger.info(f"💚 Server running... [{datetime.now().strftime('%H:%M:%S')}] - Ready for MT5 connections")
            
            time.sleep(300)  # 5 minutes
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopping system...")
        if server_process:
            server_process.terminate()
            server_process.wait()
        logger.info("✅ System stopped")

def create_deployment_guide():
    """إنشاء دليل النشر"""
    guide = """
# 📚 Server Deployment Guide

## 1. Server Requirements
- Ubuntu/Debian Linux
- Python 3.8+
- 4GB RAM minimum
- Port 5000 open

## 2. Installation Steps

```bash
# 1. Install Python and dependencies
sudo apt update
sudo apt install python3 python3-pip python3-venv

# 2. Create virtual environment
python3 -m venv venv_pro
source venv_pro/bin/activate

# 3. Install packages
pip install pandas numpy scikit-learn lightgbm xgboost flask

# 4. Start server
python start_unified_system_server.py
```

## 3. Firewall Configuration

```bash
# Open port 5000
sudo ufw allow 5000/tcp
```

## 4. MT5 EA Configuration

In your MT5 EA settings:
- ServerURL: http://YOUR_SERVER_IP:5000
- Example: http://69.62.121.53:5000

## 5. Testing Connection

From another machine:
```bash
curl http://YOUR_SERVER_IP:5000/status
```

## 6. Running as Service (Optional)

Create `/etc/systemd/system/forex-ml.service`:
```ini
[Unit]
Description=Forex ML Trading Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/forex-ml-trading
Environment="PATH=/root/forex-ml-trading/venv_pro/bin"
ExecStart=/root/forex-ml-trading/venv_pro/bin/python start_unified_system_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Then:
```bash
sudo systemctl enable forex-ml
sudo systemctl start forex-ml
```
"""
    
    with open('SERVER_DEPLOYMENT_GUIDE.md', 'w') as f:
        f.write(guide)
        
    logger.info("📚 Created SERVER_DEPLOYMENT_GUIDE.md")

def main():
    """التشغيل الرئيسي"""
    logger.info("\n" + "="*80)
    logger.info("🚀 Unified Forex ML Trading System - Server Mode")
    logger.info("📊 Historical Data Training + API Server")
    logger.info("🤖 Ready to receive signals from MT5")
    logger.info("="*80 + "\n")
    
    # Check requirements
    if not check_requirements():
        logger.error("❌ Requirements check failed")
        return
        
    # Create deployment guide
    create_deployment_guide()
    
    # Initial training if needed
    initial_training()
        
    # Start server
    server_process = start_unified_server()
    if not server_process:
        logger.error("❌ Failed to start server")
        return
        
    logger.info("\n" + "="*60)
    logger.info("✅ Server Ready!")
    logger.info("📊 Configure MT5 EA with server URL")
    logger.info("🌐 Server listening on port 5000")
    logger.info("="*60 + "\n")
    
    # Monitor system
    monitor_system(server_process)

if __name__ == "__main__":
    main()