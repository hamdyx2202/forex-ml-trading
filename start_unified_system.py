#!/usr/bin/env python3
"""
🚀 تشغيل النظام الموحد الكامل
📊 يشغل السيرفر والتدريب والتعلم المستمر
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
        
    # Check Python packages
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'lightgbm', 
        'xgboost', 'flask', 'MetaTrader5'
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
    
    try:
        # Start server in a subprocess
        server_process = subprocess.Popen(
            [sys.executable, 'unified_prediction_server.py'],
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
            return server_process
        else:
            logger.error("❌ Server failed to start")
            stdout, stderr = server_process.communicate()
            logger.error(f"Error: {stderr}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Failed to start server: {str(e)}")
        return None

def initial_training():
    """التدريب الأولي إذا لزم الأمر"""
    logger.info("\n" + "="*60)
    logger.info("🧠 Checking for existing models...")
    logger.info("="*60)
    
    model_dir = 'unified_models'
    if not os.path.exists(model_dir) or len(os.listdir(model_dir)) == 0:
        logger.info("No existing models found. Starting initial training...")
        
        try:
            # Run initial training
            result = subprocess.run(
                [sys.executable, 'train_initial_models.py'],
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Initial training completed")
            else:
                logger.error("❌ Initial training failed")
                logger.error(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Training timeout after 30 minutes")
            return False
        except Exception as e:
            logger.error(f"❌ Training error: {str(e)}")
            return False
    else:
        logger.info(f"✅ Found {len(os.listdir(model_dir))} existing models")
        
    return True

def create_initial_training_script():
    """إنشاء سكريبت للتدريب الأولي"""
    script_content = '''#!/usr/bin/env python3
"""
🧠 Initial Model Training
📊 Trains models for main pairs and timeframes
"""

from unified_trading_learning_system import UnifiedTradingLearningSystem
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting initial model training...")
    
    system = UnifiedTradingLearningSystem()
    
    # Train main pairs
    pairs = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'AUDUSDm']
    timeframes = ['M15', 'H1']
    
    for symbol in pairs:
        for timeframe in timeframes:
            logger.info(f"Training {symbol} {timeframe}...")
            system.train_unified_model(symbol, timeframe)
            
    logger.info("✅ Initial training completed")

if __name__ == "__main__":
    main()
'''
    
    with open('train_initial_models.py', 'w') as f:
        f.write(script_content)
        
def monitor_system(server_process):
    """مراقبة النظام"""
    logger.info("\n" + "="*60)
    logger.info("🔍 System Monitoring Started")
    logger.info("📊 Press Ctrl+C to stop")
    logger.info("="*60 + "\n")
    
    try:
        while True:
            # Check server health
            if server_process and server_process.poll() is not None:
                logger.error("❌ Server crashed! Restarting...")
                server_process = start_unified_server()
                
            # Display status every minute
            logger.info(f"💚 System running... [{datetime.now().strftime('%H:%M:%S')}]")
            
            # You can add more monitoring here:
            # - Check database size
            # - Check model performance
            # - Check memory usage
            # - etc.
            
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("\n🛑 Stopping system...")
        if server_process:
            server_process.terminate()
            server_process.wait()
        logger.info("✅ System stopped")

def create_mt5_setup_guide():
    """إنشاء دليل إعداد MT5"""
    guide = """
# 📚 MT5 Setup Guide for Unified ML System

## 1. Prerequisites
- MetaTrader 5 installed
- Python integration enabled
- Server URL configured

## 2. EA Installation
1. Copy `ForexMLBot_Advanced_V3_Unified.mq5` to: 
   `[MT5_Data_Folder]/MQL5/Experts/`

2. Compile the EA in MetaEditor (F7)

3. Add server URL to allowed URLs:
   Tools → Options → Expert Advisors → Allow WebRequest for:
   - http://localhost:5000
   - Your remote server URL (if using)

## 3. EA Configuration
- **Server URL**: http://localhost:5000 (or your server)
- **Min Confidence**: 0.65 (65%)
- **Candles to Send**: 200
- **Use Server SL/TP**: Yes ✓

## 4. Running the System

### Step 1: Start the server
```bash
python start_unified_system.py
```

### Step 2: Attach EA to charts
- Attach to desired pairs (EURUSD, GBPUSD, etc.)
- Use M15 or H1 timeframes
- Enable AutoTrading

### Step 3: Monitor
- Check EA panel for statistics
- Monitor server logs
- Review trade results

## 5. Important Notes
- The server must be running before EA starts
- EA sends 200 candles every minute
- Server updates models every 30 minutes
- Trade results are sent back for learning

## 6. Troubleshooting
- **No signals**: Check server logs, increase timeout
- **WebRequest error**: Add URL to allowed list
- **Low confidence**: Normal, system is selective
"""
    
    with open('MT5_SETUP_GUIDE.md', 'w', encoding='utf-8') as f:
        f.write(guide)
        
    logger.info("📚 Created MT5_SETUP_GUIDE.md")

def main():
    """التشغيل الرئيسي"""
    logger.info("\n" + "="*80)
    logger.info("🚀 Unified Forex ML Trading System")
    logger.info("📊 Historical Data + Live Trading + Continuous Learning")
    logger.info("🤖 6 ML Models | 200+ Features | Dynamic SL/TP")
    logger.info("="*80 + "\n")
    
    # Check requirements
    if not check_requirements():
        logger.error("❌ Requirements check failed")
        return
        
    # Create helper scripts
    create_initial_training_script()
    create_mt5_setup_guide()
    
    # Initial training if needed
    if not initial_training():
        logger.error("❌ Initial training failed")
        return
        
    # Start server
    server_process = start_unified_server()
    if not server_process:
        logger.error("❌ Failed to start server")
        return
        
    logger.info("\n" + "="*60)
    logger.info("✅ System Ready!")
    logger.info("📊 Next steps:")
    logger.info("   1. Open MT5")
    logger.info("   2. Attach EA to charts")
    logger.info("   3. Enable AutoTrading")
    logger.info("="*60 + "\n")
    
    # Monitor system
    monitor_system(server_process)

if __name__ == "__main__":
    main()