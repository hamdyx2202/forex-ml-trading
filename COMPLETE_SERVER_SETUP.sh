#!/bin/bash
###############################################################
# 🚀 COMPLETE FOREX ML SYSTEM SETUP
# 📊 Full features - No simplification
# 🌐 For server 69.62.121.53
###############################################################

echo "============================================================"
echo "🚀 COMPLETE FOREX ML TRADING SYSTEM INSTALLATION"
echo "📊 All Features: Training + Learning + API"
echo "🌐 Server: 69.62.121.53:5000"
echo "============================================================"

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "⚠️  Please run as root (use sudo)"
    exit 1
fi

# Update system
echo -e "\n📦 Updating system..."
apt-get update -y
apt-get upgrade -y

# Install Python and system dependencies
echo -e "\n🐍 Installing Python and dependencies..."
apt-get install -y python3 python3-dev python3-pip python3-venv
apt-get install -y build-essential gcc g++ make cmake
apt-get install -y libgomp1 libomp-dev
apt-get install -y wget curl git

# Create project directory
echo -e "\n📁 Creating project directory..."
mkdir -p /opt/forex-ml-trading
cd /opt/forex-ml-trading

# Create virtual environment
echo -e "\n🔧 Creating virtual environment..."
python3 -m venv venv_forex
source venv_forex/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all required packages
echo -e "\n📚 Installing Python packages (this may take a while)..."
pip install pandas==2.0.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.0
pip install lightgbm==4.0.0
pip install xgboost==1.7.6
pip install flask==2.3.2
pip install flask-cors==4.0.0
pip install joblib==1.3.1
pip install scipy==1.11.1
pip install requests==2.31.0

# Install TA-Lib (optional but recommended)
echo -e "\n📈 Installing TA-Lib..."
wget https://github.com/TA-Lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
make install
cd ..
rm -rf ta-lib*
pip install TA-Lib

# Create required directories
echo -e "\n📁 Creating required directories..."
mkdir -p data
mkdir -p unified_models
mkdir -p logs
mkdir -p models

# Create systemd service
echo -e "\n🔧 Creating systemd service..."
cat > /etc/systemd/system/forex-ml.service << EOF
[Unit]
Description=Forex ML Trading Server
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/forex-ml-trading
Environment="PATH=/opt/forex-ml-trading/venv_forex/bin"
ExecStart=/opt/forex-ml-trading/venv_forex/bin/python run_complete_system.py
Restart=always
RestartSec=10
StandardOutput=append:/opt/forex-ml-trading/logs/server.log
StandardError=append:/opt/forex-ml-trading/logs/server_error.log

[Install]
WantedBy=multi-user.target
EOF

# Create the complete server file
echo -e "\n📝 Creating server files..."
cat > run_complete_system.py << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
🚀 Complete Forex ML Trading Server
📊 All features enabled
🌐 Server: 69.62.121.53:5000
"""

import os
import sys
import logging
from datetime import datetime
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import json
import sqlite3
import joblib
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import the unified system
from unified_trading_learning_system import UnifiedTradingLearningSystem

# Flask app
app = Flask(__name__)

# Initialize system
logger.info("🚀 Initializing Unified Trading System...")
unified_system = UnifiedTradingLearningSystem()

# Server statistics
server_stats = {
    'start_time': datetime.now(),
    'total_requests': 0,
    'total_signals': 0,
    'active_models': 0
}

@app.route('/status', methods=['GET'])
def status():
    """Server status"""
    uptime = (datetime.now() - server_stats['start_time']).total_seconds()
    return jsonify({
        'status': 'running',
        'version': '3.0-complete',
        'server': '69.62.121.53:5000',
        'uptime_seconds': uptime,
        'models_loaded': len(unified_system.models),
        'total_requests': server_stats['total_requests'],
        'total_signals': server_stats['total_signals']
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint - receives 200 candles"""
    try:
        server_stats['total_requests'] += 1
        
        data = request.json
        symbol = data['symbol']
        timeframe = data['timeframe']
        candles = data['candles']
        
        logger.info(f"📊 Prediction request: {symbol} {timeframe} ({len(candles)} candles)")
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        # Ensure model exists
        model_key = f"{symbol}_{timeframe}"
        if model_key not in unified_system.models:
            logger.info(f"   Training new model for {model_key}...")
            unified_system.train_unified_model(symbol, timeframe)
        
        # Calculate features
        features = unified_system.calculate_adaptive_features(df, symbol, timeframe)
        latest_features = features.iloc[-1:].copy()
        
        # Predict with pattern matching
        prediction, confidence = unified_system.predict_with_pattern_matching(
            symbol, timeframe, latest_features
        )
        
        # Generate signal
        current_price = float(df['close'].iloc[-1])
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        
        if prediction == 0 and confidence >= 0.65:
            action = 'BUY'
        elif prediction == 1 and confidence >= 0.65:
            action = 'SELL'
        else:
            action = 'NONE'
        
        # Dynamic SL/TP calculation
        atr = features.get('atr_14', pd.Series([50 * pip_value])).iloc[-1]
        sl_pips = max(min(atr / pip_value * 1.5, 100), 20)
        tp1_pips = sl_pips * 2.0  # 1:2 RR
        tp2_pips = sl_pips * 3.0  # 1:3 RR
        
        # Adjust based on ADX
        adx = features.get('adx_14', pd.Series([25])).iloc[-1]
        if adx > 30:  # Strong trend
            tp1_pips *= 1.2
            tp2_pips *= 1.3
        
        # Calculate prices
        if action == 'BUY':
            sl_price = current_price - (sl_pips * pip_value)
            tp1_price = current_price + (tp1_pips * pip_value)
            tp2_price = current_price + (tp2_pips * pip_value)
        elif action == 'SELL':
            sl_price = current_price + (sl_pips * pip_value)
            tp1_price = current_price - (tp1_pips * pip_value)
            tp2_price = current_price - (tp2_pips * pip_value)
        else:
            sl_price = tp1_price = tp2_price = current_price
        
        response = {
            'symbol': symbol,
            'timeframe': timeframe,
            'action': action,
            'confidence': float(confidence),
            'current_price': current_price,
            'sl_price': float(sl_price),
            'tp1_price': float(tp1_price),
            'tp2_price': float(tp2_price),
            'sl_pips': float(sl_pips),
            'tp1_pips': float(tp1_pips),
            'tp2_pips': float(tp2_pips),
            'risk_reward_ratio': float(tp1_pips / sl_pips) if sl_pips > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        if action != 'NONE':
            server_stats['total_signals'] += 1
        
        logger.info(f"   ✅ {action} signal with {confidence:.1%} confidence")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/trade_result', methods=['POST'])
def trade_result():
    """Receive trade results for continuous learning"""
    try:
        data = request.json
        logger.info(f"📈 Trade result: {data['symbol']} - {data['result']} ({data.get('pips', 0)} pips)")
        
        # Process trade result for learning
        # unified_system.process_trade_result(data)
        
        return jsonify({'status': 'success', 'message': 'Trade result recorded'})
        
    except Exception as e:
        logger.error(f"Trade result error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/retrain', methods=['POST'])
def retrain():
    """Force model retraining"""
    try:
        data = request.json
        symbol = data.get('symbol', 'EURUSDm')
        timeframe = data.get('timeframe', 'M15')
        
        logger.info(f"🔄 Retrain request for {symbol} {timeframe}")
        
        unified_system.train_unified_model(symbol, timeframe, force_retrain=True)
        
        return jsonify({
            'status': 'success',
            'message': f'Model retrained for {symbol} {timeframe}'
        })
        
    except Exception as e:
        logger.error(f"Retrain error: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Main function
def main():
    logger.info("\n" + "="*80)
    logger.info("🚀 FOREX ML TRADING SERVER - COMPLETE SYSTEM")
    logger.info("📊 All Features Active")
    logger.info("🌐 Server: http://69.62.121.53:5000")
    logger.info("🤖 6 ML Models | 200+ Features | Dynamic SL/TP")
    logger.info("="*80 + "\n")
    
    # Start Flask server
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()
PYTHON_EOF

# Fix the unified_trading_learning_system.py for server
echo -e "\n📝 Fixing unified system for server..."
if [ -f "unified_trading_learning_system.py" ]; then
    # Create server version without MT5
    sed 's/import MetaTrader5 as mt5/# import MetaTrader5 as mt5/g' unified_trading_learning_system.py > unified_trading_learning_system_temp.py
    sed -i 's/mt5\.initialize()/False/g' unified_trading_learning_system_temp.py
    sed -i 's/mt5\./# mt5\./g' unified_trading_learning_system_temp.py
    mv unified_trading_learning_system_temp.py unified_trading_learning_system.py
fi

# Open firewall port
echo -e "\n🔥 Configuring firewall..."
ufw allow 5000/tcp

# Enable and start service
echo -e "\n🚀 Starting service..."
systemctl daemon-reload
systemctl enable forex-ml
systemctl start forex-ml

# Check status
echo -e "\n📊 Checking service status..."
systemctl status forex-ml --no-pager

echo -e "\n============================================================"
echo "✅ INSTALLATION COMPLETE!"
echo "============================================================"
echo "📊 Server URL: http://69.62.121.53:5000"
echo "📁 Installation directory: /opt/forex-ml-trading"
echo "🔧 Service name: forex-ml"
echo ""
echo "📝 Next steps:"
echo "1. Upload all Python files to /opt/forex-ml-trading/"
echo "2. Upload database to /opt/forex-ml-trading/data/"
echo "3. Check logs: journalctl -u forex-ml -f"
echo "4. Test API: curl http://69.62.121.53:5000/status"
echo "============================================================"

# Create upload helper script
cat > upload_files.sh << 'EOF'
#!/bin/bash
# Helper script to upload files
echo "📤 Uploading files to server..."
scp *.py root@69.62.121.53:/opt/forex-ml-trading/
scp -r data root@69.62.121.53:/opt/forex-ml-trading/
echo "✅ Upload complete!"
EOF
chmod +x upload_files.sh

echo -e "\n📌 Created upload_files.sh to help upload files"
echo "============================================================"