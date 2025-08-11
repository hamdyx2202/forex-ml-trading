#!/usr/bin/env python3
"""
System Health Check Script
يختبر جميع مكونات النظام ويبلغ عن الحالة
"""

import sys
import os
import platform
from datetime import datetime, timedelta
from loguru import logger
import sqlite3
from pathlib import Path

# Setup logging
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

def check_environment():
    """فحص البيئة"""
    print("\n" + "="*60)
    print("🔍 ENVIRONMENT CHECK")
    print("="*60)
    
    print(f"✓ Python Version: {sys.version.split()[0]}")
    print(f"✓ Platform: {platform.system()} {platform.release()}")
    print(f"✓ Working Directory: {os.getcwd()}")
    
    # Check if running in virtual environment
    if hasattr(sys, 'prefix'):
        print(f"✓ Virtual Environment: Active")
    else:
        print(f"⚠ Virtual Environment: Not Active")

def check_directories():
    """فحص المجلدات المطلوبة"""
    print("\n" + "="*60)
    print("📁 DIRECTORY CHECK")
    print("="*60)
    
    required_dirs = ['data', 'models', 'logs', 'config', 'src']
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"✓ {dir_name}/ exists")
        else:
            print(f"✗ {dir_name}/ missing")
            Path(dir_name).mkdir(exist_ok=True)
            print(f"  → Created {dir_name}/")

def check_database():
    """فحص قاعدة البيانات"""
    print("\n" + "="*60)
    print("🗄️ DATABASE CHECK")
    print("="*60)
    
    db_path = "data/forex_ml.db"
    
    if not os.path.exists(db_path):
        print(f"✗ Database not found at {db_path}")
        print("  → Run: python main_linux.py setup")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f"✓ Database exists with {len(tables)} tables:")
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cursor.fetchone()[0]
            print(f"  • {table[0]}: {count} records")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Database error: {e}")
        return False

def check_data_collection():
    """فحص جمع البيانات"""
    print("\n" + "="*60)
    print("📊 DATA COLLECTION CHECK")
    print("="*60)
    
    try:
        # Import the Linux-compatible collector
        from src.data_collector_linux import DataCollector
        
        print("✓ Data collector module loaded")
        
        # Create instance
        collector = DataCollector()
        print("✓ Data collector initialized")
        
        # Test connection
        if collector.connect_mt5():
            print("✓ Connected to data source")
            
            # Test data fetch for one symbol
            symbol = "EURUSD"
            timeframe = "H1"
            end_date = datetime.now()
            start_date = end_date - timedelta(days=7)
            
            print(f"\nTesting data fetch for {symbol} {timeframe}...")
            df = collector.get_historical_data(symbol, timeframe, start_date, end_date)
            
            if not df.empty:
                print(f"✓ Fetched {len(df)} records")
                print(f"  • Date range: {df['time'].min()} to {df['time'].max()}")
                print(f"  • Price range: ${df['close'].min():.4f} - ${df['close'].max():.4f}")
                
                # Try to save to database
                saved = collector.save_to_database(df)
                if saved > 0:
                    print(f"✓ Saved {saved} records to database")
                else:
                    print("⚠ No new records saved (may already exist)")
            else:
                print("✗ No data fetched")
                
            collector.disconnect_mt5()
        else:
            print("✗ Failed to connect to data source")
            
        return True
        
    except Exception as e:
        print(f"✗ Data collection error: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_learning_system():
    """فحص نظام التعلم"""
    print("\n" + "="*60)
    print("🧠 LEARNING SYSTEM CHECK")
    print("="*60)
    
    try:
        # Check if learning modules exist
        modules = [
            ('src.advanced_learner', 'AdvancedLearner'),
            ('src.continuous_learner', 'ContinuousLearner'),
            ('src.feature_engineer', 'FeatureEngineer'),
            ('src.model_trainer', 'ModelTrainer')
        ]
        
        for module_name, class_name in modules:
            try:
                module = __import__(module_name, fromlist=[class_name])
                cls = getattr(module, class_name)
                print(f"✓ {class_name} module loaded")
            except Exception as e:
                print(f"✗ {class_name} module error: {e}")
        
        # Check for existing models
        model_files = list(Path('models').glob('*.pkl'))
        if model_files:
            print(f"\n✓ Found {len(model_files)} trained models:")
            for model_file in model_files[:5]:  # Show first 5
                print(f"  • {model_file.name}")
        else:
            print("\n⚠ No trained models found")
            print("  → Run: python main.py train")
        
        return True
        
    except Exception as e:
        print(f"✗ Learning system error: {e}")
        return False

def check_server():
    """فحص الخادم"""
    print("\n" + "="*60)
    print("🌐 SERVER CHECK")
    print("="*60)
    
    try:
        from src.mt5_bridge_server_linux import app
        print("✓ Bridge server module loaded")
        
        # Check if server is running
        import requests
        try:
            response = requests.get('http://localhost:5000/health', timeout=2)
            if response.status_code == 200:
                print("✓ Server is running on port 5000")
            else:
                print("⚠ Server responded with status:", response.status_code)
        except:
            print("⚠ Server is not running")
            print("  → Run: python main_linux.py server")
        
        return True
        
    except Exception as e:
        print(f"✗ Server module error: {e}")
        return False

def main():
    """الفحص الرئيسي"""
    print("\n" + "="*80)
    print("🏥 FOREX ML TRADING SYSTEM - HEALTH CHECK")
    print("="*80)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Run all checks
    results = {
        'Environment': check_environment(),
        'Directories': check_directories(),
        'Database': check_database(),
        'Data Collection': check_data_collection(),
        'Learning System': check_learning_system(),
        'Server': check_server()
    }
    
    # Summary
    print("\n" + "="*60)
    print("📈 SUMMARY")
    print("="*60)
    
    total_checks = len([r for r in results.values() if r is not None])
    passed_checks = len([r for r in results.values() if r is True])
    
    print(f"\nTotal Checks: {total_checks}")
    print(f"Passed: {passed_checks}")
    print(f"Failed: {total_checks - passed_checks}")
    
    if passed_checks == total_checks:
        print("\n✅ SYSTEM IS HEALTHY AND READY!")
    else:
        print("\n⚠ SYSTEM NEEDS ATTENTION")
        print("\nRecommended actions:")
        if not results.get('Database'):
            print("1. Run: python main_linux.py setup")
        if not results.get('Learning System'):
            print("2. Run: python main.py train")
        if not results.get('Server'):
            print("3. Run: python main_linux.py server")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()