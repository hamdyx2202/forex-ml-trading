#!/usr/bin/env python3
"""
🔍 فحص جاهزية النظام
"""

import os
import sys

def check_files():
    """فحص الملفات المطلوبة"""
    print("\n📁 Checking required files...")
    
    required_files = [
        'unified_trading_learning_system.py',
        'unified_prediction_server.py',
        'train_with_real_data.py',
        'live_trading_continuous_learning.py',
        'data/forex_ml.db'
    ]
    
    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            missing.append(file)
            
    return len(missing) == 0

def check_packages():
    """فحص الحزم المطلوبة"""
    print("\n📦 Checking Python packages...")
    
    packages = {
        'pandas': 'Data processing',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning',
        'lightgbm': 'LightGBM model',
        'xgboost': 'XGBoost model',
        'flask': 'Web server'
    }
    
    missing = []
    for pkg, desc in packages.items():
        try:
            __import__(pkg)
            print(f"✅ {pkg} - {desc}")
        except ImportError:
            print(f"❌ {pkg} - {desc} - MISSING")
            missing.append(pkg)
            
    return len(missing) == 0

def check_database():
    """فحص قاعدة البيانات"""
    print("\n🗄️ Checking database...")
    
    if not os.path.exists('data/forex_ml.db'):
        print("❌ Database not found")
        return False
        
    import sqlite3
    try:
        conn = sqlite3.connect('data/forex_ml.db')
        cursor = conn.cursor()
        
        # Count records
        cursor.execute("SELECT COUNT(*) FROM price_data")
        count = cursor.fetchone()[0]
        print(f"✅ Database found: {count:,} records")
        
        # Check symbols
        cursor.execute("SELECT DISTINCT symbol FROM price_data")
        symbols = [row[0] for row in cursor.fetchall()]
        print(f"✅ Symbols: {', '.join(symbols[:5])}...")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database error: {str(e)}")
        return False

def main():
    print("="*60)
    print("🔍 FOREX ML SYSTEM - READINESS CHECK")
    print("="*60)
    
    all_good = True
    
    # Check files
    if not check_files():
        all_good = False
        
    # Check packages
    if not check_packages():
        all_good = False
        print("\n💡 To install missing packages:")
        print("   pip install pandas numpy scikit-learn lightgbm xgboost flask")
        
    # Check database
    if not check_database():
        all_good = False
        
    # Summary
    print("\n" + "="*60)
    if all_good:
        print("✅ SYSTEM READY!")
        print("🚀 Run: python3 start_forex_server.py")
    else:
        print("❌ SYSTEM NOT READY")
        print("📌 Fix the issues above first")
    print("="*60)

if __name__ == "__main__":
    main()