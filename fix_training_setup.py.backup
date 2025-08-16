#!/usr/bin/env python3
"""
Fix training setup - create directories and find correct database
إصلاح إعداد التدريب - إنشاء المجلدات والعثور على قاعدة البيانات الصحيحة
"""

import os
import sqlite3
from pathlib import Path

print("🔧 Fixing training setup...")

# 1. إنشاء مجلد النماذج
print("\n1️⃣ Creating models directory...")
os.makedirs('models/unified', exist_ok=True)
print("✅ Created models/unified/")

# 2. البحث عن قاعدة البيانات
print("\n2️⃣ Searching for database...")

# أماكن محتملة لقاعدة البيانات
possible_db_paths = [
    'forex_data.db',
    'data/forex_data.db',
    '../forex_data.db',
    'forex_ml_data.db',
    'data/forex_ml_data.db',
    'mt5_data.db',
    'data/mt5_data.db',
    'trading_data.db',
    'data/trading_data.db'
]

db_path = None
for path in possible_db_paths:
    if os.path.exists(path):
        db_path = path
        print(f"✅ Found database: {path}")
        break

if not db_path:
    print("❌ No database found!")
    print("\n🔍 Searching for any .db files...")
    
    # البحث في جميع المجلدات
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.db'):
                full_path = os.path.join(root, file)
                print(f"Found: {full_path}")
                
                # التحقق من محتوى قاعدة البيانات
                try:
                    conn = sqlite3.connect(full_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    print(f"  Tables: {[t[0] for t in tables]}")
                    
                    # البحث عن جدول يحتوي على بيانات forex
                    for table in tables:
                        table_name = table[0]
                        cursor.execute(f"PRAGMA table_info({table_name})")
                        columns = cursor.fetchall()
                        col_names = [col[1] for col in columns]
                        
                        # التحقق من وجود أعمدة OHLCV
                        if all(col in col_names for col in ['open', 'high', 'low', 'close']):
                            print(f"  ✅ Found forex data table: {table_name}")
                            db_path = full_path
                            
                            # حفظ معلومات قاعدة البيانات
                            with open('db_info.txt', 'w') as f:
                                f.write(f"Database: {full_path}\n")
                                f.write(f"Table: {table_name}\n")
                                f.write(f"Columns: {col_names}\n")
                            break
                            
                    conn.close()
                    
                    if db_path:
                        break
                        
                except Exception as e:
                    print(f"  Error reading {full_path}: {e}")

# 3. إنشاء نسخة محدثة من ملف التدريب
if db_path:
    print(f"\n3️⃣ Updating training script to use: {db_path}")
    
    # قراءة ملف التدريب الأصلي
    with open('retrain_with_unified_features.py', 'r') as f:
        content = f.read()
    
    # البحث عن اسم الجدول الصحيح
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    forex_table = None
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        col_names = [col[1] for col in columns]
        
        if all(col in col_names for col in ['open', 'high', 'low', 'close']):
            forex_table = table_name
            
            # التحقق من أسماء الأعمدة
            print(f"\n📊 Table: {table_name}")
            print(f"   Columns: {col_names}")
            
            # عينة من البيانات
            cursor.execute(f"SELECT DISTINCT symbol FROM {table_name} LIMIT 10")
            symbols = cursor.fetchall()
            print(f"   Sample symbols: {[s[0] for s in symbols]}")
            
            cursor.execute(f"SELECT DISTINCT timeframe FROM {table_name} LIMIT 10")
            timeframes = cursor.fetchall()
            print(f"   Sample timeframes: {[t[0] for t in timeframes]}")
            
            break
    
    conn.close()
    
    if forex_table:
        # تحديث ملف التدريب
        content = content.replace("db_path: str = 'forex_data.db'", f"db_path: str = '{db_path}'")
        content = content.replace("FROM forex_data", f"FROM {forex_table}")
        
        # حفظ النسخة المحدثة
        with open('retrain_with_unified_features_fixed.py', 'w') as f:
            f.write(content)
        
        print(f"✅ Created retrain_with_unified_features_fixed.py")
        print(f"   Using database: {db_path}")
        print(f"   Using table: {forex_table}")
    
else:
    print("\n❌ No suitable database found!")
    print("\n📝 Creating a sample database for testing...")
    
    # إنشاء قاعدة بيانات اختبار
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    conn = sqlite3.connect('forex_test_data.db')
    
    # إنشاء بيانات اختبار
    symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'EURJPYm', 'GBPJPYm']
    timeframes = ['PERIOD_M5', 'PERIOD_M15', 'PERIOD_H1', 'PERIOD_H4']
    
    all_data = []
    
    for symbol in symbols:
        for timeframe in timeframes:
            # تحديد عدد الدقائق لكل إطار زمني
            minutes = {
                'PERIOD_M5': 5,
                'PERIOD_M15': 15,
                'PERIOD_H1': 60,
                'PERIOD_H4': 240
            }[timeframe]
            
            # إنشاء 2000 شمعة
            start_time = datetime.now() - timedelta(days=30)
            
            for i in range(2000):
                time = start_time + timedelta(minutes=minutes * i)
                
                # بيانات عشوائية واقعية
                if 'JPY' in symbol:
                    base_price = 150.0 if symbol == 'USDJPYm' else 180.0
                else:
                    base_price = 1.1000
                
                open_price = base_price + np.random.randn() * 0.001
                close_price = open_price + np.random.randn() * 0.0005
                high_price = max(open_price, close_price) + abs(np.random.randn() * 0.0002)
                low_price = min(open_price, close_price) - abs(np.random.randn() * 0.0002)
                volume = np.random.randint(100, 10000)
                
                all_data.append({
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'time': int(time.timestamp()),
                    'open': open_price,
                    'high': high_price,
                    'low': low_price,
                    'close': close_price,
                    'volume': volume
                })
    
    # حفظ في قاعدة البيانات
    df = pd.DataFrame(all_data)
    df.to_sql('forex_data', conn, if_exists='replace', index=False)
    
    conn.close()
    
    print("✅ Created test database: forex_test_data.db")
    print(f"   Records: {len(all_data)}")
    print(f"   Symbols: {symbols}")
    print(f"   Timeframes: {timeframes}")
    
    # تحديث ملف التدريب للاستخدام
    with open('retrain_with_unified_features.py', 'r') as f:
        content = f.read()
    
    content = content.replace("db_path: str = 'forex_data.db'", "db_path: str = 'forex_test_data.db'")
    
    with open('retrain_with_unified_features_fixed.py', 'w') as f:
        f.write(content)
    
    print("\n✅ Created retrain_with_unified_features_fixed.py")

print("\n🚀 Next steps:")
print("1. Run: python retrain_with_unified_features_fixed.py")
print("2. Check models/unified/ for trained models")