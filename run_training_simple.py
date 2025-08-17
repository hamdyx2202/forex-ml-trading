#!/usr/bin/env python3
"""
Simple training script - سكريبت تدريب مبسط
يعمل مباشرة بدون التعقيدات
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

# إضافة المسار
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the trainer
from train_advanced_complete_full_features import UltimateAdvancedTrainer

def check_database():
    """التحقق من وجود قاعدة البيانات"""
    db_path = "data/forex_ml.db"
    if not os.path.exists(db_path):
        print(f"❌ Database not found at: {db_path}")
        print("Please ensure the database exists with price data")
        return False
    
    # التحقق من وجود بيانات
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # التحقق من وجود الجدول
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='price_data'")
        if not cursor.fetchone():
            print("❌ Table 'price_data' not found in database")
            return False
        
        # التحقق من عدد السجلات
        cursor.execute("SELECT COUNT(*) FROM price_data")
        count = cursor.fetchone()[0]
        print(f"✅ Found {count:,} records in database")
        
        # عرض الرموز المتاحة
        cursor.execute("SELECT DISTINCT symbol FROM price_data")
        symbols = [row[0] for row in cursor.fetchall()]
        print(f"📊 Available symbols: {', '.join(symbols[:10])}")
        if len(symbols) > 10:
            print(f"   ... and {len(symbols)-10} more")
        
        conn.close()
        return count > 1000
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def main():
    """الدالة الرئيسية"""
    print("="*80)
    print("🚀 Simple Forex ML Training Script")
    print("="*80)
    
    # التحقق من قاعدة البيانات
    if not check_database():
        print("\n⚠️ Please setup the database first!")
        print("You need to:")
        print("1. Create data/forex_ml.db")
        print("2. Add price_data table with columns: symbol, timeframe, timestamp, open, high, low, close, volume")
        print("3. Import historical data")
        return
    
    # إنشاء المدرب
    print("\n🔧 Initializing trainer...")
    trainer = UltimateAdvancedTrainer()
    
    # تعديل الإعدادات للتدريب السريع
    trainer.use_all_features = False  # استخدام ميزات أقل للسرعة
    trainer.use_all_models = False    # استخدام نموذج واحد فقط
    trainer.max_workers = 2           # عدد أقل من المعالجات
    
    # الحصول على رمز من قاعدة البيانات
    print("\n📊 Getting available symbols from database...")
    try:
        conn = sqlite3.connect("data/forex_ml.db")
        query = """
            SELECT DISTINCT symbol, timeframe, COUNT(*) as count
            FROM price_data
            WHERE symbol LIKE '%USD%'
            GROUP BY symbol, timeframe
            HAVING count >= 1000
            ORDER BY count DESC
            LIMIT 5
        """
        available = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(available) == 0:
            print("❌ No symbols with enough data found")
            return
            
        print("\n📊 Available symbols with data:")
        for idx, row in available.iterrows():
            print(f"  {idx+1}. {row['symbol']} {row['timeframe']} - {row['count']:,} records")
        
        # اختيار أول رمز متاح
        symbol = available.iloc[0]['symbol']
        timeframe = available.iloc[0]['timeframe']
        
        print(f"\n🎯 Training on {symbol} {timeframe}...")
        results = trainer.train_symbol(symbol, timeframe)
        
        if results:
            print(f"\n✅ Training completed!")
            print(f"Best strategy: {results.get('best_strategy')}")
            print(f"Best accuracy: {results.get('best_accuracy', 0):.2%}")
            
            # عرض نتائج كل استراتيجية
            print("\n📈 Strategy results:")
            for strategy_name, strategy_results in results.get('strategies', {}).items():
                print(f"  - {strategy_name}: {strategy_results.get('accuracy', 0):.2%}")
        else:
            print("❌ Training failed - no results returned")
            
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("✅ Script completed!")
    print("="*80)

if __name__ == "__main__":
    main()