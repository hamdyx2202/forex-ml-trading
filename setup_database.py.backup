#!/usr/bin/env python3
"""
Database Setup - إعداد قاعدة البيانات والجداول المطلوبة
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
import sys

def create_database():
    """إنشاء قاعدة البيانات والجداول"""
    print("🔧 إعداد قاعدة البيانات...")
    
    # إنشاء مجلد البيانات
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # الاتصال بقاعدة البيانات
    db_path = data_dir / "forex_data.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # إنشاء جدول البيانات التاريخية
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS price_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            time INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume INTEGER NOT NULL,
            spread INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timeframe, time)
        )
    """)
    
    # إنشاء فهارس لتسريع الاستعلامات
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_timeframe ON price_data(symbol, timeframe)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_time ON price_data(time)")
    
    # جدول النماذج المدربة
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trained_models (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            model_type TEXT NOT NULL,
            accuracy REAL,
            precision_score REAL,
            recall_score REAL,
            f1_score REAL,
            features_count INTEGER,
            training_samples INTEGER,
            model_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # جدول التداولات والنتائج
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            signal_time TIMESTAMP,
            signal_type TEXT,
            entry_price REAL,
            sl_price REAL,
            tp_price REAL,
            exit_price REAL,
            exit_time TIMESTAMP,
            profit_loss REAL,
            profit_pips REAL,
            model_confidence REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # جدول الأنماط المتعلمة
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learned_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            pattern_data TEXT NOT NULL,
            success_rate REAL,
            occurrences INTEGER DEFAULT 1,
            last_seen TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # جدول مستويات الدعم والمقاومة
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS support_resistance_levels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            level_type TEXT NOT NULL,
            price_level REAL NOT NULL,
            strength INTEGER DEFAULT 1,
            first_touch TIMESTAMP,
            last_touch TIMESTAMP,
            touches_count INTEGER DEFAULT 1,
            is_active BOOLEAN DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # جدول إحصائيات الأداء
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS performance_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            date DATE NOT NULL,
            total_trades INTEGER DEFAULT 0,
            winning_trades INTEGER DEFAULT 0,
            losing_trades INTEGER DEFAULT 0,
            total_pips REAL DEFAULT 0,
            max_drawdown REAL DEFAULT 0,
            sharpe_ratio REAL,
            win_rate REAL,
            avg_win REAL,
            avg_loss REAL,
            profit_factor REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timeframe, date)
        )
    """)
    
    conn.commit()
    
    # التحقق من الجداول
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print(f"✅ تم إنشاء قاعدة البيانات: {db_path}")
    print(f"📊 عدد الجداول: {len(tables)}")
    for table in tables:
        print(f"  • {table[0]}")
    
    conn.close()
    
    return db_path

def check_database():
    """التحقق من وجود قاعدة البيانات والبيانات"""
    db_path = Path("data/forex_data.db")
    
    if not db_path.exists():
        print("❌ قاعدة البيانات غير موجودة")
        return False
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # فحص البيانات
    cursor.execute("SELECT COUNT(*) FROM price_data")
    count = cursor.fetchone()[0]
    
    if count == 0:
        print("⚠️ قاعدة البيانات موجودة لكن لا توجد بيانات")
        print("💡 يجب تشغيل جامع البيانات من MT5 أولاً")
        return False
    
    # إحصائيات البيانات
    cursor.execute("""
        SELECT symbol, timeframe, COUNT(*) as records
        FROM price_data
        GROUP BY symbol, timeframe
        ORDER BY symbol, timeframe
    """)
    
    data_stats = cursor.fetchall()
    
    print(f"✅ قاعدة البيانات جاهزة")
    print(f"📊 إجمالي السجلات: {count:,}")
    print(f"📈 العملات والفريمات المتاحة: {len(data_stats)}")
    
    # عرض عينة
    print("\n📋 عينة من البيانات المتاحة:")
    for symbol, timeframe, records in data_stats[:10]:
        print(f"  • {symbol} {timeframe}: {records:,} سجل")
    
    if len(data_stats) > 10:
        print(f"  • ... و {len(data_stats) - 10} أخرى")
    
    conn.close()
    return True

def main():
    """الدالة الرئيسية"""
    print("="*60)
    print("🗄️ إعداد قاعدة البيانات لنظام Forex ML Trading")
    print("="*60)
    
    # إنشاء قاعدة البيانات
    db_path = create_database()
    
    print("\n" + "-"*60)
    
    # فحص البيانات
    has_data = check_database()
    
    print("\n" + "="*60)
    
    if has_data:
        print("✅ النظام جاهز للتدريب!")
        print("🚀 يمكنك الآن تشغيل: python train_advanced_complete.py")
    else:
        print("⚠️ يجب جمع البيانات أولاً!")
        print("📊 خطوات جمع البيانات:")
        print("  1. افتح MT5")
        print("  2. شغّل ForexMLDataCollector_Ultimate.mq5")
        print("  3. انتظر حتى اكتمال جمع البيانات")
        print("  4. ثم شغّل التدريب")

if __name__ == "__main__":
    main()