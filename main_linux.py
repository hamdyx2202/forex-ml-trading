#!/usr/bin/env python3
"""
Forex ML Trading Bot - Linux Version
نسخة مبسطة تعمل على Linux بدون MT5
"""

import sys
import os
import sqlite3
from pathlib import Path
from loguru import logger
import argparse

# إعداد السجلات
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")
logger.add("logs/main_linux.log", rotation="1 day", retention="30 days")

def setup_database():
    """إنشاء قاعدة البيانات والجداول"""
    logger.info("Setting up database...")
    
    # إنشاء المجلدات
    Path("data").mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    conn = sqlite3.connect("data/forex_ml.db")
    cursor = conn.cursor()
    
    # جدول البيانات
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
            volume REAL NOT NULL,
            spread INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(symbol, timeframe, time)
        )
    """)
    
    # جدول التعلم
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learned_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            pattern_type TEXT NOT NULL,
            pattern_data TEXT NOT NULL,
            confidence REAL NOT NULL,
            success_rate REAL DEFAULT 0,
            total_trades INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # جدول نتائج التداول
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            entry_time TIMESTAMP,
            exit_time TIMESTAMP,
            entry_price REAL,
            exit_price REAL,
            profit_loss REAL,
            profit_pips REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()
    logger.info("✅ Database setup complete")

def test_system():
    """اختبار النظام"""
    logger.info("Testing system...")
    
    # اختبار قاعدة البيانات
    try:
        conn = sqlite3.connect("data/forex_ml.db")
        cursor = conn.cursor()
        
        # عد الجداول
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        logger.info(f"✅ Database has {len(tables)} tables")
        
        conn.close()
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        return False
    
    # اختبار المجلدات
    for folder in ["data", "models", "logs"]:
        if os.path.exists(folder):
            logger.info(f"✅ Folder exists: {folder}")
        else:
            logger.error(f"❌ Missing folder: {folder}")
    
    logger.info("✅ System test complete")
    return True

def start_bridge_server():
    """تشغيل خادم الجسر"""
    logger.info("Starting bridge server...")
    
    try:
        from src.mt5_bridge_server_linux import run_server
        logger.info("✅ Bridge server starting on port 5000...")
        run_server(host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")

def show_info():
    """عرض معلومات النظام"""
    print("\n" + "="*60)
    print("🤖 Forex ML Trading System - Linux Version")
    print("="*60)
    print("\n📋 الأوامر المتاحة:")
    print("  python main_linux.py setup    - إعداد قاعدة البيانات")
    print("  python main_linux.py test     - اختبار النظام")
    print("  python main_linux.py server   - تشغيل خادم API")
    print("  python main_linux.py info     - عرض المعلومات")
    print("\n📌 ملاحظات:")
    print("  • هذه النسخة تعمل على Linux بدون MT5")
    print("  • الخادم يستقبل البيانات من EA على Windows")
    print("  • يحلل ويرجع إشارات التداول")
    print("\n🚀 للبدء:")
    print("  1. python main_linux.py setup")
    print("  2. python main_linux.py server")
    print("  3. شغّل EA على MT5 (Windows)")
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description='Forex ML Trading Bot - Linux Version')
    parser.add_argument(
        'command',
        choices=['setup', 'test', 'server', 'info'],
        help='Command to execute'
    )
    
    args = parser.parse_args()
    
    if args.command == 'setup':
        setup_database()
    elif args.command == 'test':
        test_system()
    elif args.command == 'server':
        start_bridge_server()
    elif args.command == 'info':
        show_info()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        show_info()
    else:
        main()