#!/usr/bin/env python3
"""
Forex ML Trading Bot - Main Entry Point
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Linux compatibility first
try:
    import src.linux_compatibility
except:
    pass

import argparse
import asyncio
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Conditional imports with error handling
try:
    from src.trader import Trader
except ImportError:
    logger.warning("Trader module not available on Linux")
    Trader = None

try:
    from src.data_collector import MT5DataCollector
except ImportError:
    logger.warning("MT5DataCollector not available on Linux")
    MT5DataCollector = None

from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer

try:
    from src.monitor import TradingMonitor
except ImportError:
    logger.warning("TradingMonitor not available")
    TradingMonitor = None


def setup_logging():
    """إعداد نظام السجلات"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        "logs/main.log",
        rotation="1 day",
        retention="30 days",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
    )


def collect_data():
    """جمع البيانات التاريخية"""
    logger.info("Starting data collection...")
    collector = MT5DataCollector()
    collector.update_all_pairs()
    logger.info("Data collection completed")


def train_models():
    """تدريب النماذج"""
    logger.info("Starting model training...")
    
    collector = MT5DataCollector()
    engineer = FeatureEngineer()
    trainer = ModelTrainer()
    
    pairs = ["EURUSD", "GBPUSD", "XAUUSD"]
    timeframes = ["H1", "H4"]
    
    for symbol in pairs:
        for timeframe in timeframes:
            logger.info(f"Training model for {symbol} {timeframe}")
            
            # Get data
            df = collector.get_latest_data(symbol, timeframe, limit=10000)
            if df.empty:
                logger.warning(f"No data for {symbol} {timeframe}")
                continue
            
            # Create features
            df_features = engineer.create_features(
                df, 
                target_config={'lookahead': 5, 'threshold': 0.001}
            )
            
            if df_features.empty:
                logger.warning(f"No features created for {symbol} {timeframe}")
                continue
            
            # Train models
            models = trainer.train_all_models(df_features)
            
            # Save models
            trainer.save_models(models, symbol, timeframe)
            
            logger.info(f"Model training completed for {symbol} {timeframe}")


def start_trading():
    """بدء التداول الآلي"""
    logger.info("Starting automated trading...")
    trader = Trader()
    
    try:
        trader.start_trading()
    except KeyboardInterrupt:
        logger.info("Trading stopped by user")
    except Exception as e:
        logger.error(f"Trading error: {str(e)}")
    finally:
        trader.stop_trading()


def start_monitoring():
    """بدء خدمة المراقبة"""
    logger.info("Starting monitoring service...")
    monitor = TradingMonitor()
    
    try:
        monitor.start_monitoring()
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Monitoring error: {str(e)}")


async def start_telegram_bot():
    """بدء Telegram bot"""
    logger.info("Starting Telegram bot...")
    monitor = TradingMonitor()
    await monitor.setup_telegram_bot()


def test_connection():
    """اختبار الاتصال بـ MT5"""
    logger.info("Testing connection...")
    
    # على Linux، نختبر قاعدة البيانات فقط
    import platform
    if platform.system() == 'Linux':
        logger.info("Running on Linux - Testing database setup...")
        
        import sqlite3
        from pathlib import Path
        
        # إنشاء مجلدات البيانات
        Path("data").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
        # إنشاء قاعدة البيانات
        try:
            conn = sqlite3.connect("data/forex_ml.db")
            cursor = conn.cursor()
            
            # إنشاء الجداول الأساسية
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.commit()
            conn.close()
            
            logger.info("✅ Database created successfully")
            logger.info("✅ System ready for Linux operation")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database error: {e}")
            return False
    
    # على Windows مع MT5
    if MT5DataCollector:
        collector = MT5DataCollector()
        
        if collector.connect_mt5():
            logger.info("✅ Successfully connected to MT5")
            
            try:
                import MetaTrader5 as mt5
                account_info = mt5.account_info()
                if account_info:
                    logger.info(f"Account: {account_info.login}")
                    logger.info(f"Server: {account_info.server}")
                    logger.info(f"Balance: ${account_info.balance:.2f}")
                    logger.info(f"Leverage: 1:{account_info.leverage}")
            except:
                pass
            
            collector.disconnect_mt5()
            return True
        else:
            logger.error("❌ Failed to connect to MT5")
            return False
    else:
        logger.info("❌ MT5 not available on this system")
        return False


def main():
    """النقطة الرئيسية للبرنامج"""
    parser = argparse.ArgumentParser(description='Forex ML Trading Bot')
    parser.add_argument(
        'command',
        choices=['collect', 'train', 'trade', 'monitor', 'telegram', 'test'],
        help='Command to execute'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    if args.debug:
        logger.info("Debug mode enabled")
    
    # Execute command
    if args.command == 'collect':
        collect_data()
    elif args.command == 'train':
        train_models()
    elif args.command == 'trade':
        start_trading()
    elif args.command == 'monitor':
        start_monitoring()
    elif args.command == 'telegram':
        asyncio.run(start_telegram_bot())
    elif args.command == 'test':
        test_connection()
    else:
        logger.error(f"Unknown command: {args.command}")
        sys.exit(1)


if __name__ == "__main__":
    main()