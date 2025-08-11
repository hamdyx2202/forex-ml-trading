#!/usr/bin/env python3
"""
MT5 Data Exporter - Windows
يصدر البيانات من MT5 إلى ملفات CSV لنقلها إلى Linux VPS
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
from loguru import logger
import zipfile

class MT5DataExporter:
    """تصدير بيانات MT5 إلى ملفات"""
    
    def __init__(self, config_path: str = "config/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Create export directory
        self.export_dir = Path("data/export")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        logger.add("logs/data_exporter.log", rotation="1 day")
        
    def connect_mt5(self) -> bool:
        """الاتصال بـ MT5"""
        try:
            if not mt5.initialize():
                logger.error(f"MT5 initialization failed: {mt5.last_error()}")
                return False
                
            account_info = mt5.account_info()
            if account_info:
                logger.info(f"Connected to MT5 - Account: {account_info.login}")
                logger.info(f"Server: {account_info.server}")
                logger.info(f"Balance: ${account_info.balance:.2f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def export_symbol_data(self, symbol: str, timeframe: str, days: int = 1095) -> str:
        """تصدير بيانات رمز واحد"""
        try:
            timeframe_map = {
                'M1': mt5.TIMEFRAME_M1,
                'M5': mt5.TIMEFRAME_M5,
                'M15': mt5.TIMEFRAME_M15,
                'M30': mt5.TIMEFRAME_M30,
                'H1': mt5.TIMEFRAME_H1,
                'H4': mt5.TIMEFRAME_H4,
                'D1': mt5.TIMEFRAME_D1,
                'W1': mt5.TIMEFRAME_W1,
                'MN1': mt5.TIMEFRAME_MN1
            }
            
            tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Enable symbol
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select {symbol}")
                return None
                
            # Get historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                logger.warning(f"No data for {symbol} {timeframe}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            
            # Add symbol and timeframe
            df['symbol'] = symbol
            df['timeframe'] = timeframe
            
            # Save to CSV
            filename = f"{symbol}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv"
            filepath = self.export_dir / filename
            
            df.to_csv(filepath, index=False)
            logger.info(f"Exported {len(df)} records to {filename}")
            
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Export error for {symbol} {timeframe}: {e}")
            return None
    
    def export_all_data(self) -> dict:
        """تصدير جميع البيانات المطلوبة"""
        if not self.connect_mt5():
            logger.error("Failed to connect to MT5")
            return {}
            
        results = {}
        
        try:
            for symbol in self.config["trading"]["pairs"]:
                for timeframe in self.config["trading"]["timeframes"]:
                    key = f"{symbol}_{timeframe}"
                    filepath = self.export_symbol_data(
                        symbol, 
                        timeframe,
                        days=self.config["data"]["history_days"]
                    )
                    results[key] = filepath
                    
        finally:
            mt5.shutdown()
            
        # Create summary
        success_count = sum(1 for v in results.values() if v is not None)
        logger.info(f"Export complete: {success_count}/{len(results)} successful")
        
        return results
    
    def create_data_package(self) -> str:
        """إنشاء حزمة بيانات مضغوطة"""
        # Export all data first
        export_results = self.export_all_data()
        
        if not export_results:
            logger.error("No data exported")
            return None
            
        # Create zip file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        zip_filename = f"mt5_data_{timestamp}.zip"
        zip_path = self.export_dir / zip_filename
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add all CSV files
            for csv_file in self.export_dir.glob("*.csv"):
                if csv_file.name != zip_filename:
                    zipf.write(csv_file, csv_file.name)
                    
            # Add metadata
            metadata = {
                "export_time": datetime.now().isoformat(),
                "symbols": list(set(k.split('_')[0] for k in export_results.keys())),
                "timeframes": list(set(k.split('_')[1] for k in export_results.keys())),
                "files": {k: os.path.basename(v) if v else None 
                         for k, v in export_results.items()},
                "mt5_account": mt5.account_info().login if mt5.account_info() else None
            }
            
            # Write metadata
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
                json.dump(metadata, f, indent=2)
                temp_path = f.name
                
            zipf.write(temp_path, "metadata.json")
            os.unlink(temp_path)
        
        # Clean up individual CSV files
        for csv_file in self.export_dir.glob("*.csv"):
            if csv_file.name != zip_filename:
                csv_file.unlink()
                
        logger.info(f"Created data package: {zip_filename}")
        logger.info(f"Size: {zip_path.stat().st_size / 1024 / 1024:.2f} MB")
        
        return str(zip_path)

def main():
    """تصدير البيانات"""
    print("\n" + "="*60)
    print("📤 MT5 DATA EXPORTER")
    print("="*60)
    
    exporter = MT5DataExporter()
    
    print("\n🔄 Exporting data from MT5...")
    package_path = exporter.create_data_package()
    
    if package_path:
        print(f"\n✅ Data package created: {package_path}")
        print("\n📋 Next steps:")
        print("1. Transfer this file to your Linux VPS")
        print("2. Run: python import_mt5_data.py <package_file>")
    else:
        print("\n❌ Export failed")

if __name__ == "__main__":
    main()