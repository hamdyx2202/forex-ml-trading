#!/usr/bin/env python3
"""
MT5 Data Importer - Linux
يستورد البيانات المصدرة من MT5 إلى قاعدة البيانات
"""

import sys
import json
import sqlite3
import pandas as pd
import zipfile
from pathlib import Path
from datetime import datetime
from loguru import logger
import argparse

class MT5DataImporter:
    """استيراد بيانات MT5 المصدرة"""
    
    def __init__(self, db_path: str = "data/forex_ml.db"):
        self.db_path = db_path
        self.temp_dir = Path("data/temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.add("logs/data_importer.log", rotation="1 day")
        
    def extract_package(self, package_path: str) -> dict:
        """فك ضغط حزمة البيانات"""
        try:
            with zipfile.ZipFile(package_path, 'r') as zipf:
                # Extract all files
                zipf.extractall(self.temp_dir)
                
                # Read metadata
                metadata_path = self.temp_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    logger.info(f"Package from: {metadata['export_time']}")
                    return metadata
                else:
                    logger.warning("No metadata found in package")
                    return {}
                    
        except Exception as e:
            logger.error(f"Failed to extract package: {e}")
            return None
    
    def import_csv_file(self, csv_path: Path) -> int:
        """استيراد ملف CSV واحد"""
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            if df.empty:
                logger.warning(f"Empty file: {csv_path.name}")
                return 0
                
            # Connect to database
            conn = sqlite3.connect(self.db_path)
            
            # Import data
            imported = 0
            for _, row in df.iterrows():
                try:
                    conn.execute("""
                        INSERT OR REPLACE INTO price_data 
                        (symbol, timeframe, time, open, high, low, close, volume, spread)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row['symbol'], 
                        row['timeframe'], 
                        int(row['time']),
                        float(row['open']), 
                        float(row['high']), 
                        float(row['low']), 
                        float(row['close']),
                        int(row['tick_volume']), 
                        int(row.get('spread', 0))
                    ))
                    imported += 1
                except Exception as e:
                    logger.debug(f"Skip duplicate: {e}")
                    
            conn.commit()
            conn.close()
            
            logger.info(f"Imported {imported} records from {csv_path.name}")
            return imported
            
        except Exception as e:
            logger.error(f"Import error for {csv_path}: {e}")
            return 0
    
    def import_all_data(self, package_path: str) -> dict:
        """استيراد جميع البيانات من الحزمة"""
        # Extract package
        metadata = self.extract_package(package_path)
        if metadata is None:
            return {"success": False, "error": "Failed to extract package"}
            
        results = {
            "metadata": metadata,
            "files": {},
            "total_imported": 0
        }
        
        # Import each CSV file
        for csv_file in self.temp_dir.glob("*.csv"):
            imported = self.import_csv_file(csv_file)
            results["files"][csv_file.name] = imported
            results["total_imported"] += imported
            
        # Clean up temp files
        for temp_file in self.temp_dir.iterdir():
            temp_file.unlink()
            
        # Update database status
        self._update_import_status(metadata)
        
        return results
    
    def _update_import_status(self, metadata: dict):
        """تحديث حالة الاستيراد"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Create import history table if not exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS import_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    import_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    export_time TEXT,
                    mt5_account INTEGER,
                    files_count INTEGER,
                    metadata TEXT
                )
            """)
            
            # Add import record
            conn.execute("""
                INSERT INTO import_history (export_time, mt5_account, files_count, metadata)
                VALUES (?, ?, ?, ?)
            """, (
                metadata.get('export_time'),
                metadata.get('mt5_account'),
                len(metadata.get('files', {})),
                json.dumps(metadata)
            ))
            
            conn.commit()
            
        finally:
            conn.close()
    
    def get_data_summary(self) -> dict:
        """الحصول على ملخص البيانات"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Count records per symbol/timeframe
            cursor = conn.execute("""
                SELECT symbol, timeframe, COUNT(*) as count,
                       MIN(time) as start_time, 
                       MAX(time) as end_time
                FROM price_data
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
            """)
            
            summary = {}
            for row in cursor:
                key = f"{row[0]}_{row[1]}"
                summary[key] = {
                    "count": row[2],
                    "start": datetime.fromtimestamp(row[3]).strftime('%Y-%m-%d'),
                    "end": datetime.fromtimestamp(row[4]).strftime('%Y-%m-%d')
                }
                
            return summary
            
        finally:
            conn.close()

def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(description='Import MT5 data package')
    parser.add_argument('package', help='Path to data package zip file')
    parser.add_argument('--summary', action='store_true', help='Show data summary after import')
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("📥 MT5 DATA IMPORTER")
    print("="*60)
    
    if not Path(args.package).exists():
        print(f"❌ Package not found: {args.package}")
        sys.exit(1)
        
    importer = MT5DataImporter()
    
    print(f"\n📦 Importing from: {args.package}")
    results = importer.import_all_data(args.package)
    
    if results.get("total_imported", 0) > 0:
        print(f"\n✅ Import successful!")
        print(f"Total records imported: {results['total_imported']}")
        
        if args.summary:
            print("\n📊 Data Summary:")
            summary = importer.get_data_summary()
            
            for key, info in summary.items():
                symbol, tf = key.split('_')
                print(f"\n{symbol} {tf}:")
                print(f"  • Records: {info['count']}")
                print(f"  • Period: {info['start']} to {info['end']}")
    else:
        print("\n❌ Import failed or no new data")
        
    print("\n" + "="*60)

if __name__ == "__main__":
    main()