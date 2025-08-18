#!/usr/bin/env python3
"""
🚀 Ultimate Complete Forex ML System - النظام الشامل النهائي
✨ يحتوي على جميع الميزات المتقدمة والفرضيات والتعلم المستمر
📊 نظام متكامل للتدريب والتداول
"""

import os
import sys
import gc
import json
import sqlite3
import asyncio
import logging
import argparse
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ultimate_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UltimateCompleteSystem:
    """النظام الشامل النهائي"""
    
    def __init__(self):
        """تهيئة النظام"""
        logger.info("="*100)
        logger.info("🚀 Ultimate Complete Forex ML System")
        logger.info("="*100)
        
        self.db_path = "data/forex_ml.db"
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # إعدادات النظام
        self.config = {
            'training': {
                'use_all_features': True,
                'use_all_models': False,  # للأداء
                'min_data_points': 1000,
                'batch_size': 5000,
                'max_workers': 2
            },
            'continuous_learning': {
                'enabled': True,
                'update_interval': 3600,  # ساعة
                'min_accuracy': 0.55
            },
            'hypotheses': {
                'enabled': True,
                'max_active': 20,
                'validation_threshold': 0.6
            },
            'trading': {
                'paper_trading': True,
                'risk_per_trade': 0.02,
                'min_confidence': 0.65,
                'use_sl_tp': True
            }
        }
        
        # المكونات
        self.trainers = {}
        self.models = {}
        self.hypotheses = []
        self.performance_history = []
        
    def check_database(self) -> bool:
        """التحقق من قاعدة البيانات"""
        if not os.path.exists(self.db_path):
            logger.error(f"❌ Database not found: {self.db_path}")
            return False
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # التحقق من الجدول
            cursor.execute("""
                SELECT COUNT(*) FROM sqlite_master 
                WHERE type='table' AND name='price_data'
            """)
            
            if cursor.fetchone()[0] == 0:
                logger.error("❌ Table 'price_data' not found")
                conn.close()
                return False
            
            # عدد السجلات
            cursor.execute("SELECT COUNT(*) FROM price_data")
            count = cursor.fetchone()[0]
            logger.info(f"✅ Database OK - {count:,} records found")
            
            conn.close()
            return count > 0
            
        except Exception as e:
            logger.error(f"❌ Database error: {e}")
            return False
    
    def get_available_symbols(self, min_records: int = 5000) -> pd.DataFrame:
        """الحصول على الرموز المتاحة"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT symbol, timeframe, COUNT(*) as count,
                       MIN(time) as start_time, MAX(time) as end_time
                FROM price_data
                GROUP BY symbol, timeframe
                HAVING count >= ?
                ORDER BY count DESC
            """
            df = pd.read_sql_query(query, conn, params=(min_records,))
            conn.close()
            
            logger.info(f"📊 Found {len(df)} symbol/timeframe combinations")
            return df
            
        except Exception as e:
            logger.error(f"Error getting symbols: {e}")
            return pd.DataFrame()
    
    async def train_all_systems(self):
        """تدريب جميع الأنظمة"""
        logger.info("\n" + "="*80)
        logger.info("🎯 Starting Complete Training Process")
        logger.info("="*80)
        
        # 1. التحقق من قاعدة البيانات
        if not self.check_database():
            logger.error("❌ Cannot proceed without database")
            return
        
        # 2. الحصول على الرموز المتاحة
        available = self.get_available_symbols()
        if available.empty:
            logger.error("❌ No symbols with sufficient data")
            return
        
        # 3. تدريب النماذج الأساسية
        logger.info("\n📊 Phase 1: Basic Model Training")
        await self._train_basic_models(available)
        
        # 4. تدريب النماذج المتقدمة
        logger.info("\n📊 Phase 2: Advanced Model Training")
        await self._train_advanced_models(available)
        
        # 5. إنشاء الفرضيات
        logger.info("\n📊 Phase 3: Hypothesis Generation")
        await self._generate_hypotheses()
        
        # 6. بدء التعلم المستمر
        if self.config['continuous_learning']['enabled']:
            logger.info("\n📊 Phase 4: Starting Continuous Learning")
            asyncio.create_task(self._start_continuous_learning())
        
        logger.info("\n✅ Complete training process finished!")
    
    async def _train_basic_models(self, available: pd.DataFrame):
        """تدريب النماذج الأساسية"""
        try:
            # استيراد المدرب المحسن
            from train_optimized_system import OptimizedTrainer
            
            trainer = OptimizedTrainer()
            
            # تدريب أفضل 5 رموز
            for idx, row in available.head(5).iterrows():
                symbol = row['symbol']
                timeframe = row['timeframe']
                
                logger.info(f"\n🎯 Training {symbol} {timeframe}")
                result = trainer.train_symbol(symbol, timeframe)
                
                if result and result.get('best_accuracy', 0) > 0:
                    key = f"{symbol}_{timeframe}"
                    self.models[key] = result
                    logger.info(f"✅ Model trained: {result['best_accuracy']:.2%}")
                
                # تنظيف الذاكرة
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error in basic training: {e}")
    
    async def _train_advanced_models(self, available: pd.DataFrame):
        """تدريب النماذج المتقدمة"""
        try:
            # استيراد المدرب المتقدم
            from train_advanced_complete_full_features import UltimateAdvancedTrainer
            
            trainer = UltimateAdvancedTrainer()
            trainer.use_all_features = True
            trainer.use_all_models = False  # للأداء
            
            # تدريب رموز إضافية
            for idx, row in available[5:8].iterrows():  # 3 رموز إضافية
                symbol = row['symbol']
                timeframe = row['timeframe']
                
                logger.info(f"\n🎯 Advanced training {symbol} {timeframe}")
                result = trainer.train_symbol(symbol, timeframe)
                
                if result:
                    key = f"{symbol}_{timeframe}_advanced"
                    self.models[key] = result
                    logger.info(f"✅ Advanced model trained")
                
                gc.collect()
                
        except Exception as e:
            logger.error(f"Error in advanced training: {e}")
    
    async def _generate_hypotheses(self):
        """توليد الفرضيات"""
        try:
            from hypothesis_system import HypothesisManager
            
            manager = HypothesisManager()
            
            # بيانات السوق
            market_data = {
                'symbols': list(self.models.keys()),
                'timestamp': datetime.now()
            }
            
            # بيانات الأداء
            performance_data = {
                'models': self.models,
                'accuracy_avg': np.mean([m.get('best_accuracy', 0) for m in self.models.values()])
            }
            
            # توليد الفرضيات
            hypotheses = await manager.update_hypotheses(market_data, performance_data)
            self.hypotheses = hypotheses
            
            logger.info(f"✅ Generated {len(hypotheses)} hypotheses")
            
            # حفظ الفرضيات
            manager.save_hypotheses('hypotheses/current_hypotheses.json')
            
        except Exception as e:
            logger.error(f"Error generating hypotheses: {e}")
    
    async def _start_continuous_learning(self):
        """بدء التعلم المستمر"""
        try:
            from continuous_learning_ultimate import ContinuousLearningSystem
            
            system = ContinuousLearningSystem()
            
            # تكوين النظام
            system.learning_config['update_frequency'] = 'hourly'
            system.learning_config['performance_threshold'] = self.config['continuous_learning']['min_accuracy']
            
            # بدء التعلم
            logger.info("🔄 Starting continuous learning system...")
            await system.start_continuous_learning()
            
        except Exception as e:
            logger.error(f"Error in continuous learning: {e}")
    
    def generate_summary_report(self):
        """توليد تقرير ملخص"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'database': {
                'path': self.db_path,
                'status': 'OK' if self.check_database() else 'ERROR'
            },
            'models': {
                'total': len(self.models),
                'average_accuracy': np.mean([m.get('best_accuracy', 0) for m in self.models.values()]),
                'best_model': max(self.models.items(), key=lambda x: x[1].get('best_accuracy', 0))[0] if self.models else None
            },
            'hypotheses': {
                'total': len(self.hypotheses),
                'active': len([h for h in self.hypotheses if h.is_active]) if self.hypotheses else 0
            },
            'config': self.config
        }
        
        # حفظ التقرير
        report_path = Path('reports') / f'system_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Report saved to: {report_path}")
        
        # طباعة الملخص
        logger.info("\n" + "="*80)
        logger.info("📊 SYSTEM SUMMARY")
        logger.info("="*80)
        logger.info(f"Models trained: {report['models']['total']}")
        logger.info(f"Average accuracy: {report['models']['average_accuracy']:.2%}")
        logger.info(f"Best model: {report['models']['best_model']}")
        logger.info(f"Active hypotheses: {report['hypotheses']['active']}")
        logger.info("="*80)

async def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(description='Ultimate Complete Forex ML System')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'training', 'continuous', 'report'],
                       help='System mode')
    parser.add_argument('--config', type=str, help='Path to config file')
    
    args = parser.parse_args()
    
    # إنشاء النظام
    system = UltimateCompleteSystem()
    
    # تحميل إعدادات مخصصة إن وجدت
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            system.config.update(custom_config)
    
    try:
        if args.mode == 'full':
            # تشغيل النظام الكامل
            await system.train_all_systems()
            system.generate_summary_report()
            
        elif args.mode == 'training':
            # التدريب فقط
            await system._train_basic_models(system.get_available_symbols())
            await system._train_advanced_models(system.get_available_symbols())
            
        elif args.mode == 'continuous':
            # التعلم المستمر فقط
            await system._start_continuous_learning()
            
        elif args.mode == 'report':
            # توليد تقرير فقط
            system.generate_summary_report()
            
    except KeyboardInterrupt:
        logger.info("\n⏹️ System interrupted by user")
    except Exception as e:
        logger.error(f"❌ System error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # تشغيل النظام
    asyncio.run(main())