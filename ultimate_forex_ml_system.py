#!/usr/bin/env python3
"""
🚀 Ultimate Forex ML Trading System - النظام النهائي الشامل
✨ يدمج جميع المكونات المتقدمة
📊 تدريب - تعلم مستمر - فرضيات - تداول
"""

import os
import sys
import asyncio
import logging
import argparse
from datetime import datetime
from pathlib import Path
import json

# إضافة المسار
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, 'src'))

# Import all components
from train_advanced_complete_ultimate import UltimateAdvancedTrainer
from train_advanced_complete_full_features import UltimateAdvancedTrainer as UltimateFeaturesTrainer
from continuous_learning_ultimate import ContinuousLearningSystem
from hypothesis_system import HypothesisManager
# Import with fallback for missing modules
try:
    from instrument_manager import InstrumentManager
except ImportError:
    print("Warning: InstrumentManager not found, using dummy implementation")
    class InstrumentManager:
        def __init__(self):
            pass
        def get_all_instruments(self):
            return []

try:
    from performance_tracker import PerformanceTracker
except ImportError:
    print("Warning: PerformanceTracker not found, using dummy implementation")
    class PerformanceTracker:
        def __init__(self):
            self.metrics = {}
        def get_recent_performance(self):
            return {'accuracy': 0.0, 'profit': 0.0}

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('ultimate_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UltimateForexMLSystem:
    """النظام الشامل النهائي"""
    
    def __init__(self):
        """تهيئة النظام"""
        logger.info("🚀 Initializing Ultimate Forex ML System...")
        
        # المكونات الرئيسية
        self.advanced_trainer = UltimateAdvancedTrainer()
        self.features_trainer = UltimateFeaturesTrainer()
        self.continuous_learner = ContinuousLearningSystem()
        self.hypothesis_manager = HypothesisManager()
        self.instrument_manager = InstrumentManager()
        self.performance_tracker = PerformanceTracker()
        
        # الإعدادات
        self.config = self._load_config()
        
        # حالة النظام
        self.is_running = False
        self.mode = 'training'  # training, continuous, live
        
        logger.info("✅ System initialized successfully!")
    
    def _load_config(self) -> dict:
        """تحميل الإعدادات"""
        default_config = {
            'training': {
                'initial_training': True,
                'retrain_interval': 24,  # hours
                'min_accuracy': 0.85,
                'use_all_features': True,
                'use_all_models': True,
                'parallel_training': True,
                'max_workers': 4
            },
            'continuous_learning': {
                'enabled': True,
                'update_frequency': 'hourly',
                'performance_threshold': 0.55,
                'adaptive_learning': True,
                'hypothesis_testing': True,
                'max_active_hypotheses': 20
            },
            'trading': {
                'paper_trading': True,
                'live_trading': False,
                'risk_per_trade': 0.02,
                'max_concurrent_trades': 10,
                'min_confidence': 0.7,
                'use_sl_tp': True,
                'trailing_stop': True
            },
            'symbols': {
                'forex_majors': [
                    'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF',
                    'AUD/USD', 'USD/CAD', 'NZD/USD'
                ],
                'forex_minors': [
                    'EUR/GBP', 'EUR/JPY', 'GBP/JPY', 'EUR/CHF',
                    'EUR/AUD', 'EUR/CAD', 'EUR/NZD', 'GBP/CHF',
                    'GBP/AUD', 'GBP/CAD', 'GBP/NZD', 'AUD/JPY',
                    'AUD/CHF', 'AUD/CAD', 'AUD/NZD', 'CAD/JPY',
                    'CAD/CHF', 'NZD/JPY', 'NZD/CHF', 'CHF/JPY'
                ],
                'commodities': [
                    'XAU/USD', 'XAG/USD', 'WTI/USD', 'BRENT/USD'
                ],
                'crypto': [
                    'BTC/USD', 'ETH/USD', 'BNB/USD', 'XRP/USD'
                ],
                'indices': [
                    'US30', 'US500', 'NAS100', 'DE30', 'UK100'
                ]
            },
            'timeframes': ['M5', 'M15', 'M30', 'H1', 'H4', 'D1'],
            'features': {
                'technical_indicators': True,
                'market_structure': True,
                'price_action': True,
                'volume_analysis': True,
                'sentiment_analysis': True,
                'intermarket_analysis': True,
                'machine_learning_features': True,
                'custom_features': True
            }
        }
        
        # محاولة تحميل إعدادات مخصصة
        config_path = Path('config/system_config.json')
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    custom_config = json.load(f)
                # دمج الإعدادات
                default_config.update(custom_config)
                logger.info("📁 Loaded custom configuration")
            except Exception as e:
                logger.warning(f"⚠️ Failed to load custom config: {e}")
        
        return default_config
    
    async def run_initial_training(self):
        """تشغيل التدريب الأولي"""
        logger.info("\n" + "="*80)
        logger.info("🎯 Starting Initial Training Phase")
        logger.info("="*80)
        
        # الحصول على جميع الرموز للتدريب
        all_symbols = []
        if self.config['training']['initial_training']:
            all_symbols.extend(self.config['symbols']['forex_majors'])
            all_symbols.extend(self.config['symbols']['forex_minors'])
            all_symbols.extend(self.config['symbols']['commodities'])
            all_symbols.extend(self.config['symbols']['crypto'])
            all_symbols.extend(self.config['symbols']['indices'])
        
        # إزالة التكرار
        all_symbols = list(set(all_symbols))
        logger.info(f"📊 Total symbols to train: {len(all_symbols)}")
        
        # تدريب باستخدام النظام المتقدم
        logger.info("\n🔧 Phase 1: Advanced Training with Full Features")
        try:
            # تعديل إعدادات المدرب
            self.features_trainer.use_all_features = self.config['training']['use_all_features']
            self.features_trainer.use_all_models = self.config['training']['use_all_models']
            self.features_trainer.max_workers = self.config['training']['max_workers']
            
            # بدء التدريب
            training_results = await self._train_all_symbols_async(all_symbols)
            
            # حفظ النتائج
            self._save_training_results(training_results)
            
            logger.info("✅ Initial training completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Error in initial training: {str(e)}")
            raise
    
    async def _train_all_symbols_async(self, symbols: list) -> dict:
        """تدريب جميع الرموز بشكل متوازي"""
        results = {}
        
        # تقسيم الرموز على دفعات
        batch_size = self.config['training']['max_workers']
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            logger.info(f"\n📦 Training batch {i//batch_size + 1}/{len(symbols)//batch_size + 1}")
            
            # تدريب الدفعة بشكل متوازي
            tasks = []
            for symbol in batch:
                for timeframe in self.config['timeframes']:
                    task = asyncio.create_task(
                        self._train_symbol_async(symbol, timeframe)
                    )
                    tasks.append((symbol, timeframe, task))
            
            # انتظار اكتمال الدفعة
            for symbol, timeframe, task in tasks:
                try:
                    result = await task
                    if result:
                        key = f"{symbol}_{timeframe}"
                        results[key] = result
                except Exception as e:
                    logger.error(f"❌ Error training {symbol} {timeframe}: {str(e)}")
        
        return results
    
    async def _train_symbol_async(self, symbol: str, timeframe: str) -> dict:
        """تدريب رمز واحد"""
        loop = asyncio.get_event_loop()
        
        # تشغيل التدريب في thread منفصل
        try:
            result = await loop.run_in_executor(
                None,
                self.features_trainer.train_symbol,
                symbol,
                timeframe
            )
            return result
        except Exception as e:
            logger.error(f"Error training {symbol} {timeframe}: {str(e)}")
            return None
    
    async def start_continuous_learning(self):
        """بدء التعلم المستمر"""
        logger.info("\n" + "="*80)
        logger.info("🔄 Starting Continuous Learning Phase")
        logger.info("="*80)
        
        if not self.config['continuous_learning']['enabled']:
            logger.info("⚠️ Continuous learning is disabled in config")
            return
        
        try:
            # بدء نظام الفرضيات
            if self.config['continuous_learning']['hypothesis_testing']:
                logger.info("🔬 Starting hypothesis system...")
                asyncio.create_task(self._run_hypothesis_system())
            
            # بدء التعلم المستمر
            logger.info("🎓 Starting continuous learning system...")
            await self.continuous_learner.start_continuous_learning()
            
        except Exception as e:
            logger.error(f"❌ Error in continuous learning: {str(e)}")
            raise
    
    async def _run_hypothesis_system(self):
        """تشغيل نظام الفرضيات"""
        while self.is_running:
            try:
                # تحديث الفرضيات
                market_data = await self._get_current_market_data()
                performance_data = self.performance_tracker.get_recent_performance()
                
                hypotheses = await self.hypothesis_manager.update_hypotheses(
                    market_data,
                    performance_data
                )
                
                logger.info(f"🔬 Active hypotheses: {len(hypotheses)}")
                
                # حفظ الفرضيات
                self.hypothesis_manager.save_hypotheses('hypotheses/current_hypotheses.json')
                
                # انتظار الدورة التالية
                await asyncio.sleep(3600)  # كل ساعة
                
            except Exception as e:
                logger.error(f"Error in hypothesis system: {str(e)}")
                await asyncio.sleep(300)  # انتظار 5 دقائق عند الخطأ
    
    async def _get_current_market_data(self) -> dict:
        """الحصول على بيانات السوق الحالية"""
        # هنا يتم جلب البيانات الحالية
        # هذا مثال مبسط
        return {
            'timestamp': datetime.now(),
            'symbols': self.config['symbols']['forex_majors'],
            'market_state': 'normal'
        }
    
    def _save_training_results(self, results: dict):
        """حفظ نتائج التدريب"""
        results_dir = Path('results')
        results_dir.mkdir(exist_ok=True)
        
        # تلخيص النتائج
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_models': len(results),
            'successful_models': sum(1 for r in results.values() if r.get('best_accuracy', 0) > 0.85),
            'average_accuracy': sum(r.get('best_accuracy', 0) for r in results.values()) / len(results) if results else 0,
            'models': {}
        }
        
        # تفاصيل كل نموذج
        for key, result in results.items():
            summary['models'][key] = {
                'best_strategy': result.get('best_strategy'),
                'best_accuracy': result.get('best_accuracy', 0),
                'strategies': {
                    name: {
                        'accuracy': strategy.get('accuracy', 0),
                        'precision': strategy.get('precision', 0),
                        'recall': strategy.get('recall', 0),
                        'f1': strategy.get('f1', 0)
                    }
                    for name, strategy in result.get('strategies', {}).items()
                }
            }
        
        # حفظ الملخص
        summary_path = results_dir / f'training_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"📊 Training summary saved to: {summary_path}")
        logger.info(f"✅ Successful models: {summary['successful_models']}/{summary['total_models']}")
        logger.info(f"📈 Average accuracy: {summary['average_accuracy']:.2%}")
    
    async def start_system(self, mode: str = 'full'):
        """بدء النظام"""
        self.is_running = True
        self.mode = mode
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 Starting Ultimate Forex ML System in {mode.upper()} mode")
        logger.info(f"{'='*80}")
        
        try:
            if mode in ['full', 'training']:
                # التدريب الأولي
                await self.run_initial_training()
            
            if mode in ['full', 'continuous']:
                # التعلم المستمر
                await self.start_continuous_learning()
            
            if mode in ['full', 'trading']:
                # التداول (يحتاج تطوير إضافي)
                logger.info("📈 Trading mode is under development...")
            
        except KeyboardInterrupt:
            logger.info("\n⏹️ System interrupted by user")
            await self.stop_system()
        except Exception as e:
            logger.error(f"❌ System error: {str(e)}")
            await self.stop_system()
            raise
    
    async def stop_system(self):
        """إيقاف النظام"""
        logger.info("🛑 Stopping system...")
        self.is_running = False
        
        # إيقاف التعلم المستمر
        if hasattr(self.continuous_learner, 'is_learning'):
            self.continuous_learner.is_learning = False
        
        logger.info("✅ System stopped successfully")
    
    def get_system_status(self) -> dict:
        """الحصول على حالة النظام"""
        return {
            'is_running': self.is_running,
            'mode': self.mode,
            'timestamp': datetime.now().isoformat(),
            'components': {
                'advanced_trainer': 'ready',
                'features_trainer': 'ready',
                'continuous_learner': 'ready' if self.continuous_learner else 'not initialized',
                'hypothesis_manager': 'ready' if self.hypothesis_manager else 'not initialized'
            },
            'config': self.config
        }

async def main():
    """الدالة الرئيسية"""
    parser = argparse.ArgumentParser(description='Ultimate Forex ML Trading System')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'training', 'continuous', 'trading'],
                       help='System mode')
    parser.add_argument('--config', type=str, default='config/system_config.json',
                       help='Path to configuration file')
    
    args = parser.parse_args()
    
    # إنشاء النظام
    system = UltimateForexMLSystem()
    
    # عرض حالة النظام
    status = system.get_system_status()
    logger.info(f"\n📊 System Status:")
    logger.info(json.dumps(status, indent=2))
    
    # بدء النظام
    await system.start_system(mode=args.mode)

if __name__ == "__main__":
    # تشغيل النظام
    asyncio.run(main())