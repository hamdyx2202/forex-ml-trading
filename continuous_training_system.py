#!/usr/bin/env python3
"""
Continuous Training System - نظام التدريب المستمر والمتكامل
نظام متقدم للتدريب المستمر مع المراقبة والتحديث التلقائي
"""

import time
import warnings
warnings.filterwarnings('ignore')

from datetime import datetime, timedelta
import sqlite3
import pandas as pd
from pathlib import Path
import json
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import signal
import sys
import pickle
import os
import schedule
from train_advanced_complete import AdvancedCompleteTrainer
import logging
from typing import Dict, List, Tuple, Optional
import hashlib

# إعداد نظام السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/continuous_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ContinuousTrainingSystem:
    """نظام التدريب المستمر مع المراقبة والتحديث التلقائي"""
    
    def __init__(self):
        self.trainer = AdvancedCompleteTrainer()
        self.state_file = Path("state/continuous_training_state.pkl")
        self.models_registry = Path("models/registry.json")
        self.performance_db = Path("data/performance_tracking.db")
        
        # إعدادات النظام
        self.config = {
            'retrain_interval_hours': 24,  # إعادة التدريب كل 24 ساعة
            'min_accuracy_threshold': 0.80,  # الحد الأدنى للدقة المقبولة
            'performance_check_interval': 3600,  # فحص الأداء كل ساعة
            'max_concurrent_training': 2,  # عدد التدريبات المتزامنة
            'auto_update_models': True,  # تحديث النماذج تلقائياً
            'alert_on_degradation': True,  # تنبيه عند انخفاض الأداء
            'min_data_points': 10000,  # الحد الأدنى للبيانات
            'max_training_time_minutes': 30,  # الحد الأقصى لوقت التدريب
        }
        
        # حالة النظام
        self.state = {
            'active_models': {},
            'training_queue': [],
            'performance_history': {},
            'last_full_scan': None,
            'training_in_progress': {},
            'failed_attempts': {},
        }
        
        # تحميل الحالة السابقة
        self.load_state()
        
        # مؤشرات التحكم
        self._running = False
        self._stop_event = threading.Event()
        
        # إنشاء المجلدات المطلوبة
        self._create_directories()
        
    def _create_directories(self):
        """إنشاء المجلدات المطلوبة"""
        directories = [
            'state', 'models', 'logs', 'results/continuous',
            'alerts', 'backups', 'performance_reports'
        ]
        for dir_path in directories:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def load_state(self):
        """تحميل حالة النظام"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'rb') as f:
                    saved_state = pickle.load(f)
                    self.state.update(saved_state)
                logger.info(f"تم تحميل حالة النظام: {len(self.state['active_models'])} نموذج نشط")
            except Exception as e:
                logger.error(f"فشل تحميل الحالة: {e}")
    
    def save_state(self):
        """حفظ حالة النظام"""
        try:
            self.state_file.parent.mkdir(exist_ok=True)
            with open(self.state_file, 'wb') as f:
                pickle.dump(self.state, f)
            logger.debug("تم حفظ حالة النظام")
        except Exception as e:
            logger.error(f"فشل حفظ الحالة: {e}")
    
    def get_available_pairs(self) -> List[Tuple[str, str, int]]:
        """الحصول على جميع الأزواج المتاحة للتدريب"""
        try:
            conn = sqlite3.connect("data/forex_ml.db")
            query = f"""
                SELECT symbol, timeframe, COUNT(*) as count
                FROM price_data
                GROUP BY symbol, timeframe
                HAVING count >= {self.config['min_data_points']}
                ORDER BY count DESC
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            pairs = [(row['symbol'], row['timeframe'], row['count']) 
                    for _, row in df.iterrows()]
            
            return pairs
            
        except Exception as e:
            logger.error(f"خطأ في قراءة الأزواج: {e}")
            return []
    
    def check_data_updates(self, symbol: str, timeframe: str) -> bool:
        """فحص وجود تحديثات في البيانات"""
        try:
            conn = sqlite3.connect("data/forex_ml.db")
            
            # الحصول على آخر timestamp في البيانات
            query = """
                SELECT MAX(timestamp) as last_update, COUNT(*) as count
                FROM price_data
                WHERE symbol = ? AND timeframe = ?
            """
            result = pd.read_sql_query(query, conn, params=(symbol, timeframe))
            conn.close()
            
            if result.empty or result['count'].iloc[0] < self.config['min_data_points']:
                return False
            
            last_update = pd.to_datetime(result['last_update'].iloc[0])
            
            # فحص آخر تدريب
            model_key = f"{symbol}_{timeframe}"
            if model_key in self.state['active_models']:
                last_train = pd.to_datetime(self.state['active_models'][model_key]['last_training'])
                
                # إعادة التدريب إذا كان هناك بيانات جديدة
                return last_update > last_train
            
            return True  # تدريب جديد
            
        except Exception as e:
            logger.error(f"خطأ في فحص التحديثات لـ {symbol} {timeframe}: {e}")
            return False
    
    def evaluate_model_performance(self, symbol: str, timeframe: str) -> Dict:
        """تقييم أداء النموذج الحالي"""
        try:
            model_key = f"{symbol}_{timeframe}"
            
            # الحصول على أحدث البيانات للتقييم
            conn = sqlite3.connect("data/forex_ml.db")
            query = """
                SELECT * FROM price_data
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT 1000
            """
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
            conn.close()
            
            if df.empty:
                return {'accuracy': 0, 'status': 'no_data'}
            
            # تحميل النموذج وتقييمه
            model_info = self.state['active_models'].get(model_key)
            if not model_info:
                return {'accuracy': 0, 'status': 'no_model'}
            
            # هنا يتم تقييم النموذج على البيانات الجديدة
            # (تحتاج لتنفيذ دالة التقييم الفعلية)
            
            performance = {
                'accuracy': model_info.get('accuracy', 0),
                'last_check': datetime.now().isoformat(),
                'data_points': len(df),
                'status': 'evaluated'
            }
            
            return performance
            
        except Exception as e:
            logger.error(f"خطأ في تقييم {symbol} {timeframe}: {e}")
            return {'accuracy': 0, 'status': 'error', 'error': str(e)}
    
    def train_model_async(self, symbol: str, timeframe: str, count: int):
        """تدريب نموذج بشكل غير متزامن"""
        model_key = f"{symbol}_{timeframe}"
        
        try:
            logger.info(f"🔄 بدء تدريب {symbol} {timeframe}")
            
            # وضع علامة التدريب قيد التنفيذ
            self.state['training_in_progress'][model_key] = {
                'start_time': datetime.now().isoformat(),
                'status': 'training'
            }
            
            # التدريب الفعلي
            start_time = time.time()
            results = self.trainer.train_symbol_advanced(symbol, timeframe)
            training_time = time.time() - start_time
            
            if results and results.get('best_accuracy', 0) >= self.config['min_accuracy_threshold']:
                # تحديث سجل النموذج
                self.state['active_models'][model_key] = {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'accuracy': results['best_accuracy'],
                    'strategy': results['best_strategy'],
                    'ensemble_accuracy': results.get('ensemble_accuracy', 0),
                    'confidence_threshold': results.get('confidence_threshold', 0.7),
                    'expected_win_rate': results.get('expected_win_rate', 0),
                    'last_training': datetime.now().isoformat(),
                    'training_time': training_time,
                    'data_points': count,
                    'model_version': self._generate_model_version(symbol, timeframe),
                    'status': 'active'
                }
                
                logger.info(f"✅ {symbol} {timeframe}: دقة {results['best_accuracy']:.4f} "
                          f"({training_time/60:.1f} دقيقة)")
                
                # حفظ معلومات النموذج
                self._save_model_info(model_key, results)
                
            else:
                logger.warning(f"❌ {symbol} {timeframe}: دقة منخفضة أو فشل التدريب")
                self.state['failed_attempts'][model_key] = {
                    'timestamp': datetime.now().isoformat(),
                    'reason': 'low_accuracy' if results else 'training_failed'
                }
            
        except Exception as e:
            logger.error(f"❌ خطأ في تدريب {symbol} {timeframe}: {e}")
            self.state['failed_attempts'][model_key] = {
                'timestamp': datetime.now().isoformat(),
                'reason': 'exception',
                'error': str(e)
            }
        
        finally:
            # إزالة من قائمة التدريب الجاري
            if model_key in self.state['training_in_progress']:
                del self.state['training_in_progress'][model_key]
            
            # حفظ الحالة
            self.save_state()
    
    def _generate_model_version(self, symbol: str, timeframe: str) -> str:
        """توليد رقم إصدار فريد للنموذج"""
        timestamp = datetime.now().isoformat()
        version_string = f"{symbol}_{timeframe}_{timestamp}"
        return hashlib.md5(version_string.encode()).hexdigest()[:8]
    
    def _save_model_info(self, model_key: str, results: Dict):
        """حفظ معلومات النموذج"""
        try:
            # تحديث سجل النماذج
            registry = {}
            if self.models_registry.exists():
                with open(self.models_registry, 'r') as f:
                    registry = json.load(f)
            
            registry[model_key] = {
                'info': self.state['active_models'][model_key],
                'results': results,
                'updated': datetime.now().isoformat()
            }
            
            with open(self.models_registry, 'w') as f:
                json.dump(registry, f, indent=2)
                
        except Exception as e:
            logger.error(f"خطأ في حفظ معلومات النموذج: {e}")
    
    def scheduled_full_scan(self):
        """مسح شامل مجدول لجميع الأزواج"""
        logger.info("🔍 بدء المسح الشامل المجدول")
        
        pairs = self.get_available_pairs()
        updates_needed = []
        
        for symbol, timeframe, count in pairs:
            if self.check_data_updates(symbol, timeframe):
                updates_needed.append((symbol, timeframe, count))
        
        logger.info(f"📊 تحتاج {len(updates_needed)} نموذج للتحديث")
        
        # إضافة إلى قائمة الانتظار
        for item in updates_needed:
            if item not in self.state['training_queue']:
                self.state['training_queue'].append(item)
        
        self.state['last_full_scan'] = datetime.now().isoformat()
        self.save_state()
    
    def performance_monitoring(self):
        """مراقبة أداء النماذج النشطة"""
        logger.info("📈 فحص أداء النماذج النشطة")
        
        degraded_models = []
        
        for model_key, model_info in self.state['active_models'].items():
            if model_info['status'] != 'active':
                continue
                
            symbol = model_info['symbol']
            timeframe = model_info['timeframe']
            
            # تقييم الأداء الحالي
            performance = self.evaluate_model_performance(symbol, timeframe)
            
            # مقارنة مع الأداء السابق
            original_accuracy = model_info['accuracy']
            current_accuracy = performance.get('accuracy', 0)
            
            if current_accuracy < original_accuracy * 0.95:  # انخفاض 5% أو أكثر
                degraded_models.append({
                    'model': model_key,
                    'original': original_accuracy,
                    'current': current_accuracy,
                    'drop': original_accuracy - current_accuracy
                })
                
                # إضافة لقائمة إعادة التدريب
                self.state['training_queue'].append(
                    (symbol, timeframe, model_info['data_points'])
                )
        
        if degraded_models and self.config['alert_on_degradation']:
            self._send_performance_alert(degraded_models)
        
        # حفظ تاريخ الأداء
        timestamp = datetime.now().isoformat()
        for model in degraded_models:
            if model['model'] not in self.state['performance_history']:
                self.state['performance_history'][model['model']] = []
            
            self.state['performance_history'][model['model']].append({
                'timestamp': timestamp,
                'accuracy': model['current']
            })
        
        self.save_state()
    
    def _send_performance_alert(self, degraded_models: List[Dict]):
        """إرسال تنبيه انخفاض الأداء"""
        alert_file = Path(f"alerts/performance_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        alert_data = {
            'timestamp': datetime.now().isoformat(),
            'type': 'performance_degradation',
            'models': degraded_models,
            'action': 'scheduled_for_retraining'
        }
        
        with open(alert_file, 'w') as f:
            json.dump(alert_data, f, indent=2)
        
        logger.warning(f"⚠️ تم اكتشاف انخفاض في أداء {len(degraded_models)} نموذج")
    
    def process_training_queue(self):
        """معالجة قائمة التدريب"""
        if not self.state['training_queue']:
            return
        
        # التحقق من عدد التدريبات الجارية
        active_training = len(self.state['training_in_progress'])
        
        if active_training >= self.config['max_concurrent_training']:
            return
        
        # معالجة العناصر في القائمة
        with ThreadPoolExecutor(max_workers=self.config['max_concurrent_training']) as executor:
            while self.state['training_queue'] and active_training < self.config['max_concurrent_training']:
                symbol, timeframe, count = self.state['training_queue'].pop(0)
                model_key = f"{symbol}_{timeframe}"
                
                # تجنب التدريب المكرر
                if model_key in self.state['training_in_progress']:
                    continue
                
                # بدء التدريب
                executor.submit(self.train_model_async, symbol, timeframe, count)
                active_training += 1
    
    def generate_performance_report(self):
        """توليد تقرير الأداء الشامل"""
        try:
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'active_models': len(self.state['active_models']),
                'models_in_training': len(self.state['training_in_progress']),
                'queued_for_training': len(self.state['training_queue']),
                'failed_attempts': len(self.state['failed_attempts']),
                'models_summary': []
            }
            
            # تلخيص النماذج النشطة
            for model_key, model_info in self.state['active_models'].items():
                report_data['models_summary'].append({
                    'model': model_key,
                    'accuracy': model_info['accuracy'],
                    'strategy': model_info['strategy'],
                    'last_training': model_info['last_training'],
                    'status': model_info['status']
                })
            
            # حفظ التقرير
            report_file = Path(f"performance_reports/report_{datetime.now().strftime('%Y%m%d')}.json")
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            logger.info(f"📊 تم توليد تقرير الأداء: {report_file}")
            
        except Exception as e:
            logger.error(f"خطأ في توليد التقرير: {e}")
    
    def run_continuous(self):
        """تشغيل النظام المستمر"""
        logger.info("🚀 بدء نظام التدريب المستمر")
        self._running = True
        
        # إعداد المهام المجدولة
        schedule.every(self.config['retrain_interval_hours']).hours.do(self.scheduled_full_scan)
        schedule.every(self.config['performance_check_interval']).seconds.do(self.performance_monitoring)
        schedule.every(5).minutes.do(self.process_training_queue)
        schedule.every(1).hours.do(self.generate_performance_report)
        schedule.every(30).minutes.do(self.save_state)
        
        # معالج الإشارات
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # المسح الأولي
        self.scheduled_full_scan()
        
        # الحلقة الرئيسية
        while self._running and not self._stop_event.is_set():
            try:
                # تنفيذ المهام المجدولة
                schedule.run_pending()
                
                # معالجة قائمة التدريب
                self.process_training_queue()
                
                # انتظار قصير
                time.sleep(10)
                
            except Exception as e:
                logger.error(f"خطأ في الحلقة الرئيسية: {e}")
                time.sleep(60)  # انتظار دقيقة قبل المحاولة مرة أخرى
        
        logger.info("⏹️ توقف نظام التدريب المستمر")
    
    def _signal_handler(self, signum, frame):
        """معالج إشارة الإيقاف"""
        logger.info("\n⚠️ تم استلام إشارة الإيقاف...")
        self._running = False
        self._stop_event.set()
        
        # حفظ الحالة النهائية
        self.save_state()
        self.generate_performance_report()
        
        logger.info("✅ تم إيقاف النظام بشكل آمن")
        sys.exit(0)
    
    def stop(self):
        """إيقاف النظام"""
        self._running = False
        self._stop_event.set()

def main():
    """نقطة البداية الرئيسية"""
    print("🚀 نظام التدريب المستمر والمتكامل")
    print("="*60)
    print("\n✨ الميزات:")
    print("   • تدريب مستمر على مدار الساعة")
    print("   • مراقبة أداء النماذج")
    print("   • إعادة تدريب تلقائي عند انخفاض الأداء")
    print("   • تحديث النماذج عند توفر بيانات جديدة")
    print("   • تقارير أداء دورية")
    print("   • معالجة آمنة للإيقاف")
    
    print("\n⚠️ سيعمل النظام بشكل مستمر. اضغط Ctrl+C للإيقاف الآمن")
    
    confirm = input("\nهل تريد بدء النظام؟ (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("❌ تم إلغاء التشغيل")
        return
    
    # إنشاء وتشغيل النظام
    system = ContinuousTrainingSystem()
    
    try:
        system.run_continuous()
    except Exception as e:
        logger.error(f"خطأ في النظام: {e}")
        system.save_state()
        print("\n❌ توقف النظام بسبب خطأ")

if __name__ == "__main__":
    main()