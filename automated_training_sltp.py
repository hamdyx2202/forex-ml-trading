#!/usr/bin/env python3
"""
Automated Training System with SL/TP Support
نظام التدريب الآلي مع دعم وقف الخسارة والأهداف
"""

import schedule
import time
from datetime import datetime, timedelta
from pathlib import Path
import json
from loguru import logger
import threading
import sqlite3
import pandas as pd
import requests

# إضافة المسار
import sys
sys.path.append(str(Path(__file__).parent))

from integrated_training_sltp import IntegratedTrainingSystemSLTP
from performance_tracker import PerformanceTracker
from instrument_manager import InstrumentManager

class AutomatedTrainingSLTP:
    """نظام تدريب آلي يعمل 24/7 مع تحسين SL/TP"""
    
    def __init__(self):
        self.training_system = IntegratedTrainingSystemSLTP()
        self.performance_tracker = PerformanceTracker()
        self.instrument_manager = InstrumentManager()
        
        self.config = {
            'training_schedule': {
                'daily_training_time': '03:00',  # وقت التدريب اليومي
                'weekly_full_training': 'sunday',  # يوم التدريب الكامل
                'performance_check_interval': 60,  # دقائق
                'emergency_retrain_threshold': 0.45,  # حد الأداء للتدريب الطارئ
            },
            'notification_settings': {
                'enable_notifications': True,
                'webhook_url': None,  # يمكن إضافة webhook
                'email_notifications': False,
            },
            'training_priorities': {
                'high_priority': ['EURUSD', 'GBPUSD', 'XAUUSD', 'US30'],
                'medium_priority': ['USDJPY', 'AUDUSD', 'NZDUSD', 'EURJPY'],
                'low_priority': []  # باقي الأزواج
            },
            'resource_management': {
                'max_concurrent_training': 3,
                'cpu_usage_limit': 80,  # نسبة مئوية
                'memory_usage_limit': 70,  # نسبة مئوية
            }
        }
        
        self.training_history = []
        self.is_running = False
        
    def start(self):
        """بدء النظام الآلي"""
        logger.info("🤖 Starting Automated Training System with SL/TP optimization...")
        
        self.is_running = True
        
        # جدولة المهام
        self.schedule_tasks()
        
        # بدء خيط مراقبة الأداء
        performance_thread = threading.Thread(target=self.performance_monitor_loop, daemon=True)
        performance_thread.start()
        
        # بدء خيط الجدولة
        schedule_thread = threading.Thread(target=self.schedule_loop, daemon=True)
        schedule_thread.start()
        
        logger.info("✅ Automated training system is running!")
        
    def schedule_tasks(self):
        """جدولة مهام التدريب"""
        # تدريب يومي
        schedule.every().day.at(self.config['training_schedule']['daily_training_time']).do(
            self.daily_training_task
        )
        
        # تدريب أسبوعي شامل
        getattr(schedule.every(), self.config['training_schedule']['weekly_full_training']).at("02:00").do(
            self.weekly_full_training_task
        )
        
        # تنظيف يومي
        schedule.every().day.at("23:00").do(self.cleanup_old_models)
        
        # تقرير يومي
        schedule.every().day.at("08:00").do(self.generate_daily_report)
        
    def schedule_loop(self):
        """حلقة تنفيذ المهام المجدولة"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # فحص كل دقيقة
            
    def performance_monitor_loop(self):
        """مراقبة مستمرة للأداء"""
        while self.is_running:
            try:
                self.check_model_performance()
                time.sleep(self.config['training_schedule']['performance_check_interval'] * 60)
            except Exception as e:
                logger.error(f"Error in performance monitoring: {str(e)}")
                time.sleep(300)  # انتظار 5 دقائق عند الخطأ
                
    def daily_training_task(self):
        """مهمة التدريب اليومية"""
        logger.info("📅 Starting daily training task...")
        
        try:
            # تحديد الأزواج التي تحتاج تدريب
            pairs_to_train = self.identify_pairs_needing_training()
            
            if pairs_to_train:
                # ترتيب حسب الأولوية
                prioritized_pairs = self.prioritize_pairs(pairs_to_train)
                
                # تدريب الأزواج
                results = self.train_pairs_batch(prioritized_pairs)
                
                # حفظ السجل
                self.log_training_session(results, 'daily')
                
                # إرسال إشعار
                self.send_notification(f"Daily training completed: {len(results)} models updated")
            else:
                logger.info("No models need training today")
                
        except Exception as e:
            logger.error(f"Error in daily training: {str(e)}")
            self.send_notification(f"Daily training failed: {str(e)}", level='error')
            
    def weekly_full_training_task(self):
        """تدريب أسبوعي شامل"""
        logger.info("📅 Starting weekly full training...")
        
        try:
            # تدريب جميع الأدوات المهمة
            instrument_types = ['forex_major', 'metals', 'indices']
            
            results = self.training_system.train_all_instruments(
                instrument_types=instrument_types,
                force_retrain=True
            )
            
            # تحليل النتائج
            success_rate = len([r for r in results if r.get('success', False)]) / len(results)
            
            self.log_training_session(results, 'weekly_full')
            
            self.send_notification(
                f"Weekly training completed: {len(results)} models, "
                f"Success rate: {success_rate:.1%}"
            )
            
        except Exception as e:
            logger.error(f"Error in weekly training: {str(e)}")
            self.send_notification(f"Weekly training failed: {str(e)}", level='error')
            
    def check_model_performance(self):
        """فحص أداء النماذج"""
        logger.info("🔍 Checking model performance...")
        
        poor_performing = []
        
        # فحص جميع النماذج النشطة
        models_dir = Path("models/unified_sltp")
        
        for model_file in models_dir.glob("*.pkl"):
            try:
                # استخراج معلومات النموذج
                parts = model_file.stem.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    timeframe = parts[1]
                    
                    # فحص الأداء
                    performance = self.performance_tracker.get_pair_performance(
                        symbol, timeframe
                    )
                    
                    if performance and performance.get('win_rate', 0) < self.config['training_schedule']['emergency_retrain_threshold']:
                        poor_performing.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'win_rate': performance.get('win_rate', 0),
                            'reason': 'low_win_rate'
                        })
                        
            except Exception as e:
                logger.error(f"Error checking {model_file}: {str(e)}")
                
        # تدريب طارئ للنماذج ضعيفة الأداء
        if poor_performing:
            logger.warning(f"Found {len(poor_performing)} poor performing models")
            self.emergency_retrain(poor_performing)
            
    def emergency_retrain(self, poor_models):
        """إعادة تدريب طارئة للنماذج ضعيفة الأداء"""
        logger.warning("🚨 Starting emergency retraining...")
        
        # تحديد عدد النماذج للتدريب بناءً على الموارد
        max_models = min(len(poor_models), self.config['resource_management']['max_concurrent_training'])
        
        # اختيار الأسوأ أداءً
        worst_models = sorted(poor_models, key=lambda x: x['win_rate'])[:max_models]
        
        for model in worst_models:
            try:
                logger.info(f"Retraining {model['symbol']} {model['timeframe']} (Win rate: {model['win_rate']:.1%})")
                
                success = self.training_system.train_single_model(
                    model['symbol'],
                    model['timeframe'],
                    self.instrument_manager.get_instrument_type(model['symbol'])
                )
                
                if success:
                    logger.info(f"✅ Successfully retrained {model['symbol']} {model['timeframe']}")
                    
            except Exception as e:
                logger.error(f"Failed to retrain {model['symbol']}: {str(e)}")
                
        self.send_notification(
            f"Emergency retraining completed for {len(worst_models)} models",
            level='warning'
        )
        
    def identify_pairs_needing_training(self):
        """تحديد الأزواج التي تحتاج تدريب"""
        pairs_to_train = []
        
        # الحصول على جميع الأزواج النشطة
        active_instruments = self.instrument_manager.get_all_instruments()
        
        for instrument in active_instruments:
            for timeframe in ['M5', 'M15', 'H1', 'H4']:
                if self.needs_training(instrument['symbol'], timeframe):
                    pairs_to_train.append({
                        'symbol': instrument['symbol'],
                        'timeframe': timeframe,
                        'type': instrument['type'],
                        'priority': self.get_pair_priority(instrument['symbol'])
                    })
                    
        return pairs_to_train
        
    def needs_training(self, symbol, timeframe):
        """فحص ما إذا كان الزوج يحتاج تدريب"""
        model_path = Path(f"models/unified_sltp/{symbol}_{timeframe}_unified.pkl")
        
        # إذا لم يوجد نموذج
        if not model_path.exists():
            return True
            
        # فحص عمر النموذج
        model_age = time.time() - model_path.stat().st_mtime
        max_age = 7 * 24 * 3600  # 7 أيام
        
        if model_age > max_age:
            return True
            
        # فحص الأداء
        performance = self.performance_tracker.get_pair_performance(symbol, timeframe)
        if performance and performance.get('win_rate', 0) < 0.5:
            return True
            
        return False
        
    def get_pair_priority(self, symbol):
        """الحصول على أولوية الزوج"""
        if symbol in self.config['training_priorities']['high_priority']:
            return 1
        elif symbol in self.config['training_priorities']['medium_priority']:
            return 2
        else:
            return 3
            
    def prioritize_pairs(self, pairs):
        """ترتيب الأزواج حسب الأولوية"""
        return sorted(pairs, key=lambda x: (x['priority'], x['symbol']))
        
    def train_pairs_batch(self, pairs):
        """تدريب مجموعة من الأزواج"""
        results = []
        
        # تدريب بالدفعات
        batch_size = self.config['resource_management']['max_concurrent_training']
        
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            
            for pair in batch:
                try:
                    result = self.training_system.train_single_model(
                        pair['symbol'],
                        pair['timeframe'],
                        pair['type']
                    )
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error training {pair['symbol']} {pair['timeframe']}: {str(e)}")
                    results.append({
                        'pair': pair['symbol'],
                        'timeframe': pair['timeframe'],
                        'success': False,
                        'error': str(e)
                    })
                    
        return results
        
    def cleanup_old_models(self):
        """تنظيف النماذج القديمة"""
        logger.info("🧹 Cleaning up old models...")
        
        models_dir = Path("models/unified_sltp")
        backup_dir = Path("models/backup")
        backup_dir.mkdir(exist_ok=True)
        
        # الاحتفاظ بآخر 3 نسخ من كل نموذج
        model_versions = {}
        
        for model_file in models_dir.glob("*.pkl"):
            base_name = '_'.join(model_file.stem.split('_')[:2])
            
            if base_name not in model_versions:
                model_versions[base_name] = []
                
            model_versions[base_name].append(model_file)
            
        # ترتيب ونقل النماذج القديمة
        cleaned = 0
        for base_name, files in model_versions.items():
            if len(files) > 3:
                # ترتيب حسب التاريخ
                files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                # نقل القديمة للنسخ الاحتياطي
                for old_file in files[3:]:
                    old_file.rename(backup_dir / old_file.name)
                    cleaned += 1
                    
        logger.info(f"✅ Moved {cleaned} old models to backup")
        
    def generate_daily_report(self):
        """توليد تقرير يومي"""
        logger.info("📊 Generating daily report...")
        
        report = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'models_count': 0,
            'average_performance': {},
            'training_sessions': len(self.training_history),
            'top_performers': [],
            'worst_performers': [],
            'sltp_optimization_stats': {}
        }
        
        # جمع إحصائيات النماذج
        models_dir = Path("models/unified_sltp")
        performances = []
        
        for model_file in models_dir.glob("*.pkl"):
            try:
                parts = model_file.stem.split('_')
                if len(parts) >= 2:
                    symbol = parts[0]
                    timeframe = parts[1]
                    
                    perf = self.performance_tracker.get_pair_performance(symbol, timeframe)
                    if perf:
                        performances.append({
                            'symbol': symbol,
                            'timeframe': timeframe,
                            'win_rate': perf.get('win_rate', 0),
                            'sharpe': perf.get('sharpe_ratio', 0),
                            'avg_sl': perf.get('avg_sl_pips', 0),
                            'avg_tp': perf.get('avg_tp_pips', 0),
                            'avg_rr': perf.get('avg_risk_reward', 0)
                        })
                        
            except:
                pass
                
        if performances:
            # حساب المتوسطات
            report['models_count'] = len(performances)
            report['average_performance'] = {
                'win_rate': sum(p['win_rate'] for p in performances) / len(performances),
                'sharpe_ratio': sum(p['sharpe'] for p in performances) / len(performances),
                'avg_sl_pips': sum(p['avg_sl'] for p in performances) / len(performances),
                'avg_tp_pips': sum(p['avg_tp'] for p in performances) / len(performances),
                'avg_risk_reward': sum(p['avg_rr'] for p in performances) / len(performances)
            }
            
            # أفضل وأسوأ النماذج
            sorted_perfs = sorted(performances, key=lambda x: x['win_rate'], reverse=True)
            report['top_performers'] = sorted_perfs[:5]
            report['worst_performers'] = sorted_perfs[-5:]
            
        # حفظ التقرير
        reports_dir = Path("reports")
        reports_dir.mkdir(exist_ok=True)
        
        report_path = reports_dir / f"daily_report_{report['date']}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        # إرسال ملخص
        summary = f"""
📊 Daily Report - {report['date']}
Models: {report['models_count']}
Avg Win Rate: {report['average_performance'].get('win_rate', 0):.1%}
Avg Sharpe: {report['average_performance'].get('sharpe_ratio', 0):.2f}
Avg SL: {report['average_performance'].get('avg_sl_pips', 0):.1f} pips
Avg TP: {report['average_performance'].get('avg_tp_pips', 0):.1f} pips
Avg R:R: {report['average_performance'].get('avg_risk_reward', 0):.2f}
"""
        
        self.send_notification(summary)
        logger.info(f"✅ Report saved to {report_path}")
        
    def log_training_session(self, results, session_type):
        """تسجيل جلسة التدريب"""
        session = {
            'timestamp': datetime.now().isoformat(),
            'type': session_type,
            'results': results,
            'summary': {
                'total': len(results),
                'successful': len([r for r in results if r.get('success', False)]),
                'failed': len([r for r in results if not r.get('success', False)])
            }
        }
        
        self.training_history.append(session)
        
        # الاحتفاظ بآخر 100 جلسة
        if len(self.training_history) > 100:
            self.training_history = self.training_history[-100:]
            
    def send_notification(self, message, level='info'):
        """إرسال إشعار"""
        if not self.config['notification_settings']['enable_notifications']:
            return
            
        # طباعة في السجل
        if level == 'error':
            logger.error(f"🚨 {message}")
        elif level == 'warning':
            logger.warning(f"⚠️ {message}")
        else:
            logger.info(f"📢 {message}")
            
        # إرسال webhook إذا كان متاحاً
        if self.config['notification_settings']['webhook_url']:
            try:
                requests.post(
                    self.config['notification_settings']['webhook_url'],
                    json={
                        'text': message,
                        'level': level,
                        'timestamp': datetime.now().isoformat()
                    },
                    timeout=5
                )
            except:
                pass
                
    def stop(self):
        """إيقاف النظام"""
        logger.info("Stopping automated training system...")
        self.is_running = False
        
    def get_status(self):
        """الحصول على حالة النظام"""
        return {
            'is_running': self.is_running,
            'next_daily_training': schedule.jobs[0].next_run if schedule.jobs else None,
            'training_history_count': len(self.training_history),
            'last_training': self.training_history[-1] if self.training_history else None
        }


if __name__ == "__main__":
    # إنشاء وتشغيل النظام
    auto_trainer = AutomatedTrainingSLTP()
    
    # بدء النظام
    auto_trainer.start()
    
    # البقاء نشطاً
    logger.info("✅ Automated training system is running... Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(60)
            
            # عرض الحالة كل 30 دقيقة
            if datetime.now().minute % 30 == 0:
                status = auto_trainer.get_status()
                logger.info(f"Status: Running={status['is_running']}, "
                          f"History={status['training_history_count']} sessions")
                
    except KeyboardInterrupt:
        logger.info("\n👋 Stopping automated training...")
        auto_trainer.stop()