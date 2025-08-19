#!/usr/bin/env python3
"""
🧹 Model Manager - إدارة وتنظيف النماذج
📊 يحتفظ بأفضل النماذج ويحذف القديمة/الضعيفة
"""

import os
import json
import joblib
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self, models_dir='./trained_models'):
        self.models_dir = models_dir
        self.metadata_file = os.path.join(models_dir, 'models_metadata.json')
        self.load_metadata()
    
    def load_metadata(self):
        """تحميل بيانات النماذج"""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
    
    def save_metadata(self):
        """حفظ بيانات النماذج"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def register_model(self, model_path, accuracy, symbol, timeframe, model_type):
        """تسجيل نموذج جديد"""
        model_key = os.path.basename(model_path)
        
        self.metadata[model_key] = {
            'path': model_path,
            'accuracy': accuracy,
            'symbol': symbol,
            'timeframe': timeframe,
            'model_type': model_type,
            'created_at': datetime.now().isoformat(),
            'last_used': datetime.now().isoformat(),
            'usage_count': 0,
            'performance_history': []
        }
        
        self.save_metadata()
        logger.info(f"✅ Registered model: {model_key} (Accuracy: {accuracy:.2%})")
    
    def update_model_usage(self, model_key, performance=None):
        """تحديث استخدام النموذج"""
        if model_key in self.metadata:
            self.metadata[model_key]['last_used'] = datetime.now().isoformat()
            self.metadata[model_key]['usage_count'] += 1
            
            if performance is not None:
                self.metadata[model_key]['performance_history'].append({
                    'date': datetime.now().isoformat(),
                    'performance': performance
                })
                
                # احتفظ بآخر 100 أداء فقط
                if len(self.metadata[model_key]['performance_history']) > 100:
                    self.metadata[model_key]['performance_history'] = \
                        self.metadata[model_key]['performance_history'][-100:]
            
            self.save_metadata()
    
    def cleanup_old_models(self, keep_days=30, min_accuracy=0.60, keep_best_n=3):
        """
        تنظيف النماذج القديمة والضعيفة
        
        المعايير:
        1. احذف النماذج الأقدم من keep_days يوم (إلا إذا كانت الأفضل)
        2. احذف النماذج بدقة أقل من min_accuracy
        3. احتفظ بأفضل keep_best_n نموذج لكل زوج/فريم
        """
        logger.info("🧹 Starting model cleanup...")
        
        current_time = datetime.now()
        models_to_delete = []
        
        # تجميع النماذج حسب symbol_timeframe
        models_by_pair = {}
        
        for model_key, info in self.metadata.items():
            pair_key = f"{info['symbol']}_{info['timeframe']}"
            
            if pair_key not in models_by_pair:
                models_by_pair[pair_key] = []
            
            models_by_pair[pair_key].append((model_key, info))
        
        # تحليل كل مجموعة
        for pair_key, models in models_by_pair.items():
            logger.info(f"\n📊 Analyzing {pair_key} ({len(models)} models)")
            
            # ترتيب حسب الدقة والاستخدام
            models.sort(key=lambda x: (
                x[1]['accuracy'],
                x[1]['usage_count'],
                -self._get_age_days(x[1]['created_at'])
            ), reverse=True)
            
            # الاحتفاظ بأفضل N
            for i, (model_key, info) in enumerate(models):
                age_days = self._get_age_days(info['created_at'])
                last_used_days = self._get_age_days(info['last_used'])
                
                # معايير الحذف
                should_delete = False
                reason = ""
                
                # 1. ليس من أفضل N والدقة منخفضة
                if i >= keep_best_n and info['accuracy'] < min_accuracy:
                    should_delete = True
                    reason = f"Low accuracy ({info['accuracy']:.2%}) and not in top {keep_best_n}"
                
                # 2. قديم جداً وغير مستخدم
                elif age_days > keep_days and last_used_days > keep_days and i >= keep_best_n:
                    should_delete = True
                    reason = f"Old ({age_days} days) and unused for {last_used_days} days"
                
                # 3. أداء سيء مؤخراً
                elif self._get_recent_performance(info) < 0.50 and i >= keep_best_n:
                    should_delete = True
                    reason = f"Poor recent performance ({self._get_recent_performance(info):.2%})"
                
                if should_delete:
                    models_to_delete.append((model_key, info['path'], reason))
                else:
                    logger.info(f"   ✅ Keep: {info['model_type']} - Acc: {info['accuracy']:.2%}, "
                              f"Used: {info['usage_count']} times")
        
        # حذف النماذج المحددة
        if models_to_delete:
            logger.info(f"\n🗑️ Deleting {len(models_to_delete)} models...")
            
            for model_key, model_path, reason in models_to_delete:
                try:
                    # حذف الملف
                    if os.path.exists(model_path):
                        os.remove(model_path)
                        logger.info(f"   ❌ Deleted: {model_key} - {reason}")
                    
                    # حذف من metadata
                    del self.metadata[model_key]
                    
                except Exception as e:
                    logger.error(f"   ⚠️ Error deleting {model_key}: {e}")
            
            self.save_metadata()
            logger.info(f"✅ Cleanup complete! Deleted {len(models_to_delete)} models")
        else:
            logger.info("✅ No models to delete - all models are performing well!")
        
        return len(models_to_delete)
    
    def _get_age_days(self, date_str):
        """حساب العمر بالأيام"""
        try:
            date = datetime.fromisoformat(date_str)
            return (datetime.now() - date).days
        except:
            return 0
    
    def _get_recent_performance(self, info):
        """حساب الأداء الأخير"""
        if not info.get('performance_history'):
            return info['accuracy']  # استخدم الدقة الأصلية
        
        # متوسط آخر 20 أداء
        recent = info['performance_history'][-20:]
        return sum(p['performance'] for p in recent) / len(recent)
    
    def get_model_stats(self):
        """إحصائيات النماذج"""
        total_models = len(self.metadata)
        
        if total_models == 0:
            return "No models found"
        
        # حساب الإحصائيات
        avg_accuracy = sum(m['accuracy'] for m in self.metadata.values()) / total_models
        total_usage = sum(m['usage_count'] for m in self.metadata.values())
        
        # النماذج حسب النوع
        model_types = {}
        for info in self.metadata.values():
            model_type = info['model_type']
            if model_type not in model_types:
                model_types[model_type] = 0
            model_types[model_type] += 1
        
        stats = f"""
📊 Model Statistics:
━━━━━━━━━━━━━━━━━━━━━
Total Models: {total_models}
Average Accuracy: {avg_accuracy:.2%}
Total Usage: {total_usage:,}

Models by Type:"""
        
        for model_type, count in model_types.items():
            stats += f"\n  • {model_type}: {count}"
        
        return stats
    
    def get_best_models(self, n=10):
        """الحصول على أفضل N نموذج"""
        sorted_models = sorted(
            self.metadata.items(),
            key=lambda x: (x[1]['accuracy'], x[1]['usage_count']),
            reverse=True
        )
        
        return sorted_models[:n]

def main():
    """مثال للاستخدام"""
    manager = ModelManager()
    
    # عرض الإحصائيات
    print(manager.get_model_stats())
    
    # تنظيف النماذج
    deleted = manager.cleanup_old_models(
        keep_days=30,      # احذف الأقدم من 30 يوم
        min_accuracy=0.65, # احذف أقل من 65% دقة
        keep_best_n=3      # احتفظ بأفضل 3 لكل زوج
    )
    
    print(f"\n🧹 Deleted {deleted} models")
    
    # أفضل 5 نماذج
    print("\n🏆 Top 5 Models:")
    for i, (key, info) in enumerate(manager.get_best_models(5), 1):
        print(f"{i}. {info['symbol']} {info['timeframe']} - "
              f"{info['model_type']}: {info['accuracy']:.2%}")

if __name__ == "__main__":
    main()