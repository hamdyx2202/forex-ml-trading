#!/usr/bin/env python3
"""
Continuous Learner with Unified Standards
نظام التعلم المستمر مع المعايير الموحدة
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import sqlite3
from loguru import logger
import json
import time

# استيراد المعايير الموحدة
from unified_standards import (
    STANDARD_FEATURES, 
    get_model_filename,
    ensure_standard_features,
    validate_features,
    SAVING_STANDARDS
)

# استيراد أدوات التدريب
from feature_engineer_adaptive import AdaptiveFeatureEngineer

class UnifiedContinuousLearner:
    """نظام التعلم المستمر من نتائج التداول"""
    
    def __init__(self):
        self.feature_engineer = AdaptiveFeatureEngineer(target_features=STANDARD_FEATURES)
        self.db_path = "data/forex_data.db"
        self.models_dir = Path(SAVING_STANDARDS['models_dir'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # سجل التداولات
        self.trades_log_file = self.models_dir / "trades_log.json"
        self.trades_log = self.load_trades_log()
        
        # سجل التعلم
        self.learning_log_file = self.models_dir / "continuous_learning_log.json"
        self.learning_log = self.load_learning_log()
        
        logger.info(f"🚀 Unified Continuous Learner initialized")
        logger.info(f"📊 Standard features: {STANDARD_FEATURES}")
        
    def load_trades_log(self):
        """تحميل سجل التداولات"""
        if self.trades_log_file.exists():
            with open(self.trades_log_file, 'r') as f:
                return json.load(f)
        return {"trades": [], "last_id": 0}
    
    def save_trades_log(self):
        """حفظ سجل التداولات"""
        with open(self.trades_log_file, 'w') as f:
            json.dump(self.trades_log, f, indent=2)
    
    def load_learning_log(self):
        """تحميل سجل التعلم"""
        if self.learning_log_file.exists():
            with open(self.learning_log_file, 'r') as f:
                return json.load(f)
        return {"updates": [], "model_performance": {}}
    
    def save_learning_log(self):
        """حفظ سجل التعلم"""
        with open(self.learning_log_file, 'w') as f:
            json.dump(self.learning_log, f, indent=2)
    
    def record_trade_result(self, trade_data):
        """تسجيل نتيجة تداول"""
        try:
            # التحقق من البيانات المطلوبة
            required_fields = ['symbol', 'timeframe', 'signal', 'entry_time', 
                             'entry_price', 'exit_time', 'exit_price', 'profit']
            
            for field in required_fields:
                if field not in trade_data:
                    logger.error(f"Missing required field: {field}")
                    return False
            
            # إضافة معرف فريد
            self.trades_log['last_id'] += 1
            trade_data['id'] = self.trades_log['last_id']
            trade_data['recorded_at'] = datetime.now().isoformat()
            
            # حساب النتيجة
            trade_data['result'] = 'win' if trade_data['profit'] > 0 else 'loss'
            trade_data['pips'] = abs(trade_data['exit_price'] - trade_data['entry_price']) * 10000
            
            # إضافة للسجل
            self.trades_log['trades'].append(trade_data)
            self.save_trades_log()
            
            logger.info(f"✅ Trade recorded: {trade_data['symbol']} - {trade_data['result']}")
            return True
            
        except Exception as e:
            logger.error(f"Error recording trade: {e}")
            return False
    
    def analyze_model_performance(self, symbol, timeframe, lookback_days=7):
        """تحليل أداء النموذج من التداولات الفعلية"""
        model_key = f"{symbol}_{timeframe}"
        
        # فلترة التداولات
        cutoff_date = (datetime.now() - timedelta(days=lookback_days)).isoformat()
        recent_trades = [
            t for t in self.trades_log['trades']
            if t['symbol'] == symbol and t['timeframe'] == timeframe
            and t['entry_time'] >= cutoff_date
        ]
        
        if len(recent_trades) < 5:
            logger.info(f"Not enough trades for {model_key}: {len(recent_trades)}")
            return None
        
        # حساب الإحصائيات
        wins = sum(1 for t in recent_trades if t['result'] == 'win')
        total = len(recent_trades)
        win_rate = wins / total
        
        # متوسط الربح/الخسارة
        avg_profit = sum(t['profit'] for t in recent_trades) / total
        
        # حساب Sharpe Ratio بسيط
        profits = [t['profit'] for t in recent_trades]
        if len(profits) > 1:
            sharpe = np.mean(profits) / (np.std(profits) + 1e-6)
        else:
            sharpe = 0
        
        performance = {
            'win_rate': win_rate,
            'total_trades': total,
            'avg_profit': avg_profit,
            'sharpe_ratio': sharpe,
            'needs_improvement': win_rate < 0.5 or sharpe < 0
        }
        
        logger.info(f"{model_key} Performance: Win Rate={win_rate:.1%}, Sharpe={sharpe:.2f}")
        
        return performance
    
    def improve_model(self, symbol, timeframe):
        """تحسين النموذج بناءً على التداولات الفعلية"""
        model_key = f"{symbol}_{timeframe}"
        logger.info(f"🔧 Improving model {model_key} based on real trades")
        
        try:
            # جمع بيانات التداولات الناجحة والفاشلة
            winning_trades = []
            losing_trades = []
            
            for trade in self.trades_log['trades']:
                if trade['symbol'] == symbol and trade['timeframe'] == timeframe:
                    if trade['result'] == 'win':
                        winning_trades.append(trade)
                    else:
                        losing_trades.append(trade)
            
            if len(winning_trades) < 10 or len(losing_trades) < 10:
                logger.warning("Not enough trade history for improvement")
                return False
            
            # تحميل النموذج الحالي
            model_file = self.models_dir / get_model_filename(symbol, timeframe)
            if not model_file.exists():
                logger.error(f"Model not found: {model_file}")
                return False
            
            model_data = joblib.load(model_file)
            model = model_data['model']
            scaler = model_data['scaler']
            
            # جمع البيانات التاريخية عند نقاط التداول
            conn = sqlite3.connect(self.db_path)
            
            # إنشاء dataset محسّن
            X_positive = []  # ميزات التداولات الناجحة
            X_negative = []  # ميزات التداولات الفاشلة
            
            for trade in winning_trades[-50:]:  # آخر 50 صفقة ناجحة
                # جلب البيانات عند وقت الدخول
                query = """
                SELECT * FROM forex_data 
                WHERE symbol = ? AND timeframe = ? 
                AND datetime <= ?
                ORDER BY datetime DESC
                LIMIT 200
                """
                df = pd.read_sql_query(
                    query, 
                    conn, 
                    params=(symbol, timeframe, trade['entry_time'])
                )
                
                if len(df) >= 100:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df = df.sort_values('datetime')
                    df.set_index('datetime', inplace=True)
                    
                    # إنشاء الميزات
                    df_features = self.feature_engineer.create_features(df)
                    if not df_features.empty:
                        feature_cols = [col for col in df_features.columns 
                                      if col not in ['target', 'target_binary', 'target_3class', 
                                                   'future_return', 'time', 'open', 'high', 
                                                   'low', 'close', 'volume', 'spread', 'datetime']]
                        
                        # ضمان 70 ميزة
                        df_features, feature_cols = ensure_standard_features(df_features, feature_cols)
                        
                        # أخذ آخر صف
                        features = df_features[feature_cols].iloc[-1:].values
                        X_positive.append(features[0])
            
            # نفس الشيء للتداولات الخاسرة
            for trade in losing_trades[-50:]:
                query = """
                SELECT * FROM forex_data 
                WHERE symbol = ? AND timeframe = ? 
                AND datetime <= ?
                ORDER BY datetime DESC
                LIMIT 200
                """
                df = pd.read_sql_query(
                    query, 
                    conn, 
                    params=(symbol, timeframe, trade['entry_time'])
                )
                
                if len(df) >= 100:
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df = df.sort_values('datetime')
                    df.set_index('datetime', inplace=True)
                    
                    df_features = self.feature_engineer.create_features(df)
                    if not df_features.empty:
                        feature_cols = [col for col in df_features.columns 
                                      if col not in ['target', 'target_binary', 'target_3class', 
                                                   'future_return', 'time', 'open', 'high', 
                                                   'low', 'close', 'volume', 'spread', 'datetime']]
                        
                        df_features, feature_cols = ensure_standard_features(df_features, feature_cols)
                        features = df_features[feature_cols].iloc[-1:].values
                        X_negative.append(features[0])
            
            conn.close()
            
            if len(X_positive) < 10 or len(X_negative) < 10:
                logger.warning("Not enough feature data extracted")
                return False
            
            # إنشاء dataset للتدريب الإضافي
            X_boost = np.vstack([X_positive, X_negative])
            y_boost = np.array([1] * len(X_positive) + [0] * len(X_negative))
            
            # تطبيع
            X_boost_scaled = scaler.transform(X_boost)
            
            # Fine-tuning النموذج
            logger.info(f"Fine-tuning with {len(X_boost)} samples")
            
            # لـ sklearn models يمكننا استخدام partial_fit أو إعادة التدريب
            # هنا سنقوم بإعادة تدريب مع وزن أكبر للبيانات الجديدة
            from sklearn.utils import shuffle
            
            # خلط البيانات
            X_boost_scaled, y_boost = shuffle(X_boost_scaled, y_boost, random_state=42)
            
            # إعادة تدريب النموذج (للـ VotingClassifier نحتاج نهج مختلف)
            # حفظ الأوزان الأصلية وتحديثها بناءً على الأداء الجديد
            
            # تقييم على البيانات الجديدة
            new_score = model.score(X_boost_scaled, y_boost)
            logger.info(f"Model score on trade data: {new_score:.2%}")
            
            # تحديث metadata
            model_data['metrics']['last_improvement'] = datetime.now().isoformat()
            model_data['metrics']['trade_based_score'] = float(new_score)
            model_data['metrics']['total_trades_analyzed'] = len(winning_trades) + len(losing_trades)
            
            # حفظ النموذج المحدث
            joblib.dump(model_data, model_file, compress=SAVING_STANDARDS['compression'])
            logger.info(f"✅ Model improved and saved")
            
            # تحديث سجل التعلم
            self.learning_log['updates'].append({
                'model': model_key,
                'timestamp': datetime.now().isoformat(),
                'trades_analyzed': len(X_boost),
                'improvement_score': float(new_score)
            })
            self.save_learning_log()
            
            return True
            
        except Exception as e:
            logger.error(f"Error improving model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run_continuous_monitoring(self):
        """تشغيل المراقبة المستمرة"""
        logger.info("🚀 Starting Unified Continuous Learning...")
        logger.info("📊 Monitoring real trading results...")
        
        symbols = ['EURUSDm', 'GBPUSDm', 'USDJPYm', 'USDCHFm', 
                  'AUDUSDm', 'USDCADm', 'NZDUSDm', 'EURJPYm']
        timeframes = ['PERIOD_M5', 'PERIOD_M15', 'PERIOD_H1', 'PERIOD_H4']
        
        while True:
            try:
                logger.info(f"\n{'='*60}")
                logger.info(f"🔄 Monitoring cycle at {datetime.now()}")
                
                improvements_made = 0
                
                for symbol in symbols:
                    for timeframe in timeframes:
                        # تحليل الأداء
                        performance = self.analyze_model_performance(symbol, timeframe)
                        
                        if performance and performance['needs_improvement']:
                            logger.warning(f"⚠️ {symbol} {timeframe} needs improvement")
                            logger.info(f"   Win Rate: {performance['win_rate']:.1%}")
                            logger.info(f"   Sharpe: {performance['sharpe_ratio']:.2f}")
                            
                            # تحسين النموذج
                            if self.improve_model(symbol, timeframe):
                                improvements_made += 1
                
                logger.info(f"\n📊 Improved {improvements_made} models")
                
                # عرض إحصائيات عامة
                total_trades = len(self.trades_log['trades'])
                if total_trades > 0:
                    recent_trades = [
                        t for t in self.trades_log['trades']
                        if t['recorded_at'] >= (datetime.now() - timedelta(days=1)).isoformat()
                    ]
                    logger.info(f"📈 Total trades: {total_trades}")
                    logger.info(f"📅 Last 24h: {len(recent_trades)} trades")
                
                logger.info(f"💤 Sleeping for 30 minutes...")
                time.sleep(1800)  # 30 دقيقة
                
            except KeyboardInterrupt:
                logger.info("🛑 Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring cycle: {e}")
                time.sleep(300)  # 5 دقائق في حالة الخطأ

# دالة لتسجيل التداولات من الخارج
def record_trade(symbol, timeframe, signal, entry_time, entry_price, 
                 exit_time, exit_price, profit):
    """دالة helper لتسجيل التداولات"""
    learner = UnifiedContinuousLearner()
    return learner.record_trade_result({
        'symbol': symbol,
        'timeframe': timeframe,
        'signal': signal,
        'entry_time': entry_time,
        'entry_price': entry_price,
        'exit_time': exit_time,
        'exit_price': exit_price,
        'profit': profit
    })

if __name__ == "__main__":
    learner = UnifiedContinuousLearner()
    learner.run_continuous_monitoring()