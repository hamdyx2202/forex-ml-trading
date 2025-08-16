#!/usr/bin/env python3
"""
Advanced Continuous Learning System V2
نظام التعلم المستمر المتقدم - النسخة المحدثة
يشمل جميع التحسينات الأخيرة: SL/TP ديناميكي، 5 استراتيجيات، 200+ مؤشر
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from loguru import logger
import json
import warnings
import threading
import time
import talib
warnings.filterwarnings('ignore')

# إضافة المسار
import sys
sys.path.append(str(Path(__file__).parent))

from train_advanced_complete import AdvancedCompleteTrainer

class AdvancedContinuousLearnerV2:
    """نظام التعلم المستمر المتقدم مع جميع الميزات الحديثة"""
    
    def __init__(self):
        # استخدام نفس إعدادات النظام المتقدم
        self.advanced_trainer = AdvancedCompleteTrainer()
        self.models_dir = Path("models/continuous_advanced_v2")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = "data/forex_ml.db"
        
        # الاستراتيجيات المتعددة
        self.strategies = self.advanced_trainer.training_strategies
        self.sl_tp_settings = self.advanced_trainer.sl_tp_settings
        
        # نماذج الإشارات (5 نماذج ensemble)
        self.signal_models = {}
        self.sl_models = {}
        self.tp_models = {}
        
        # تتبع الأداء المتقدم
        self.performance_history = {}
        self.update_intervals = {
            'ultra_short': 1800,    # 30 دقيقة
            'scalping': 3600,       # ساعة
            'short_term': 7200,     # ساعتين
            'medium_term': 14400,   # 4 ساعات
            'long_term': 86400      # يوم
        }
        
        # معايير التحديث الذكية
        self.performance_thresholds = {
            'accuracy_drop': 0.05,      # انخفاض 5% في الدقة
            'consecutive_losses': 5,     # 5 خسائر متتالية
            'profit_factor_min': 1.2,    # عامل الربح الأدنى
            'win_rate_min': 0.55,        # معدل الفوز الأدنى
            'max_drawdown': 0.15         # أقصى تراجع 15%
        }
        
        # حدود البيانات الجديدة
        self.min_new_data = {
            'ultra_short': 500,
            'scalping': 1000,
            'short_term': 2000,
            'medium_term': 3000,
            'long_term': 5000
        }
        
        # تهيئة النماذج
        self._initialize_ensemble_models()
        
    def _initialize_ensemble_models(self):
        """تهيئة نماذج Ensemble المتقدمة"""
        for strategy in self.strategies.keys():
            # نماذج الإشارات
            self.signal_models[strategy] = {
                'lightgbm': lgb.LGBMClassifier(
                    objective='multiclass',
                    num_class=3,
                    n_estimators=1000,
                    max_depth=10,
                    learning_rate=0.01,
                    feature_fraction=0.9,
                    bagging_fraction=0.9,
                    verbosity=-1
                ),
                'xgboost': xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=3,
                    n_estimators=1000,
                    max_depth=10,
                    learning_rate=0.01,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    use_label_encoder=False,
                    eval_metric='mlogloss'
                ),
                'catboost': CatBoostClassifier(
                    loss_function='MultiClass',
                    classes_count=3,
                    iterations=1000,
                    depth=10,
                    learning_rate=0.01,
                    verbose=False
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=500,
                    max_depth=15,
                    min_samples_split=10,
                    class_weight='balanced',
                    random_state=42
                ),
                'neural_network': MLPClassifier(
                    hidden_layer_sizes=(200, 100, 50),
                    activation='relu',
                    learning_rate='adaptive',
                    max_iter=1000,
                    early_stopping=True,
                    random_state=42
                )
            }
            
            # نماذج Stop Loss
            self.sl_models[strategy] = {
                'lightgbm': lgb.LGBMRegressor(
                    n_estimators=1000,
                    max_depth=10,
                    learning_rate=0.01,
                    verbosity=-1
                ),
                'xgboost': xgb.XGBRegressor(
                    n_estimators=1000,
                    max_depth=10,
                    learning_rate=0.01,
                    eval_metric='rmse'
                ),
                'catboost': CatBoostRegressor(
                    iterations=1000,
                    depth=10,
                    learning_rate=0.01,
                    verbose=False
                ),
                'random_forest': RandomForestRegressor(
                    n_estimators=500,
                    max_depth=15,
                    random_state=42
                ),
                'neural_network': MLPRegressor(
                    hidden_layer_sizes=(200, 100, 50),
                    activation='relu',
                    max_iter=1000,
                    early_stopping=True,
                    random_state=42
                )
            }
            
            # نماذج Take Profit (3 مستويات)
            self.tp_models[strategy] = {
                'tp1': {model_name: self._create_tp_model(model_name) 
                       for model_name in ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'neural_network']},
                'tp2': {model_name: self._create_tp_model(model_name) 
                       for model_name in ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'neural_network']},
                'tp3': {model_name: self._create_tp_model(model_name) 
                       for model_name in ['lightgbm', 'xgboost', 'catboost', 'random_forest', 'neural_network']}
            }
    
    def _create_tp_model(self, model_type):
        """إنشاء نموذج Take Profit"""
        if model_type == 'lightgbm':
            return lgb.LGBMRegressor(n_estimators=1000, max_depth=10, learning_rate=0.01, verbosity=-1)
        elif model_type == 'xgboost':
            return xgb.XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.01, eval_metric='rmse')
        elif model_type == 'catboost':
            return CatBoostRegressor(iterations=1000, depth=10, learning_rate=0.01, verbose=False)
        elif model_type == 'random_forest':
            return RandomForestRegressor(n_estimators=500, max_depth=15, random_state=42)
        else:  # neural_network
            return MLPRegressor(hidden_layer_sizes=(200, 100, 50), activation='relu', 
                              max_iter=1000, early_stopping=True, random_state=42)
    
    def start_continuous_learning(self, pairs, timeframes):
        """بدء التعلم المستمر المتقدم"""
        logger.info("🚀 Starting Advanced Continuous Learning V2...")
        logger.info(f"📊 Monitoring {len(pairs)} pairs × {len(timeframes)} timeframes")
        logger.info(f"🎯 Strategies: {list(self.strategies.keys())}")
        
        # تدريب أولي شامل
        logger.info("📚 Initial training phase...")
        for pair in pairs:
            for timeframe in timeframes:
                for strategy in self.strategies.keys():
                    self.train_strategy_models(pair, timeframe, strategy, force_train=True)
        
        # بدء خيوط التحديث المستمر لكل استراتيجية
        for strategy in self.strategies.keys():
            update_thread = threading.Thread(
                target=self._strategy_update_loop,
                args=(pairs, timeframes, strategy),
                daemon=True,
                name=f"ContinuousLearning_{strategy}"
            )
            update_thread.start()
            logger.info(f"✅ Started continuous learning thread for {strategy}")
        
        # خيط لمراقبة الأداء العام
        monitor_thread = threading.Thread(
            target=self._performance_monitor_loop,
            args=(pairs, timeframes),
            daemon=True,
            name="PerformanceMonitor"
        )
        monitor_thread.start()
        
        logger.info("✅ Advanced Continuous Learning System V2 is running!")
    
    def _strategy_update_loop(self, pairs, timeframes, strategy):
        """حلقة التحديث المستمر لاستراتيجية معينة"""
        update_interval = self.update_intervals[strategy]
        
        while True:
            try:
                time.sleep(update_interval)
                
                logger.info(f"🔄 Checking updates for {strategy} strategy...")
                
                for pair in pairs:
                    for timeframe in timeframes:
                        # فحص الأداء الحالي
                        performance = self.evaluate_strategy_performance(pair, timeframe, strategy)
                        
                        # تحديد ما إذا كان يجب التحديث
                        if self.should_update_strategy(pair, timeframe, strategy, performance):
                            logger.info(f"📈 Updating {strategy} models for {pair} {timeframe}")
                            self.train_strategy_models(pair, timeframe, strategy)
                        
            except Exception as e:
                logger.error(f"❌ Error in {strategy} update loop: {e}")
                time.sleep(60)  # انتظار دقيقة قبل المحاولة مرة أخرى
    
    def _performance_monitor_loop(self, pairs, timeframes):
        """مراقبة الأداء العام وإنشاء التقارير"""
        while True:
            try:
                time.sleep(3600)  # كل ساعة
                
                # إنشاء تقرير أداء شامل
                report = self.generate_performance_report(pairs, timeframes)
                
                # حفظ التقرير
                report_path = self.models_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(report_path, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
                
                # طباعة ملخص
                self._print_performance_summary(report)
                
            except Exception as e:
                logger.error(f"❌ Error in performance monitor: {e}")
    
    def train_strategy_models(self, pair, timeframe, strategy, force_train=False):
        """تدريب جميع نماذج استراتيجية معينة"""
        try:
            # تحميل البيانات
            df = self.advanced_trainer.load_data_advanced(pair, timeframe)
            if df is None or len(df) < self.min_new_data[strategy]:
                logger.warning(f"⚠️ Insufficient data for {pair} {timeframe} {strategy}")
                return
            
            # إنشاء الميزات المتقدمة
            features_df = self.advanced_trainer.create_ultra_advanced_features(df, pair)
            
            # إنشاء الأهداف مع SL/TP
            targets, confidences, sl_tp_info, quality = self.advanced_trainer.create_advanced_targets_with_sl_tp(
                df, strategy
            )
            
            # فلترة الصفقات عالية الجودة
            high_quality_mask = quality > 0.7
            
            if high_quality_mask.sum() < 100:
                logger.warning(f"⚠️ Not enough high quality trades for {pair} {timeframe} {strategy}")
                return
            
            # إعداد البيانات
            X = features_df.values[high_quality_mask]
            y_signal = targets[high_quality_mask]
            confidence = confidences[high_quality_mask]
            
            # إعداد بيانات SL/TP
            sl_values = []
            tp1_values = []
            tp2_values = []
            tp3_values = []
            
            for i, info in enumerate(sl_tp_info):
                if high_quality_mask[i] and info is not None:
                    sl_values.append(info['stop_loss'])
                    tp1_values.append(info['take_profits'][0] if len(info['take_profits']) > 0 else 0)
                    tp2_values.append(info['take_profits'][1] if len(info['take_profits']) > 1 else 0)
                    tp3_values.append(info['take_profits'][2] if len(info['take_profits']) > 2 else 0)
            
            # تقسيم البيانات
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y_signal[:split_idx], y_signal[split_idx:]
            sl_train, sl_test = sl_values[:split_idx], sl_values[split_idx:]
            tp1_train, tp1_test = tp1_values[:split_idx], tp1_values[split_idx:]
            tp2_train, tp2_test = tp2_values[:split_idx], tp2_values[split_idx:]
            tp3_train, tp3_test = tp3_values[:split_idx], tp3_values[split_idx:]
            
            # معايرة البيانات
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # تدريب نماذج الإشارات
            logger.info(f"🎯 Training signal models for {strategy}...")
            signal_scores = {}
            
            for model_name, model in self.signal_models[strategy].items():
                model.fit(X_train_scaled, y_train)
                score = model.score(X_test_scaled, y_test)
                signal_scores[model_name] = score
                logger.info(f"  • {model_name}: {score:.4f}")
            
            # تدريب نماذج Stop Loss
            logger.info(f"🛑 Training Stop Loss models for {strategy}...")
            sl_scores = {}
            
            for model_name, model in self.sl_models[strategy].items():
                model.fit(X_train_scaled, sl_train)
                sl_pred = model.predict(X_test_scaled)
                sl_mae = np.mean(np.abs(sl_pred - sl_test))
                sl_scores[model_name] = sl_mae
                logger.info(f"  • {model_name} MAE: {sl_mae:.4f}")
            
            # تدريب نماذج Take Profit
            logger.info(f"🎯 Training Take Profit models for {strategy}...")
            tp_scores = {'tp1': {}, 'tp2': {}, 'tp3': {}}
            
            for tp_level, tp_train, tp_test in [('tp1', tp1_train, tp1_test), 
                                               ('tp2', tp2_train, tp2_test), 
                                               ('tp3', tp3_train, tp3_test)]:
                for model_name, model in self.tp_models[strategy][tp_level].items():
                    model.fit(X_train_scaled, tp_train)
                    tp_pred = model.predict(X_test_scaled)
                    tp_mae = np.mean(np.abs(tp_pred - tp_test))
                    tp_scores[tp_level][model_name] = tp_mae
                    logger.info(f"  • {model_name} {tp_level} MAE: {tp_mae:.4f}")
            
            # حفظ النماذج والأداء
            self._save_strategy_models(pair, timeframe, strategy, {
                'signal_models': self.signal_models[strategy],
                'sl_models': self.sl_models[strategy],
                'tp_models': self.tp_models[strategy],
                'scaler': scaler,
                'feature_names': features_df.columns.tolist(),
                'performance': {
                    'signal_scores': signal_scores,
                    'sl_scores': sl_scores,
                    'tp_scores': tp_scores,
                    'training_date': datetime.now(),
                    'data_size': len(X),
                    'quality_trades': high_quality_mask.sum()
                }
            })
            
            # تحديث سجل الأداء
            self._update_performance_history(pair, timeframe, strategy, signal_scores, sl_scores, tp_scores)
            
            logger.info(f"✅ Successfully trained {strategy} models for {pair} {timeframe}")
            
        except Exception as e:
            logger.error(f"❌ Error training {strategy} for {pair} {timeframe}: {e}")
    
    def should_update_strategy(self, pair, timeframe, strategy, current_performance):
        """تحديد ما إذا كان يجب تحديث النماذج"""
        # الحصول على الأداء السابق
        key = f"{pair}_{timeframe}_{strategy}"
        
        if key not in self.performance_history:
            return True  # لا يوجد سجل سابق
        
        history = self.performance_history[key]
        
        # معايير التحديث
        reasons = []
        
        # 1. انخفاض الدقة
        if current_performance['accuracy'] < history['best_accuracy'] - self.performance_thresholds['accuracy_drop']:
            reasons.append(f"Accuracy dropped from {history['best_accuracy']:.4f} to {current_performance['accuracy']:.4f}")
        
        # 2. خسائر متتالية
        if current_performance.get('consecutive_losses', 0) >= self.performance_thresholds['consecutive_losses']:
            reasons.append(f"Consecutive losses: {current_performance['consecutive_losses']}")
        
        # 3. عامل الربح منخفض
        if current_performance.get('profit_factor', 1.0) < self.performance_thresholds['profit_factor_min']:
            reasons.append(f"Low profit factor: {current_performance['profit_factor']:.2f}")
        
        # 4. معدل فوز منخفض
        if current_performance.get('win_rate', 0.5) < self.performance_thresholds['win_rate_min']:
            reasons.append(f"Low win rate: {current_performance['win_rate']:.2%}")
        
        # 5. تراجع كبير
        if current_performance.get('drawdown', 0) > self.performance_thresholds['max_drawdown']:
            reasons.append(f"High drawdown: {current_performance['drawdown']:.2%}")
        
        # 6. بيانات جديدة كافية
        new_data_count = self._get_new_data_count(pair, timeframe, history['last_update'])
        if new_data_count >= self.min_new_data[strategy]:
            reasons.append(f"New data available: {new_data_count} records")
        
        if reasons:
            logger.info(f"📊 Update needed for {strategy} {pair} {timeframe}:")
            for reason in reasons:
                logger.info(f"   • {reason}")
            return True
        
        return False
    
    def evaluate_strategy_performance(self, pair, timeframe, strategy):
        """تقييم أداء استراتيجية معينة"""
        try:
            # تحميل البيانات الحديثة
            conn = sqlite3.connect(self.db_path)
            
            # الحصول على الصفقات الأخيرة
            query = """
                SELECT * FROM trade_results 
                WHERE symbol = ? AND timeframe = ? AND strategy = ?
                AND timestamp > datetime('now', '-7 days')
                ORDER BY timestamp DESC
            """
            
            trades_df = pd.read_sql_query(query, conn, params=(pair, timeframe, strategy))
            conn.close()
            
            if len(trades_df) == 0:
                return {
                    'accuracy': 0.5,
                    'consecutive_losses': 0,
                    'profit_factor': 1.0,
                    'win_rate': 0.5,
                    'drawdown': 0
                }
            
            # حساب المقاييس
            wins = len(trades_df[trades_df['profit'] > 0])
            losses = len(trades_df[trades_df['profit'] <= 0])
            
            # معدل الفوز
            win_rate = wins / len(trades_df) if len(trades_df) > 0 else 0.5
            
            # عامل الربح
            total_wins = trades_df[trades_df['profit'] > 0]['profit'].sum()
            total_losses = abs(trades_df[trades_df['profit'] < 0]['profit'].sum())
            profit_factor = total_wins / total_losses if total_losses > 0 else total_wins
            
            # خسائر متتالية
            consecutive_losses = 0
            current_streak = 0
            for profit in trades_df['profit'].values:
                if profit <= 0:
                    current_streak += 1
                    consecutive_losses = max(consecutive_losses, current_streak)
                else:
                    current_streak = 0
            
            # التراجع
            cumulative_profit = trades_df['profit'].cumsum()
            running_max = cumulative_profit.cummax()
            drawdown = (running_max - cumulative_profit).max()
            max_profit = cumulative_profit.max()
            drawdown_pct = drawdown / max_profit if max_profit > 0 else 0
            
            # الدقة (نسبة التنبؤات الصحيحة)
            if 'predicted_signal' in trades_df.columns and 'actual_signal' in trades_df.columns:
                accuracy = (trades_df['predicted_signal'] == trades_df['actual_signal']).mean()
            else:
                accuracy = win_rate  # استخدام معدل الفوز كبديل
            
            return {
                'accuracy': accuracy,
                'consecutive_losses': consecutive_losses,
                'profit_factor': profit_factor,
                'win_rate': win_rate,
                'drawdown': drawdown_pct,
                'total_trades': len(trades_df),
                'total_profit': trades_df['profit'].sum()
            }
            
        except Exception as e:
            logger.error(f"❌ Error evaluating performance: {e}")
            return {
                'accuracy': 0.5,
                'consecutive_losses': 0,
                'profit_factor': 1.0,
                'win_rate': 0.5,
                'drawdown': 0
            }
    
    def _get_new_data_count(self, pair, timeframe, last_update):
        """الحصول على عدد البيانات الجديدة منذ آخر تحديث"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT COUNT(*) FROM price_data
                WHERE symbol = ? AND timeframe = ?
                AND time > ?
            """
            
            last_update_timestamp = int(last_update.timestamp()) if isinstance(last_update, datetime) else last_update
            
            cursor = conn.cursor()
            cursor.execute(query, (pair, timeframe, last_update_timestamp))
            count = cursor.fetchone()[0]
            conn.close()
            
            return count
            
        except Exception as e:
            logger.error(f"❌ Error getting new data count: {e}")
            return 0
    
    def _save_strategy_models(self, pair, timeframe, strategy, models_data):
        """حفظ نماذج الاستراتيجية"""
        save_dir = self.models_dir / f"{pair}_{timeframe}_{strategy}"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # حفظ كل مكون
        joblib.dump(models_data, save_dir / f"complete_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        
        # حفظ نسخة "latest" للوصول السريع
        joblib.dump(models_data, save_dir / "latest_models.pkl")
        
        logger.info(f"💾 Saved models to {save_dir}")
    
    def _update_performance_history(self, pair, timeframe, strategy, signal_scores, sl_scores, tp_scores):
        """تحديث سجل الأداء"""
        key = f"{pair}_{timeframe}_{strategy}"
        
        # حساب أفضل دقة من نماذج الإشارات
        best_accuracy = max(signal_scores.values())
        
        self.performance_history[key] = {
            'best_accuracy': best_accuracy,
            'last_update': datetime.now(),
            'signal_scores': signal_scores,
            'sl_scores': sl_scores,
            'tp_scores': tp_scores,
            'update_count': self.performance_history.get(key, {}).get('update_count', 0) + 1
        }
    
    def generate_performance_report(self, pairs, timeframes):
        """إنشاء تقرير أداء شامل"""
        report = {
            'timestamp': datetime.now(),
            'pairs': pairs,
            'timeframes': timeframes,
            'strategies': list(self.strategies.keys()),
            'overall_performance': {},
            'strategy_performance': {},
            'pair_performance': {},
            'recommendations': []
        }
        
        # أداء كل استراتيجية
        for strategy in self.strategies.keys():
            strategy_stats = {
                'total_accuracy': [],
                'total_profit': 0,
                'total_trades': 0,
                'best_pair': None,
                'worst_pair': None
            }
            
            for pair in pairs:
                for timeframe in timeframes:
                    performance = self.evaluate_strategy_performance(pair, timeframe, strategy)
                    
                    strategy_stats['total_accuracy'].append(performance['accuracy'])
                    strategy_stats['total_profit'] += performance.get('total_profit', 0)
                    strategy_stats['total_trades'] += performance.get('total_trades', 0)
                    
                    # تتبع أفضل وأسوأ الأزواج
                    if not strategy_stats['best_pair'] or performance['accuracy'] > strategy_stats['best_pair'][1]:
                        strategy_stats['best_pair'] = (f"{pair}_{timeframe}", performance['accuracy'])
                    
                    if not strategy_stats['worst_pair'] or performance['accuracy'] < strategy_stats['worst_pair'][1]:
                        strategy_stats['worst_pair'] = (f"{pair}_{timeframe}", performance['accuracy'])
            
            # حساب المتوسطات
            avg_accuracy = np.mean(strategy_stats['total_accuracy']) if strategy_stats['total_accuracy'] else 0
            
            report['strategy_performance'][strategy] = {
                'average_accuracy': avg_accuracy,
                'total_profit': strategy_stats['total_profit'],
                'total_trades': strategy_stats['total_trades'],
                'best_pair': strategy_stats['best_pair'],
                'worst_pair': strategy_stats['worst_pair']
            }
        
        # توصيات
        for strategy, stats in report['strategy_performance'].items():
            if stats['average_accuracy'] < 0.6:
                report['recommendations'].append(
                    f"Consider retraining {strategy} strategy (accuracy: {stats['average_accuracy']:.2%})"
                )
            
            if stats['worst_pair'] and stats['worst_pair'][1] < 0.5:
                report['recommendations'].append(
                    f"Remove or retrain {strategy} for {stats['worst_pair'][0]} (accuracy: {stats['worst_pair'][1]:.2%})"
                )
        
        return report
    
    def _print_performance_summary(self, report):
        """طباعة ملخص الأداء"""
        logger.info("\n" + "="*80)
        logger.info("📊 PERFORMANCE SUMMARY")
        logger.info("="*80)
        
        # أداء كل استراتيجية
        logger.info("\n🎯 Strategy Performance:")
        for strategy, stats in report['strategy_performance'].items():
            logger.info(f"\n{strategy.upper()}:")
            logger.info(f"  • Average Accuracy: {stats['average_accuracy']:.2%}")
            logger.info(f"  • Total Profit: ${stats['total_profit']:.2f}")
            logger.info(f"  • Total Trades: {stats['total_trades']}")
            if stats['best_pair']:
                logger.info(f"  • Best Pair: {stats['best_pair'][0]} ({stats['best_pair'][1]:.2%})")
            if stats['worst_pair']:
                logger.info(f"  • Worst Pair: {stats['worst_pair'][0]} ({stats['worst_pair'][1]:.2%})")
        
        # توصيات
        if report['recommendations']:
            logger.info("\n💡 Recommendations:")
            for rec in report['recommendations']:
                logger.info(f"  • {rec}")
        
        logger.info("\n" + "="*80)
    
    def predict_with_sl_tp(self, pair, timeframe, features):
        """التنبؤ مع SL/TP لجميع الاستراتيجيات"""
        predictions = {}
        
        for strategy in self.strategies.keys():
            try:
                # تحميل النماذج
                model_path = self.models_dir / f"{pair}_{timeframe}_{strategy}" / "latest_models.pkl"
                
                if not model_path.exists():
                    logger.warning(f"⚠️ No models found for {pair} {timeframe} {strategy}")
                    continue
                
                models_data = joblib.load(model_path)
                scaler = models_data['scaler']
                
                # معايرة البيانات
                features_scaled = scaler.transform(features.reshape(1, -1))
                
                # التنبؤ بالإشارة (متوسط جميع النماذج)
                signal_predictions = []
                signal_probabilities = []
                
                for model_name, model in models_data['signal_models'].items():
                    pred = model.predict(features_scaled)[0]
                    signal_predictions.append(pred)
                    
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(features_scaled)[0]
                        signal_probabilities.append(proba)
                
                # حساب الإشارة النهائية (الأغلبية)
                final_signal = int(np.median(signal_predictions))
                
                # حساب الثقة
                if signal_probabilities:
                    avg_proba = np.mean(signal_probabilities, axis=0)
                    confidence = np.max(avg_proba)
                else:
                    # حساب الثقة من التوافق
                    signal_counts = np.bincount(signal_predictions)
                    confidence = signal_counts[final_signal] / len(signal_predictions)
                
                # التنبؤ بـ Stop Loss
                sl_predictions = []
                for model_name, model in models_data['sl_models'].items():
                    sl_pred = model.predict(features_scaled)[0]
                    sl_predictions.append(sl_pred)
                
                final_sl = np.median(sl_predictions)
                
                # التنبؤ بـ Take Profit (3 مستويات)
                tp_predictions = {'tp1': [], 'tp2': [], 'tp3': []}
                
                for tp_level in ['tp1', 'tp2', 'tp3']:
                    for model_name, model in models_data['tp_models'][tp_level].items():
                        tp_pred = model.predict(features_scaled)[0]
                        tp_predictions[tp_level].append(tp_pred)
                
                final_tp1 = np.median(tp_predictions['tp1'])
                final_tp2 = np.median(tp_predictions['tp2'])
                final_tp3 = np.median(tp_predictions['tp3'])
                
                predictions[strategy] = {
                    'signal': final_signal,
                    'confidence': confidence,
                    'stop_loss': final_sl,
                    'take_profit_1': final_tp1,
                    'take_profit_2': final_tp2,
                    'take_profit_3': final_tp3,
                    'strategy_settings': self.strategies[strategy],
                    'sl_tp_settings': self.sl_tp_settings[strategy]
                }
                
            except Exception as e:
                logger.error(f"❌ Error predicting for {strategy}: {e}")
        
        return predictions

def main():
    """مثال على الاستخدام"""
    # إنشاء المتعلم المستمر المتقدم
    learner = AdvancedContinuousLearnerV2()
    
    # الأزواج والفترات الزمنية
    pairs = ['EURUSDm', 'GBPUSDm', 'XAUUSDm']
    timeframes = ['M5', 'M15', 'H1']
    
    # بدء التعلم المستمر
    learner.start_continuous_learning(pairs, timeframes)
    
    # الحفاظ على البرنامج قيد التشغيل
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("🛑 Stopping continuous learning...")

if __name__ == "__main__":
    main()