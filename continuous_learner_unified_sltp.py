#!/usr/bin/env python3
"""
Continuous Learner with SL/TP Support
متعلم مستمر مع دعم تدريب وقف الخسارة والأهداف
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from loguru import logger
import json
import warnings
import threading
import time
warnings.filterwarnings('ignore')

# إضافة المسار
import sys
sys.path.append(str(Path(__file__).parent))

from feature_engineer_adaptive_75 import AdaptiveFeatureEngineer75
from unified_standards import STANDARD_FEATURES, get_model_filename
from performance_tracker import PerformanceTracker

class ContinuousLearnerWithSLTP:
    """متعلم مستمر يحدث النماذج باستمرار مع SL/TP"""
    
    def __init__(self):
        self.feature_engineer = AdaptiveFeatureEngineer75(target_features=75)
        self.models_dir = Path("models/unified_sltp")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = "trading_data.db"
        self.performance_tracker = PerformanceTracker(lookback_days=30)
        
        # نماذج متعددة
        self.signal_models = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
            'xgb': XGBClassifier(n_estimators=200, max_depth=10, random_state=42),
            'mlp': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        self.sl_models = {
            'rf': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'xgb': XGBRegressor(n_estimators=200, max_depth=10, random_state=42),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        self.tp_models = {
            'rf': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'xgb': XGBRegressor(n_estimators=200, max_depth=10, random_state=42),
            'mlp': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
        }
        
        # تتبع الأداء
        self.model_performance = {}
        self.update_interval = 3600  # تحديث كل ساعة
        self.min_new_data = 100  # الحد الأدنى للبيانات الجديدة
        
    def start_continuous_learning(self, pairs, timeframes):
        """بدء التعلم المستمر"""
        logger.info("🚀 Starting continuous learning with SL/TP optimization...")
        
        # تدريب أولي
        for pair in pairs:
            for timeframe in timeframes:
                self.train_or_update_model(pair, timeframe, force_train=True)
        
        # بدء خيط التحديث المستمر
        update_thread = threading.Thread(
            target=self._continuous_update_loop,
            args=(pairs, timeframes),
            daemon=True
        )
        update_thread.start()
        
        logger.info("✅ Continuous learning started!")
        
    def _continuous_update_loop(self, pairs, timeframes):
        """حلقة التحديث المستمر"""
        while True:
            try:
                time.sleep(self.update_interval)
                
                for pair in pairs:
                    for timeframe in timeframes:
                        # فحص الأداء الحالي
                        current_performance = self.check_model_performance(pair, timeframe)
                        
                        # تحديث إذا انخفض الأداء أو توفرت بيانات جديدة
                        if self.should_update_model(pair, timeframe, current_performance):
                            self.train_or_update_model(pair, timeframe)
                            
            except Exception as e:
                logger.error(f"Error in continuous update loop: {str(e)}")
                time.sleep(60)  # انتظار دقيقة قبل المحاولة مرة أخرى
                
    def should_update_model(self, pair, timeframe, current_performance):
        """تحديد ما إذا كان النموذج يحتاج تحديث"""
        # تحديث إذا انخفض الأداء
        if current_performance and current_performance.get('accuracy', 0) < 0.55:
            logger.info(f"📉 Performance drop detected for {pair} {timeframe}")
            return True
            
        # تحديث إذا توفرت بيانات جديدة كافية
        new_data_count = self.count_new_data(pair, timeframe)
        if new_data_count >= self.min_new_data:
            logger.info(f"📊 New data available for {pair} {timeframe}: {new_data_count} bars")
            return True
            
        return False
        
    def train_or_update_model(self, pair, timeframe, force_train=False):
        """تدريب أو تحديث النموذج مع SL/TP"""
        logger.info(f"🔄 Updating model for {pair} {timeframe}...")
        
        try:
            # جمع البيانات
            df = self.load_recent_data(pair, timeframe)
            if df is None or len(df) < 500:
                logger.warning(f"Not enough data for {pair} {timeframe}")
                return False
                
            # إضافة الميزات
            df_features = self.feature_engineer.engineer_features(df, pair)
            
            # حساب الأهداف المثلى مع التحليل الديناميكي
            df_with_targets = self.calculate_dynamic_targets(df_features, pair)
            
            # إعداد البيانات
            X, y_signal, y_sl, y_tp = self.prepare_training_data(df_with_targets)
            
            if X is None:
                return False
                
            # التحقق من وجود نموذج سابق
            existing_model = self.load_existing_model(pair, timeframe)
            
            if existing_model and not force_train:
                # تحديث تدريجي
                updated_model = self.incremental_update(
                    existing_model, X, y_signal, y_sl, y_tp
                )
                self.save_model(updated_model, pair, timeframe)
            else:
                # تدريب كامل
                new_model = self.full_training(X, y_signal, y_sl, y_tp, pair, timeframe)
                self.save_model(new_model, pair, timeframe)
                
            # تحديث تتبع الأداء
            self.update_performance_tracking(pair, timeframe)
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating {pair} {timeframe}: {str(e)}")
            return False
            
    def calculate_dynamic_targets(self, df, pair):
        """حساب SL/TP الديناميكي بناءً على ظروف السوق"""
        logger.info("🎯 Calculating dynamic SL/TP targets...")
        
        df = df.copy()
        pip_value = self._get_pip_value(pair)
        
        # إضافة أعمدة
        df['optimal_sl_pips'] = 0.0
        df['optimal_tp_pips'] = 0.0
        df['market_condition'] = 'normal'
        df['risk_level'] = 'medium'
        
        # حساب مؤشرات السوق
        df['volatility'] = df['ATR'] / df['close'] * 100  # نسبة التقلب
        df['trend_strength'] = abs(df['SMA_50'] - df['SMA_200']) / df['close'] * 100
        
        lookforward = 100
        
        for i in range(len(df) - lookforward):
            current_price = df['close'].iloc[i]
            future_prices = df[['high', 'low', 'close']].iloc[i+1:i+lookforward+1]
            
            # تحديد حالة السوق
            volatility = df['volatility'].iloc[i]
            trend = df['trend_strength'].iloc[i]
            
            if volatility > 2.0:
                df.loc[i, 'market_condition'] = 'volatile'
                df.loc[i, 'risk_level'] = 'high'
            elif volatility < 0.5:
                df.loc[i, 'market_condition'] = 'quiet'
                df.loc[i, 'risk_level'] = 'low'
            elif trend > 1.5:
                df.loc[i, 'market_condition'] = 'trending'
                df.loc[i, 'risk_level'] = 'medium'
            
            # تحديد الإشارة
            signal = self._determine_signal_advanced(df.iloc[i])
            
            if signal == 'BUY':
                # تحليل الحركة المستقبلية
                max_profit = future_prices['high'].max() - current_price
                max_profit_index = future_prices['high'].idxmax()
                
                prices_before_peak = future_prices.loc[:max_profit_index]
                max_drawdown = current_price - prices_before_peak['low'].min() if len(prices_before_peak) > 0 else 0
                
                # حساب SL/TP بناءً على حالة السوق
                if df.loc[i, 'market_condition'] == 'volatile':
                    # في السوق المتقلب: SL أوسع، TP أكبر
                    optimal_sl = max(max_drawdown * 1.5, 20 * pip_value)
                    optimal_tp = max_profit * 0.6  # أهداف أقل طموحاً
                elif df.loc[i, 'market_condition'] == 'trending':
                    # في الترند: SL محكم، TP كبير
                    optimal_sl = max(max_drawdown * 1.0, 15 * pip_value)
                    optimal_tp = max_profit * 0.8  # أهداف طموحة
                else:
                    # السوق العادي
                    optimal_sl = max(max_drawdown * 1.2, 10 * pip_value)
                    optimal_tp = max_profit * 0.7
                
                df.loc[i, 'optimal_sl_pips'] = optimal_sl / pip_value
                df.loc[i, 'optimal_tp_pips'] = optimal_tp / pip_value
                
            elif signal == 'SELL':
                # نفس المنطق للبيع
                max_profit = current_price - future_prices['low'].min()
                max_profit_index = future_prices['low'].idxmin()
                
                prices_before_bottom = future_prices.loc[:max_profit_index]
                max_drawdown = prices_before_bottom['high'].max() - current_price if len(prices_before_bottom) > 0 else 0
                
                if df.loc[i, 'market_condition'] == 'volatile':
                    optimal_sl = max(max_drawdown * 1.5, 20 * pip_value)
                    optimal_tp = max_profit * 0.6
                elif df.loc[i, 'market_condition'] == 'trending':
                    optimal_sl = max(max_drawdown * 1.0, 15 * pip_value)
                    optimal_tp = max_profit * 0.8
                else:
                    optimal_sl = max(max_drawdown * 1.2, 10 * pip_value)
                    optimal_tp = max_profit * 0.7
                
                df.loc[i, 'optimal_sl_pips'] = optimal_sl / pip_value
                df.loc[i, 'optimal_tp_pips'] = optimal_tp / pip_value
            
            else:  # NO_TRADE
                # قيم افتراضية بناءً على التقلب
                df.loc[i, 'optimal_sl_pips'] = 20 + (volatility * 10)
                df.loc[i, 'optimal_tp_pips'] = 40 + (volatility * 20)
        
        # إحصائيات
        logger.info(f"   Market conditions: {df['market_condition'].value_counts().to_dict()}")
        logger.info(f"   Avg SL: {df['optimal_sl_pips'].mean():.1f} pips")
        logger.info(f"   Avg TP: {df['optimal_tp_pips'].mean():.1f} pips")
        
        return df
        
    def _determine_signal_advanced(self, row):
        """تحديد الإشارة بطريقة متقدمة"""
        score = 0
        
        # RSI
        if 'RSI' in row:
            if row['RSI'] < 25:
                score += 3
            elif row['RSI'] < 35:
                score += 2
            elif row['RSI'] > 75:
                score -= 3
            elif row['RSI'] > 65:
                score -= 2
                
        # MACD
        if 'MACD' in row and 'MACD_signal' in row:
            macd_diff = row['MACD'] - row['MACD_signal']
            if macd_diff > 0:
                score += 2 if abs(macd_diff) > row['ATR'] * 0.1 else 1
            else:
                score -= 2 if abs(macd_diff) > row['ATR'] * 0.1 else -1
                
        # Moving Averages
        if 'SMA_50' in row and 'SMA_200' in row:
            if row['close'] > row['SMA_50'] > row['SMA_200']:
                score += 3
            elif row['close'] < row['SMA_50'] < row['SMA_200']:
                score -= 3
                
        # Bollinger Bands
        if 'BB_upper' in row and 'BB_lower' in row:
            bb_position = (row['close'] - row['BB_lower']) / (row['BB_upper'] - row['BB_lower'])
            if bb_position < 0.2:
                score += 2
            elif bb_position > 0.8:
                score -= 2
                
        # Support/Resistance
        if 'distance_to_support_pct' in row and 'distance_to_resistance_pct' in row:
            if row['distance_to_support_pct'] < 0.5:
                score += 3
            elif row['distance_to_resistance_pct'] < 0.5:
                score -= 3
                
        # Volume
        if 'volume_ratio' in row and row['volume_ratio'] > 1.5:
            score = int(score * 1.2)  # تعزيز الإشارة مع الحجم
            
        # القرار النهائي
        if score >= 5:
            return 'BUY'
        elif score <= -5:
            return 'SELL'
        else:
            return 'NO_TRADE'
            
    def incremental_update(self, existing_model, X, y_signal, y_sl, y_tp):
        """تحديث تدريجي للنموذج"""
        logger.info("📈 Performing incremental update...")
        
        # تقسيم البيانات
        X_train, X_test, y_signal_train, y_signal_test, y_sl_train, y_sl_test, y_tp_train, y_tp_test = \
            train_test_split(X, y_signal, y_sl, y_tp, test_size=0.2, random_state=42)
            
        # تطبيع
        X_train_scaled = existing_model['scaler'].transform(X_train)
        X_test_scaled = existing_model['scaler'].transform(X_test)
        
        # تحديث نماذج الإشارات
        signal_model = existing_model['signal_model']
        
        # دمج التنبؤات القديمة مع البيانات الجديدة (ensemble approach)
        old_predictions = signal_model.predict_proba(X_train_scaled)
        
        # إعادة تدريب مع وزن أكبر للبيانات الجديدة
        sample_weights = np.ones(len(X_train))
        sample_weights[-int(len(X_train)*0.3):] = 2.0  # وزن مضاعف لآخر 30%
        
        signal_model.fit(X_train_scaled, y_signal_train, sample_weight=sample_weights)
        
        # تحديث نماذج SL/TP
        mask = y_signal_train != 2
        if mask.sum() > 50:
            existing_model['sl_model'].fit(
                X_train_scaled[mask], 
                y_sl_train[mask],
                sample_weight=sample_weights[mask]
            )
            existing_model['tp_model'].fit(
                X_train_scaled[mask], 
                y_tp_train[mask],
                sample_weight=sample_weights[mask]
            )
            
        # تقييم الأداء
        new_accuracy = signal_model.score(X_test_scaled, y_signal_test)
        logger.info(f"   Updated accuracy: {new_accuracy:.4f}")
        
        # تحديث البيانات الوصفية
        existing_model['last_update'] = datetime.now().isoformat()
        existing_model['incremental_updates'] = existing_model.get('incremental_updates', 0) + 1
        existing_model['latest_accuracy'] = new_accuracy
        
        return existing_model
        
    def full_training(self, X, y_signal, y_sl, y_tp, pair, timeframe):
        """تدريب كامل للنموذج"""
        logger.info("🏋️ Performing full training...")
        
        # تقسيم البيانات
        X_train, X_test, y_signal_train, y_signal_test, y_sl_train, y_sl_test, y_tp_train, y_tp_test = \
            train_test_split(X, y_signal, y_sl, y_tp, test_size=0.2, random_state=42)
            
        # تطبيع
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # تدريب جميع النماذج وتقييمها
        signal_results = {}
        sl_results = {}
        tp_results = {}
        
        # تدريب نماذج الإشارات
        for name, model in self.signal_models.items():
            model.fit(X_train_scaled, y_signal_train)
            accuracy = model.score(X_test_scaled, y_signal_test)
            signal_results[name] = accuracy
            logger.info(f"   {name} Signal Accuracy: {accuracy:.4f}")
            
        # تدريب نماذج SL/TP
        mask = y_signal_train != 2
        if mask.sum() > 100:
            for name, model in self.sl_models.items():
                model.fit(X_train_scaled[mask], y_sl_train[mask])
                mae = np.mean(np.abs(model.predict(X_test_scaled[y_signal_test != 2]) - 
                                   y_sl_test[y_signal_test != 2]))
                sl_results[name] = mae
                logger.info(f"   {name} SL MAE: {mae:.2f} pips")
                
            for name, model in self.tp_models.items():
                model.fit(X_train_scaled[mask], y_tp_train[mask])
                mae = np.mean(np.abs(model.predict(X_test_scaled[y_signal_test != 2]) - 
                                   y_tp_test[y_signal_test != 2]))
                tp_results[name] = mae
                logger.info(f"   {name} TP MAE: {mae:.2f} pips")
                
        # اختيار أفضل النماذج
        best_signal = max(signal_results, key=signal_results.get)
        best_sl = min(sl_results, key=sl_results.get) if sl_results else 'rf'
        best_tp = min(tp_results, key=tp_results.get) if tp_results else 'rf'
        
        # إنشاء النموذج النهائي
        model_data = {
            'signal_model': self.signal_models[best_signal],
            'sl_model': self.sl_models[best_sl],
            'tp_model': self.tp_models[best_tp],
            'scaler': scaler,
            'feature_names': list(X.columns),
            'n_features': STANDARD_FEATURES,
            'pair': pair,
            'timeframe': timeframe,
            'metrics': {
                'signal_accuracy': signal_results[best_signal],
                'sl_mae': sl_results.get(best_sl, 0),
                'tp_mae': tp_results.get(best_tp, 0),
                'all_results': {
                    'signal': signal_results,
                    'sl': sl_results,
                    'tp': tp_results
                }
            },
            'training_date': datetime.now().isoformat(),
            'version': '3.0_continuous_sltp',
            'best_models': {
                'signal': best_signal,
                'sl': best_sl,
                'tp': best_tp
            }
        }
        
        return model_data
        
    def check_model_performance(self, pair, timeframe):
        """فحص أداء النموذج الحالي"""
        try:
            # تحميل سجل الأداء
            performance = self.performance_tracker.get_pair_performance(pair, timeframe)
            
            if performance:
                return {
                    'accuracy': performance.get('win_rate', 0),
                    'avg_profit': performance.get('avg_profit', 0),
                    'sharpe_ratio': performance.get('sharpe_ratio', 0),
                    'max_drawdown': performance.get('max_drawdown', 0)
                }
                
        except Exception as e:
            logger.error(f"Error checking performance: {str(e)}")
            
        return None
        
    def load_recent_data(self, pair, timeframe, days=30):
        """تحميل البيانات الحديثة"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # تحميل آخر X يوم
            query = f"""
            SELECT * FROM ohlcv_data 
            WHERE symbol = ? AND timeframe = ?
            AND timestamp > datetime('now', '-{days} days')
            ORDER BY timestamp DESC
            """
            
            df = pd.read_sql_query(query, conn, params=(pair, timeframe))
            conn.close()
            
            if len(df) > 0:
                df = df.sort_values('timestamp').reset_index(drop=True)
                return df
                
        except Exception as e:
            logger.error(f"Error loading recent data: {str(e)}")
            
        return None
        
    def count_new_data(self, pair, timeframe):
        """عد البيانات الجديدة منذ آخر تدريب"""
        try:
            # تحميل معلومات آخر تدريب
            filename = get_model_filename(pair, timeframe)
            filepath = self.models_dir / filename
            
            if filepath.exists():
                model_data = joblib.load(filepath)
                last_training = datetime.fromisoformat(model_data['training_date'])
                
                # عد البيانات الجديدة
                conn = sqlite3.connect(self.db_path)
                query = """
                SELECT COUNT(*) FROM ohlcv_data 
                WHERE symbol = ? AND timeframe = ?
                AND timestamp > ?
                """
                
                cursor = conn.cursor()
                cursor.execute(query, (pair, timeframe, last_training))
                count = cursor.fetchone()[0]
                conn.close()
                
                return count
                
        except Exception as e:
            logger.error(f"Error counting new data: {str(e)}")
            
        return 0
        
    def prepare_training_data(self, df):
        """إعداد البيانات للتدريب"""
        df = df[df['optimal_sl_pips'] > 0].copy()
        
        feature_cols = [col for col in df.columns if col not in [
            'open', 'high', 'low', 'close', 'volume', 'time', 'timestamp',
            'optimal_sl_pips', 'optimal_tp_pips', 'market_condition', 
            'risk_level', 'volatility', 'trend_strength'
        ]]
        
        X = df[feature_cols]
        
        # تحويل الإشارات
        signal_map = {'BUY': 0, 'SELL': 1, 'NO_TRADE': 2}
        y_signal = df.apply(lambda row: signal_map.get(
            self._determine_signal_advanced(row), 2
        ), axis=1)
        
        y_sl = df['optimal_sl_pips']
        y_tp = df['optimal_tp_pips']
        
        # إزالة القيم الناقصة
        mask = ~(X.isna().any(axis=1) | y_signal.isna() | y_sl.isna() | y_tp.isna())
        
        return X[mask], y_signal[mask], y_sl[mask], y_tp[mask]
        
    def save_model(self, model_data, pair, timeframe):
        """حفظ النموذج"""
        filename = get_model_filename(pair, timeframe)
        filepath = self.models_dir / filename
        joblib.dump(model_data, filepath)
        logger.info(f"✅ Model saved: {filename}")
        
    def load_existing_model(self, pair, timeframe):
        """تحميل نموذج موجود"""
        try:
            filename = get_model_filename(pair, timeframe)
            filepath = self.models_dir / filename
            
            if filepath.exists():
                return joblib.load(filepath)
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            
        return None
        
    def update_performance_tracking(self, pair, timeframe):
        """تحديث تتبع الأداء"""
        # يتم التحديث بواسطة نظام التداول الفعلي
        pass
        
    def _get_pip_value(self, pair):
        """الحصول على قيمة النقطة"""
        if 'JPY' in pair:
            return 0.01
        elif 'XAU' in pair or 'GOLD' in pair:
            return 0.1
        elif 'BTC' in pair or any(idx in pair for idx in ['US30', 'NAS', 'DAX']):
            return 1.0
        else:
            return 0.0001
            
    def get_model_stats(self):
        """الحصول على إحصائيات جميع النماذج"""
        stats = []
        
        for filepath in self.models_dir.glob("*.pkl"):
            try:
                model_data = joblib.load(filepath)
                stats.append({
                    'pair': model_data['pair'],
                    'timeframe': model_data['timeframe'],
                    'accuracy': model_data['metrics']['signal_accuracy'],
                    'sl_mae': model_data['metrics']['sl_mae'],
                    'tp_mae': model_data['metrics']['tp_mae'],
                    'last_update': model_data.get('last_update', model_data['training_date']),
                    'updates': model_data.get('incremental_updates', 0)
                })
            except:
                pass
                
        return pd.DataFrame(stats)


if __name__ == "__main__":
    # اختبار النظام
    learner = ContinuousLearnerWithSLTP()
    
    # أزواج للتدريب المستمر
    pairs = ["EURUSD", "GBPUSD", "XAUUSD"]
    timeframes = ["H1", "H4"]
    
    # بدء التعلم المستمر
    learner.start_continuous_learning(pairs, timeframes)
    
    # الانتظار لعرض الإحصائيات
    time.sleep(10)
    
    # عرض الإحصائيات
    stats = learner.get_model_stats()
    if not stats.empty:
        print("\n📊 Model Statistics:")
        print(stats.to_string())
    
    # البقاء نشطاً
    print("\n✅ Continuous learning is running... Press Ctrl+C to stop")
    try:
        while True:
            time.sleep(60)
            # عرض تحديث دوري
            stats = learner.get_model_stats()
            if not stats.empty:
                print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Active models: {len(stats)}")
    except KeyboardInterrupt:
        print("\n👋 Stopping continuous learning...")