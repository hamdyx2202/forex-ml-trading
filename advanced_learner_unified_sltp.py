#!/usr/bin/env python3
"""
Advanced Learner with SL/TP Training Support
متعلم متقدم مع دعم تدريب وقف الخسارة والأهداف
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
import joblib
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3
from loguru import logger
import json
import warnings
warnings.filterwarnings('ignore')

# إضافة المسار
import sys
sys.path.append(str(Path(__file__).parent))

from feature_engineer_adaptive_75 import AdaptiveFeatureEngineer75
from unified_standards import STANDARD_FEATURES, get_model_filename

class AdvancedLearnerWithSLTP:
    """متعلم متقدم مع دعم كامل لتدريب SL/TP"""
    
    def __init__(self):
        self.feature_engineer = AdaptiveFeatureEngineer75(target_features=75)
        self.models_dir = Path("models/unified_sltp")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = "trading_data.db"
        
        # نماذج للإشارات (تصنيف)
        self.signal_models = {
            'rf': RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42),
            'xgb': XGBClassifier(n_estimators=200, max_depth=10, random_state=42)
        }
        
        # نماذج لـ SL/TP (انحدار)
        self.sl_models = {
            'rf': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'xgb': XGBRegressor(n_estimators=200, max_depth=10, random_state=42)
        }
        
        self.tp_models = {
            'rf': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42),
            'xgb': XGBRegressor(n_estimators=200, max_depth=10, random_state=42)
        }
        
    def train_model_with_sltp(self, pair, timeframe):
        """تدريب النموذج مع تعلم SL/TP الأمثل"""
        logger.info(f"🎯 Training {pair} {timeframe} with SL/TP optimization...")
        
        try:
            # جمع البيانات
            df = self.load_data_from_db(pair, timeframe)
            if df is None or len(df) < 1000:
                logger.warning(f"Not enough data for {pair} {timeframe}")
                return False
            
            # إضافة الميزات
            df_features = self.feature_engineer.engineer_features(df, pair)
            
            # حساب الأهداف المثلى من البيانات التاريخية
            df_with_targets = self.calculate_optimal_targets(df_features, pair)
            
            # إعداد البيانات للتدريب
            X, y_signal, y_sl, y_tp = self.prepare_training_data(df_with_targets)
            
            if X is None:
                return False
            
            # تقسيم البيانات
            X_train, X_test, y_signal_train, y_signal_test, y_sl_train, y_sl_test, y_tp_train, y_tp_test = \
                train_test_split(X, y_signal, y_sl, y_tp, test_size=0.2, random_state=42)
            
            # تطبيع البيانات
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # تدريب نماذج الإشارات
            signal_results = {}
            for name, model in self.signal_models.items():
                model.fit(X_train_scaled, y_signal_train)
                accuracy = model.score(X_test_scaled, y_signal_test)
                signal_results[name] = accuracy
                logger.info(f"   {name} Signal Accuracy: {accuracy:.4f}")
            
            # تدريب نماذج SL
            sl_results = {}
            for name, model in self.sl_models.items():
                # تدريب فقط على الصفقات الرابحة والخاسرة (ليس NO_TRADE)
                mask = y_signal_train != 2  # 2 = NO_TRADE
                if mask.sum() > 100:
                    model.fit(X_train_scaled[mask], y_sl_train[mask])
                    mae = np.mean(np.abs(model.predict(X_test_scaled[y_signal_test != 2]) - 
                                       y_sl_test[y_signal_test != 2]))
                    sl_results[name] = mae
                    logger.info(f"   {name} SL MAE: {mae:.2f} pips")
            
            # تدريب نماذج TP
            tp_results = {}
            for name, model in self.tp_models.items():
                mask = y_signal_train != 2
                if mask.sum() > 100:
                    model.fit(X_train_scaled[mask], y_tp_train[mask])
                    mae = np.mean(np.abs(model.predict(X_test_scaled[y_signal_test != 2]) - 
                                       y_tp_test[y_signal_test != 2]))
                    tp_results[name] = mae
                    logger.info(f"   {name} TP MAE: {mae:.2f} pips")
            
            # اختيار أفضل النماذج
            best_signal_model = max(signal_results, key=signal_results.get)
            best_sl_model = min(sl_results, key=sl_results.get) if sl_results else 'rf'
            best_tp_model = min(tp_results, key=tp_results.get) if tp_results else 'rf'
            
            # حفظ النموذج المدمج
            model_data = {
                'signal_model': self.signal_models[best_signal_model],
                'sl_model': self.sl_models[best_sl_model],
                'tp_model': self.tp_models[best_tp_model],
                'scaler': scaler,
                'feature_names': list(X.columns),
                'n_features': STANDARD_FEATURES,
                'pair': pair,
                'timeframe': timeframe,
                'metrics': {
                    'signal_accuracy': signal_results[best_signal_model],
                    'sl_mae': sl_results.get(best_sl_model, 0),
                    'tp_mae': tp_results.get(best_tp_model, 0),
                    'signal_results': signal_results,
                    'sl_results': sl_results,
                    'tp_results': tp_results
                },
                'training_date': datetime.now().isoformat(),
                'version': '2.0_with_sltp'
            }
            
            # حفظ النموذج
            filename = get_model_filename(pair, timeframe)
            filepath = self.models_dir / filename
            joblib.dump(model_data, filepath)
            
            logger.info(f"✅ Model saved: {filename}")
            logger.info(f"   Best models: Signal={best_signal_model}, SL={best_sl_model}, TP={best_tp_model}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training {pair} {timeframe}: {str(e)}")
            return False
    
    def calculate_optimal_targets(self, df, pair):
        """حساب SL/TP الأمثل من البيانات التاريخية"""
        logger.info("🎯 Calculating optimal SL/TP targets...")
        
        df = df.copy()
        
        # إضافة أعمدة للأهداف
        df['optimal_sl_pips'] = 0.0
        df['optimal_tp_pips'] = 0.0
        df['actual_outcome'] = 'NO_TRADE'
        
        # حساب قيمة النقطة حسب الزوج
        pip_value = self._get_pip_value(pair)
        
        # تحليل كل نقطة في التاريخ
        lookforward = 100  # عدد الشموع للنظر للأمام
        
        for i in range(len(df) - lookforward):
            current_price = df['close'].iloc[i]
            future_prices = df[['high', 'low', 'close']].iloc[i+1:i+lookforward+1]
            
            # تحديد الاتجاه بناءً على المؤشرات
            signal = self._determine_signal_from_indicators(df.iloc[i])
            
            if signal == 'BUY':
                # حساب أقصى ربح ممكن
                max_profit = future_prices['high'].max() - current_price
                max_profit_index = future_prices['high'].idxmax()
                
                # حساب أقصى خسارة قبل الوصول لأقصى ربح
                prices_before_peak = future_prices.loc[:max_profit_index]
                max_drawdown = current_price - prices_before_peak['low'].min() if len(prices_before_peak) > 0 else 0
                
                # حساب SL/TP الأمثل
                optimal_sl = max(max_drawdown * 1.2, 10 * pip_value)  # على الأقل 10 نقاط
                optimal_tp = max_profit * 0.7  # 70% من أقصى ربح ممكن
                
                # تحديد النتيجة الفعلية
                if max_profit > optimal_sl * 2:  # ربح جيد
                    df.loc[i, 'actual_outcome'] = 'WIN'
                elif max_drawdown > optimal_sl:  # خسارة
                    df.loc[i, 'actual_outcome'] = 'LOSS'
                else:
                    df.loc[i, 'actual_outcome'] = 'BREAKEVEN'
                
                df.loc[i, 'optimal_sl_pips'] = optimal_sl / pip_value
                df.loc[i, 'optimal_tp_pips'] = optimal_tp / pip_value
                
            elif signal == 'SELL':
                # حساب أقصى ربح ممكن
                max_profit = current_price - future_prices['low'].min()
                max_profit_index = future_prices['low'].idxmin()
                
                # حساب أقصى خسارة قبل الوصول لأقصى ربح
                prices_before_bottom = future_prices.loc[:max_profit_index]
                max_drawdown = prices_before_bottom['high'].max() - current_price if len(prices_before_bottom) > 0 else 0
                
                # حساب SL/TP الأمثل
                optimal_sl = max(max_drawdown * 1.2, 10 * pip_value)
                optimal_tp = max_profit * 0.7
                
                # تحديد النتيجة الفعلية
                if max_profit > optimal_sl * 2:
                    df.loc[i, 'actual_outcome'] = 'WIN'
                elif max_drawdown > optimal_sl:
                    df.loc[i, 'actual_outcome'] = 'LOSS'
                else:
                    df.loc[i, 'actual_outcome'] = 'BREAKEVEN'
                
                df.loc[i, 'optimal_sl_pips'] = optimal_sl / pip_value
                df.loc[i, 'optimal_tp_pips'] = optimal_tp / pip_value
            
            else:  # NO_TRADE
                df.loc[i, 'optimal_sl_pips'] = 30  # قيم افتراضية
                df.loc[i, 'optimal_tp_pips'] = 60
        
        # إحصائيات
        trades = df[df['actual_outcome'] != 'NO_TRADE']
        if len(trades) > 0:
            win_rate = (trades['actual_outcome'] == 'WIN').sum() / len(trades)
            avg_sl = trades['optimal_sl_pips'].mean()
            avg_tp = trades['optimal_tp_pips'].mean()
            avg_rr = avg_tp / avg_sl if avg_sl > 0 else 2.0
            
            logger.info(f"   Trades analyzed: {len(trades)}")
            logger.info(f"   Win rate: {win_rate:.2%}")
            logger.info(f"   Avg SL: {avg_sl:.1f} pips")
            logger.info(f"   Avg TP: {avg_tp:.1f} pips")
            logger.info(f"   Avg R:R: {avg_rr:.2f}")
        
        return df
    
    def _determine_signal_from_indicators(self, row):
        """تحديد الإشارة من المؤشرات الفنية"""
        bullish_count = 0
        bearish_count = 0
        
        # RSI
        if 'RSI' in row:
            if row['RSI'] < 30:
                bullish_count += 2
            elif row['RSI'] > 70:
                bearish_count += 2
        
        # MACD
        if 'MACD' in row and 'MACD_signal' in row:
            if row['MACD'] > row['MACD_signal']:
                bullish_count += 1
            else:
                bearish_count += 1
        
        # Moving Averages
        if 'SMA_50' in row and 'SMA_200' in row:
            if row['close'] > row['SMA_50'] > row['SMA_200']:
                bullish_count += 2
            elif row['close'] < row['SMA_50'] < row['SMA_200']:
                bearish_count += 2
        
        # Stochastic
        if 'slowk' in row:
            if row['slowk'] < 20:
                bullish_count += 1
            elif row['slowk'] > 80:
                bearish_count += 1
        
        # Support/Resistance features
        if 'distance_to_support_pct' in row and 'distance_to_resistance_pct' in row:
            if row['distance_to_support_pct'] < 1.0:  # قريب من الدعم
                bullish_count += 2
            elif row['distance_to_resistance_pct'] < 1.0:  # قريب من المقاومة
                bearish_count += 2
        
        # تحديد الإشارة
        if bullish_count >= 4 and bullish_count > bearish_count * 1.5:
            return 'BUY'
        elif bearish_count >= 4 and bearish_count > bullish_count * 1.5:
            return 'SELL'
        else:
            return 'NO_TRADE'
    
    def prepare_training_data(self, df):
        """إعداد البيانات للتدريب"""
        # إزالة الصفوف بدون أهداف
        df = df[df['optimal_sl_pips'] > 0].copy()
        
        # الميزات
        feature_cols = [col for col in df.columns if col not in [
            'open', 'high', 'low', 'close', 'volume', 'time',
            'optimal_sl_pips', 'optimal_tp_pips', 'actual_outcome'
        ]]
        
        X = df[feature_cols]
        
        # تحويل الإشارات لأرقام
        signal_map = {'BUY': 0, 'SELL': 1, 'NO_TRADE': 2}
        y_signal = df['actual_outcome'].map(
            lambda x: signal_map.get('BUY' if x == 'WIN' and df['close'].diff().iloc[-1] > 0 
                                   else 'SELL' if x == 'WIN' and df['close'].diff().iloc[-1] < 0 
                                   else 'NO_TRADE', 2)
        )
        
        # الأهداف
        y_sl = df['optimal_sl_pips']
        y_tp = df['optimal_tp_pips']
        
        # إزالة القيم الناقصة
        mask = ~(X.isna().any(axis=1) | y_signal.isna() | y_sl.isna() | y_tp.isna())
        
        return X[mask], y_signal[mask], y_sl[mask], y_tp[mask]
    
    def load_data_from_db(self, pair, timeframe):
        """تحميل البيانات من قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # جدول البيانات التاريخية
            query = f"""
            SELECT * FROM ohlcv_data 
            WHERE symbol = ? AND timeframe = ?
            ORDER BY timestamp DESC
            LIMIT 10000
            """
            
            df = pd.read_sql_query(query, conn, params=(pair, timeframe))
            conn.close()
            
            if len(df) > 0:
                # ترتيب البيانات
                df = df.sort_values('timestamp').reset_index(drop=True)
                return df
            
        except Exception as e:
            logger.error(f"Error loading data from DB: {str(e)}")
        
        return None
    
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
    
    def predict_with_sltp(self, features, pair, timeframe):
        """التنبؤ بالإشارة مع SL/TP"""
        try:
            # تحميل النموذج
            filename = get_model_filename(pair, timeframe)
            filepath = self.models_dir / filename
            
            if not filepath.exists():
                logger.warning(f"Model not found: {filename}")
                return None
            
            model_data = joblib.load(filepath)
            
            # تطبيع البيانات
            X_scaled = model_data['scaler'].transform(features)
            
            # التنبؤ بالإشارة
            signal_pred = model_data['signal_model'].predict(X_scaled)[0]
            signal_proba = model_data['signal_model'].predict_proba(X_scaled)[0]
            
            # التنبؤ بـ SL/TP
            sl_pred = model_data['sl_model'].predict(X_scaled)[0]
            tp_pred = model_data['tp_model'].predict(X_scaled)[0]
            
            # تحويل الإشارة
            signal_map = {0: 'BUY', 1: 'SELL', 2: 'NO_TRADE'}
            signal = signal_map.get(signal_pred, 'NO_TRADE')
            
            # الثقة
            confidence = np.max(signal_proba)
            
            # تعديل SL/TP بناءً على الثقة
            if confidence < 0.6:
                sl_pred *= 0.8  # تقليل المخاطرة
                tp_pred *= 0.8
            elif confidence > 0.8:
                tp_pred *= 1.2  # زيادة الهدف
            
            return {
                'signal': signal,
                'confidence': confidence,
                'sl_pips': round(sl_pred, 1),
                'tp_pips': round(tp_pred, 1),
                'risk_reward': round(tp_pred / sl_pred, 2) if sl_pred > 0 else 2.0,
                'method': 'ml_optimized'
            }
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return None


if __name__ == "__main__":
    # اختبار النظام
    learner = AdvancedLearnerWithSLTP()
    
    # تدريب نموذج واحد كمثال
    learner.train_model_with_sltp("EURUSD", "H1")