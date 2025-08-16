#!/usr/bin/env python3
"""
Simple Model Training - نظام تدريب مبسط يعمل مع الملفات المتاحة
"""

import os
import sys
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
import xgboost as xgb

# إعداد التسجيل
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

class SimpleModelTrainer:
    """نظام تدريب مبسط لجميع العملات"""
    
    def __init__(self):
        self.min_data_points = 1000
        self.test_size = 0.2
        self.random_state = 42
        
    def get_all_symbols_from_db(self):
        """الحصول على جميع العملات من قاعدة البيانات"""
        try:
            conn = sqlite3.connect("data/forex_data.db")
            query = """
                SELECT DISTINCT symbol, timeframe, COUNT(*) as count
                FROM price_data
                GROUP BY symbol, timeframe
                HAVING count >= ?
                ORDER BY symbol, timeframe
            """
            df = pd.read_sql_query(query, conn, params=(self.min_data_points,))
            conn.close()
            
            logger.info(f"✅ تم العثور على {len(df)} مجموعة بيانات")
            return df
            
        except Exception as e:
            logger.error(f"❌ خطأ في قراءة البيانات: {e}")
            return pd.DataFrame()
    
    def load_data(self, symbol, timeframe, limit=50000):
        """تحميل البيانات من قاعدة البيانات"""
        try:
            conn = sqlite3.connect("data/forex_data.db")
            query = """
                SELECT * FROM price_data 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY time DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
            conn.close()
            
            if df.empty:
                return None
                
            df = df.sort_values('time').reset_index(drop=True)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            
            logger.info(f"✅ تم تحميل {len(df)} سجل لـ {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل البيانات: {e}")
            return None
    
    def create_features(self, df):
        """إنشاء ميزات بسيطة وفعالة"""
        features = pd.DataFrame()
        
        # ميزات السعر الأساسية
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # المتوسطات المتحركة
        for period in [5, 10, 20, 50, 100]:
            sma = df['close'].rolling(period).mean()
            features[f'sma_{period}'] = df['close'] / sma - 1
            features[f'sma_{period}_slope'] = sma.pct_change(5)
        
        # RSI
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = self.calculate_rsi(df['close'], period)
        
        # ATR
        for period in [7, 14, 21]:
            features[f'atr_{period}'] = self.calculate_atr(df, period)
        
        # Bollinger Bands
        for period in [10, 20]:
            sma = df['close'].rolling(period).mean()
            std = df['close'].rolling(period).std()
            features[f'bb_upper_{period}'] = (df['close'] - (sma + 2*std)) / df['close']
            features[f'bb_lower_{period}'] = ((sma - 2*std) - df['close']) / df['close']
            features[f'bb_width_{period}'] = (4 * std) / sma
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        features['macd'] = macd / df['close']
        features['macd_signal'] = signal / df['close']
        features['macd_hist'] = (macd - signal) / df['close']
        
        # Volume
        features['volume_sma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        features['volume_change'] = df['volume'].pct_change()
        
        # التقلب
        for period in [5, 10, 20]:
            features[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
        
        # ميزات الوقت
        features['hour'] = df['time'].dt.hour
        features['day_of_week'] = df['time'].dt.dayofweek
        features['day_of_month'] = df['time'].dt.day
        features['month'] = df['time'].dt.month
        
        # جلسات التداول
        features['london_session'] = ((features['hour'] >= 8) & (features['hour'] <= 16)).astype(int)
        features['ny_session'] = ((features['hour'] >= 13) & (features['hour'] <= 22)).astype(int)
        features['asian_session'] = ((features['hour'] >= 0) & (features['hour'] <= 8)).astype(int)
        
        # نماذج الشموع البسيطة
        body = abs(df['close'] - df['open'])
        range_hl = df['high'] - df['low']
        features['body_ratio'] = body / (range_hl + 0.0001)
        features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / (range_hl + 0.0001)
        features['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / (range_hl + 0.0001)
        
        # Doji
        features['is_doji'] = (features['body_ratio'] < 0.1).astype(int)
        
        # Pin Bar
        features['is_pin_bar'] = (
            ((features['upper_shadow'] > 2 * features['body_ratio']) | 
             (features['lower_shadow'] > 2 * features['body_ratio']))
        ).astype(int)
        
        return features
    
    def calculate_rsi(self, prices, period=14):
        """حساب RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 0.0001)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, df, period=14):
        """حساب ATR"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        return atr / df['close']
    
    def create_targets(self, df, lookahead_minutes=30, min_change_pct=0.1):
        """إنشاء أهداف التصنيف"""
        # حساب التغير المستقبلي
        future_returns = df['close'].shift(-lookahead_minutes) / df['close'] - 1
        
        # تصنيف ثلاثي
        targets = pd.Series(index=df.index, dtype=int)
        targets[future_returns > min_change_pct/100] = 2  # صعود
        targets[future_returns < -min_change_pct/100] = 0  # هبوط
        targets[(future_returns >= -min_change_pct/100) & (future_returns <= min_change_pct/100)] = 1  # محايد
        
        return targets
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """تدريب نموذج مجمع"""
        # إنشاء النماذج
        models = []
        
        # LightGBM
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'n_estimators': 300
        }
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        models.append(('lightgbm', lgb_model))
        
        # XGBoost
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 3,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'mlogloss',
            'use_label_encoder': False,
            'n_estimators': 300
        }
        xgb_model = xgb.XGBClassifier(**xgb_params)
        models.append(('xgboost', xgb_model))
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        models.append(('random_forest', rf_model))
        
        # نموذج مجمع
        ensemble = VotingClassifier(models, voting='soft', n_jobs=-1)
        
        logger.info("🎯 بدء التدريب...")
        ensemble.fit(X_train, y_train)
        
        # تقييم
        train_score = ensemble.score(X_train, y_train)
        test_score = ensemble.score(X_test, y_test)
        
        y_pred = ensemble.predict(X_test)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        scores = {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        logger.info(f"✅ دقة التدريب: {train_score:.4f}")
        logger.info(f"✅ دقة الاختبار: {test_score:.4f}")
        logger.info(f"📊 Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return ensemble, scores
    
    def train_symbol(self, symbol, timeframe):
        """تدريب نموذج لعملة واحدة"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🚀 تدريب {symbol} {timeframe}")
        logger.info(f"{'='*60}")
        
        # تحميل البيانات
        df = self.load_data(symbol, timeframe)
        if df is None or len(df) < self.min_data_points:
            logger.warning(f"⚠️ بيانات غير كافية")
            return None
        
        # إنشاء الميزات
        features = self.create_features(df)
        
        # إنشاء الأهداف
        targets = self.create_targets(df)
        
        # تنظيف البيانات
        features = features.fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        # إزالة الصفوف الأخيرة بدون أهداف
        valid_idx = ~targets.isna()
        features = features[valid_idx]
        targets = targets[valid_idx]
        
        if len(features) < self.min_data_points:
            logger.warning(f"⚠️ بيانات غير كافية بعد التنظيف")
            return None
        
        # تحضير البيانات
        X = features.values
        y = targets.values
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=False
        )
        
        # معايرة البيانات
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # تدريب النموذج
        model, scores = self.train_model(X_train_scaled, y_train, X_test_scaled, y_test)
        
        # حفظ النموذج
        model_dir = Path(f"models/{symbol}_{timeframe}")
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': list(features.columns),
            'scores': scores,
            'training_date': datetime.now(),
            'symbol': symbol,
            'timeframe': timeframe,
            'samples': len(features)
        }
        
        joblib.dump(model_data, model_dir / 'model_simple.pkl')
        
        # حفظ تقرير
        with open(model_dir / 'training_report_simple.json', 'w') as f:
            report = {
                'symbol': symbol,
                'timeframe': timeframe,
                'training_date': str(datetime.now()),
                'samples': len(features),
                'features': len(features.columns),
                'scores': scores
            }
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 تم حفظ النموذج في {model_dir}")
        
        return scores
    
    def train_all(self):
        """تدريب جميع العملات المتاحة"""
        logger.info("\n" + "="*80)
        logger.info("🚀 بدء تدريب جميع العملات")
        logger.info("="*80)
        
        # الحصول على جميع العملات
        available_data = self.get_all_symbols_from_db()
        
        if available_data.empty:
            logger.error("❌ لا توجد بيانات متاحة")
            return
        
        successful = []
        failed = []
        
        # تدريب كل عملة
        for idx, row in available_data.iterrows():
            symbol = row['symbol']
            timeframe = row['timeframe']
            
            try:
                logger.info(f"\n📊 معالجة {idx+1}/{len(available_data)}")
                
                scores = self.train_symbol(symbol, timeframe)
                
                if scores:
                    successful.append({
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'accuracy': scores['test_accuracy']
                    })
                else:
                    failed.append(f"{symbol} {timeframe}")
                    
            except Exception as e:
                logger.error(f"❌ خطأ: {e}")
                failed.append(f"{symbol} {timeframe}")
        
        # الملخص
        logger.info("\n" + "="*80)
        logger.info("📊 الملخص النهائي")
        logger.info("="*80)
        logger.info(f"✅ نجح: {len(successful)}")
        logger.info(f"❌ فشل: {len(failed)}")
        
        if successful:
            logger.info("\n🏆 أفضل النماذج:")
            sorted_models = sorted(successful, key=lambda x: x['accuracy'], reverse=True)[:10]
            for model in sorted_models:
                logger.info(f"  • {model['symbol']} {model['timeframe']}: {model['accuracy']:.4f}")

def main():
    """تشغيل التدريب"""
    trainer = SimpleModelTrainer()
    
    # للاختبار السريع - عملات محددة
    test_symbols = [
        ("EURUSD", "M5"),
        ("GBPUSD", "M15"),
        ("XAUUSD", "H1"),
        ("BTCUSD", "H4")
    ]
    
    logger.info("🚀 بدء التدريب المبسط")
    
    # تدريب العملات المحددة
    for symbol, timeframe in test_symbols:
        try:
            trainer.train_symbol(symbol, timeframe)
        except Exception as e:
            logger.error(f"خطأ في {symbol} {timeframe}: {e}")
    
    logger.info("\n✅ اكتمل التدريب!")

if __name__ == "__main__":
    main()