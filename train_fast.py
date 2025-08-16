#!/usr/bin/env python3
"""
نظام التدريب السريع - نسخة محسنة للأداء
يركز على السرعة مع الحفاظ على الجودة
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import sqlite3
from pathlib import Path
import joblib
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
import ta

# إعداد السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FastTrainer:
    def __init__(self):
        self.models = {}
        self.scaler = StandardScaler()
        self.results = []
        
        # إعدادات محسنة للسرعة
        self.config = {
            'min_data_points': 5000,  # أقل من السابق
            'max_features': 50,  # ميزات أقل ولكن أكثر فعالية
            'test_size': 0.2,
            'n_jobs': multiprocessing.cpu_count() - 1,
            'quick_mode': True,  # وضع سريع
        }
        
        # إنشاء المجلدات
        Path("models").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
    
    def get_available_data(self):
        """الحصول على البيانات المتاحة"""
        try:
            conn = sqlite3.connect('data/forex_ml.db')
            query = """
                SELECT symbol, timeframe, COUNT(*) as count
                FROM price_data
                GROUP BY symbol, timeframe
                HAVING count >= ?
                ORDER BY count DESC
            """
            df = pd.read_sql_query(query, conn, params=(self.config['min_data_points'],))
            conn.close()
            
            logger.info(f"✅ وجدت {len(df)} مجموعة بيانات")
            return df.to_records(index=False).tolist()
            
        except Exception as e:
            logger.error(f"❌ خطأ في قراءة البيانات: {e}")
            return []
    
    def load_data(self, symbol, timeframe, limit=10000):
        """تحميل البيانات بشكل سريع"""
        try:
            conn = sqlite3.connect('data/forex_ml.db')
            query = """
                SELECT * FROM price_data 
                WHERE symbol = ? AND timeframe = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
            conn.close()
            
            if len(df) < self.config['min_data_points']:
                return None
            
            # ترتيب البيانات
            df = df.sort_values('timestamp')
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            logger.info(f"✅ تم تحميل {len(df)} سجل لـ {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل البيانات: {e}")
            return None
    
    def create_fast_features(self, df):
        """إنشاء ميزات سريعة وفعالة"""
        features = pd.DataFrame(index=df.index)
        
        # 1. ميزات السعر الأساسية (10 ميزات)
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['price_range'] = (df['high'] - df['low']) / df['close']
        features['body_size'] = abs(df['close'] - df['open']) / df['close']
        features['upper_shadow'] = (df['high'] - df[['close', 'open']].max(axis=1)) / df['close']
        features['lower_shadow'] = (df[['close', 'open']].min(axis=1) - df['low']) / df['close']
        features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # 2. المتوسطات المتحركة السريعة (15 ميزة)
        for period in [5, 10, 20, 50, 100]:
            sma = df['close'].rolling(period).mean()
            features[f'sma_{period}_ratio'] = df['close'] / sma
            features[f'sma_{period}_slope'] = sma.pct_change(5)
            
            if period <= 20:
                ema = df['close'].ewm(span=period).mean()
                features[f'ema_{period}_ratio'] = df['close'] / ema
        
        # 3. مؤشرات فنية أساسية (20 ميزة)
        # RSI
        features['rsi_14'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
        features['rsi_7'] = ta.momentum.RSIIndicator(df['close'], window=7).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        features['macd'] = macd.macd()
        features['macd_signal'] = macd.macd_signal()
        features['macd_diff'] = macd.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'])
        features['bb_upper'] = (bb.bollinger_hband() - df['close']) / df['close']
        features['bb_lower'] = (df['close'] - bb.bollinger_lband()) / df['close']
        features['bb_width'] = bb.bollinger_wband() / df['close']
        
        # ATR
        features['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range() / df['close']
        
        # ADX
        adx = ta.trend.ADXIndicator(df['high'], df['low'], df['close'])
        features['adx'] = adx.adx()
        features['adx_pos'] = adx.adx_pos()
        features['adx_neg'] = adx.adx_neg()
        
        # 4. ميزات إحصائية (5 ميزات)
        features['volatility_20'] = df['returns'].rolling(20).std()
        features['skew_20'] = df['returns'].rolling(20).skew()
        features['kurt_20'] = df['returns'].rolling(20).kurt()
        features['max_20'] = df['high'].rolling(20).max() / df['close']
        features['min_20'] = df['low'].rolling(20).min() / df['close']
        
        # إزالة القيم الناقصة
        features = features.fillna(method='ffill').fillna(0)
        
        # اختيار أفضل الميزات فقط
        if len(features.columns) > self.config['max_features']:
            # حساب الارتباط مع العائدات المستقبلية
            future_returns = df['close'].shift(-5).pct_change(5)
            correlations = features.corrwith(future_returns).abs()
            top_features = correlations.nlargest(self.config['max_features']).index
            features = features[top_features]
        
        logger.info(f"✅ تم إنشاء {len(features.columns)} ميزة")
        return features
    
    def create_labels(self, df, features):
        """إنشاء التسميات للتدريب"""
        # هدف بسيط: هل سيرتفع السعر 0.5% خلال 5 شموع؟
        future_returns = df['close'].shift(-5) / df['close'] - 1
        labels = (future_returns > 0.005).astype(int)
        
        # محاذاة البيانات
        mask = ~(features.isna().any(axis=1) | labels.isna())
        
        return features[mask], labels[mask]
    
    def train_single_model(self, X_train, X_test, y_train, y_test, model_type='rf'):
        """تدريب نموذج واحد بسرعة"""
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=50,  # أقل من السابق
                max_depth=10,
                min_samples_split=20,
                n_jobs=self.config['n_jobs'],
                random_state=42
            )
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=self.config['n_jobs'],
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif model_type == 'lgb':
            model = lgb.LGBMClassifier(
                n_estimators=50,
                max_depth=6,
                learning_rate=0.1,
                n_jobs=self.config['n_jobs'],
                random_state=42,
                verbose=-1
            )
        
        # التدريب
        model.fit(X_train, y_train)
        
        # التقييم
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return model, accuracy
    
    def train_models(self, features, labels):
        """تدريب جميع النماذج"""
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, 
            test_size=self.config['test_size'], 
            random_state=42,
            stratify=labels if len(np.unique(labels)) > 1 else None
        )
        
        # تحجيم البيانات
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        results = {}
        
        # تدريب النماذج
        if self.config['quick_mode']:
            # في الوضع السريع، ندرب نموذج واحد فقط
            logger.info("🚀 وضع التدريب السريع - Random Forest فقط")
            model, accuracy = self.train_single_model(
                X_train_scaled, X_test_scaled, y_train, y_test, 'rf'
            )
            results['rf'] = {'model': model, 'accuracy': accuracy}
        else:
            # تدريب 3 نماذج
            for model_type in ['rf', 'xgb', 'lgb']:
                logger.info(f"🤖 تدريب {model_type}...")
                model, accuracy = self.train_single_model(
                    X_train_scaled, X_test_scaled, y_train, y_test, model_type
                )
                results[model_type] = {'model': model, 'accuracy': accuracy}
        
        # اختيار أفضل نموذج
        best_model_type = max(results, key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model_type]['accuracy']
        
        logger.info(f"✅ أفضل نموذج: {best_model_type} بدقة {best_accuracy:.4f}")
        
        return results[best_model_type]['model'], best_accuracy, results
    
    def save_model(self, model, symbol, timeframe, accuracy):
        """حفظ النموذج والمعلومات"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # حفظ النموذج
        model_path = f"models/{symbol}_{timeframe}_fast_{timestamp}.pkl"
        joblib.dump({
            'model': model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'accuracy': accuracy,
            'timestamp': timestamp
        }, model_path)
        
        # حفظ المعلومات
        info = {
            'symbol': symbol,
            'timeframe': timeframe,
            'accuracy': accuracy,
            'model_path': model_path,
            'timestamp': timestamp,
            'features_count': len(self.feature_names)
        }
        
        self.results.append(info)
        logger.info(f"💾 تم حفظ النموذج: {model_path}")
        
        return model_path
    
    def train_pair(self, symbol, timeframe):
        """تدريب زوج واحد"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🎯 تدريب {symbol} {timeframe}")
        logger.info(f"{'='*60}")
        
        try:
            # تحميل البيانات
            df = self.load_data(symbol, timeframe)
            if df is None:
                return None
            
            # إنشاء الميزات
            features = self.create_fast_features(df)
            self.feature_names = features.columns.tolist()
            
            # إنشاء التسميات
            X, y = self.create_labels(df, features)
            
            if len(X) < 1000:
                logger.warning(f"⚠️ بيانات غير كافية بعد المعالجة: {len(X)}")
                return None
            
            # التدريب
            logger.info("🤖 بدء التدريب...")
            start_time = time.time()
            
            model, accuracy, all_results = self.train_models(X, y)
            
            training_time = time.time() - start_time
            logger.info(f"⏱️ وقت التدريب: {training_time:.1f} ثانية")
            
            # حفظ النموذج إذا كانت الدقة جيدة
            if accuracy >= 0.52:  # عتبة أقل للسرعة
                model_path = self.save_model(model, symbol, timeframe, accuracy)
                return {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'accuracy': accuracy,
                    'model_path': model_path,
                    'training_time': training_time
                }
            else:
                logger.warning(f"⚠️ دقة منخفضة: {accuracy:.4f}")
                return None
                
        except Exception as e:
            logger.error(f"❌ خطأ في التدريب: {e}")
            return None
    
    def train_all(self, max_pairs=None):
        """تدريب جميع الأزواج"""
        logger.info("\n" + "="*80)
        logger.info("🚀 بدء التدريب السريع")
        logger.info("="*80)
        
        # الحصول على البيانات المتاحة
        available_data = self.get_available_data()
        
        if max_pairs:
            available_data = available_data[:max_pairs]
        
        logger.info(f"📊 سيتم تدريب {len(available_data)} زوج")
        
        start_time = time.time()
        successful = 0
        
        # التدريب
        for i, (symbol, timeframe, count) in enumerate(available_data, 1):
            logger.info(f"\n📈 [{i}/{len(available_data)}] {symbol} {timeframe}")
            
            result = self.train_pair(symbol, timeframe)
            
            if result:
                successful += 1
                logger.info(f"✅ نجح التدريب - دقة: {result['accuracy']:.4f}")
            else:
                logger.warning(f"❌ فشل التدريب")
        
        # حفظ النتائج
        self.save_results()
        
        total_time = time.time() - start_time
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ اكتمل التدريب!")
        logger.info(f"📊 نجح: {successful}/{len(available_data)}")
        logger.info(f"⏱️ الوقت الإجمالي: {total_time/60:.1f} دقيقة")
        logger.info(f"⚡ متوسط الوقت لكل زوج: {total_time/len(available_data):.1f} ثانية")
        logger.info(f"{'='*80}")
    
    def save_results(self):
        """حفظ نتائج التدريب"""
        if not self.results:
            return
        
        # حفظ كـ JSON
        results_path = f"results/training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # حفظ كـ CSV
        df = pd.DataFrame(self.results)
        csv_path = results_path.replace('.json', '.csv')
        df.to_csv(csv_path, index=False)
        
        logger.info(f"💾 تم حفظ النتائج: {results_path}")

def main():
    """البرنامج الرئيسي"""
    import argparse
    
    parser = argparse.ArgumentParser(description='نظام التدريب السريع')
    parser.add_argument('--max-pairs', type=int, help='الحد الأقصى للأزواج')
    parser.add_argument('--full-mode', action='store_true', help='استخدام الوضع الكامل (3 نماذج)')
    
    args = parser.parse_args()
    
    # إنشاء المدرب
    trainer = FastTrainer()
    
    if args.full_mode:
        trainer.config['quick_mode'] = False
        logger.info("🔧 تم تفعيل الوضع الكامل (3 نماذج)")
    
    # بدء التدريب
    trainer.train_all(max_pairs=args.max_pairs)

if __name__ == "__main__":
    main()