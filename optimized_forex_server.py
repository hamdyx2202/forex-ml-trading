#!/usr/bin/env python3
"""
🚀 Optimized Forex ML Server
📊 يعمل مع جدول price_data الحقيقي
🤖 تدريب وتنبؤ محسن
"""

import os
import sys
import json
import sqlite3
import logging
import threading
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

# المكتبات الاختيارية
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except:
    ML_AVAILABLE = False

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('optimized_forex_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)
CORS(app)

class OptimizedForexSystem:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.db_path = './data/forex_ml.db'
        self.models_dir = './trained_models'
        
        # إنشاء المجلدات
        os.makedirs(self.models_dir, exist_ok=True)
        
        # تحميل النماذج الموجودة
        self.load_existing_models()
        
        # معلومات الأزواج المتاحة
        self.available_pairs = self.get_available_pairs()
        
        logger.info(f"✅ تم تهيئة النظام - {len(self.models)} نموذج محمل")
        logger.info(f"📊 الأزواج المتاحة: {len(self.available_pairs)}")
    
    def get_available_pairs(self):
        """الحصول على الأزواج المتاحة من قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
            SELECT DISTINCT symbol, COUNT(*) as count 
            FROM price_data 
            WHERE symbol NOT LIKE '%BRL%' 
            AND symbol NOT LIKE '%RUB%'
            GROUP BY symbol 
            HAVING count > 1000
            """
            pairs = pd.read_sql_query(query, conn)
            conn.close()
            
            return pairs['symbol'].tolist()
        except:
            return []
    
    def load_existing_models(self):
        """تحميل النماذج المدربة"""
        if not os.path.exists(self.models_dir):
            return
        
        for file in os.listdir(self.models_dir):
            if file.endswith('_model.pkl'):
                try:
                    parts = file.replace('_model.pkl', '').split('_')
                    if len(parts) >= 2:
                        symbol = parts[0]
                        timeframe = parts[1] if len(parts) > 1 else 'M15'
                        
                        model = joblib.load(os.path.join(self.models_dir, file))
                        key = f"{symbol}_{timeframe}"
                        self.models[key] = model
                        
                        # محاولة تحميل المقياس
                        scaler_file = file.replace('_model.pkl', '_scaler.pkl')
                        scaler_path = os.path.join(self.models_dir, scaler_file)
                        if os.path.exists(scaler_path):
                            self.scalers[key] = joblib.load(scaler_path)
                            
                except Exception as e:
                    logger.error(f"خطأ في تحميل {file}: {str(e)}")
    
    def calculate_features(self, df):
        """حساب الميزات البسيطة والفعالة"""
        features = pd.DataFrame(index=df.index)
        
        # الأسعار الأساسية
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['close'] = df['close']
        
        # المتوسطات المتحركة
        features['sma_10'] = df['close'].rolling(10).mean()
        features['sma_20'] = df['close'].rolling(20).mean()
        features['sma_50'] = df['close'].rolling(50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        features['macd'] = exp1 - exp2
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        sma = df['close'].rolling(window=20).mean()
        std = df['close'].rolling(window=20).std()
        features['bb_upper'] = sma + (std * 2)
        features['bb_lower'] = sma - (std * 2)
        features['bb_width'] = features['bb_upper'] - features['bb_lower']
        
        # النسب
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # ATR
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        features['atr'] = true_range.rolling(14).mean()
        
        # إزالة NaN
        features = features.dropna()
        
        return features
    
    def train_model(self, symbol, timeframe='M15'):
        """تدريب نموذج لزوج محدد"""
        try:
            logger.info(f"🤖 تدريب {symbol} {timeframe}...")
            
            # جلب البيانات
            conn = sqlite3.connect(self.db_path)
            query = f"""
            SELECT * FROM price_data 
            WHERE symbol = '{symbol}'
            ORDER BY time DESC
            LIMIT 5000
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) < 1000:
                logger.warning(f"⚠️ بيانات غير كافية لـ {symbol}")
                return False
            
            # معالجة البيانات
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # تحويل للأرقام
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            
            # حساب الميزات
            features = self.calculate_features(df)
            
            if len(features) < 100:
                return False
            
            # إعداد البيانات للتدريب
            X = features.values
            y = (df['close'].shift(-1) > df['close']).astype(int)
            y = y[features.index].values[:-1]
            X = X[:-1]
            
            # تقسيم البيانات
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # تطبيع
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # تدريب
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            model.fit(X_train_scaled, y_train)
            
            # تقييم
            accuracy = model.score(X_test_scaled, y_test)
            logger.info(f"   ✅ الدقة: {accuracy:.2%}")
            
            # حفظ
            key = f"{symbol}_{timeframe}"
            self.models[key] = model
            self.scalers[key] = scaler
            
            # حفظ على القرص
            model_path = os.path.join(self.models_dir, f"{symbol}_{timeframe}_model.pkl")
            scaler_path = os.path.join(self.models_dir, f"{symbol}_{timeframe}_scaler.pkl")
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ خطأ في التدريب: {str(e)}")
            return False
    
    def predict(self, symbol, timeframe, df):
        """التنبؤ"""
        try:
            key = f"{symbol}_{timeframe}"
            
            # إذا لم يكن هناك نموذج، حاول التدريب
            if key not in self.models:
                logger.info(f"📚 لا يوجد نموذج لـ {key}, جاري التدريب...")
                if not self.train_model(symbol, timeframe):
                    return self.simple_prediction(df)
            
            # حساب الميزات
            features = self.calculate_features(df)
            
            if features.empty:
                return self.simple_prediction(df)
            
            # التنبؤ
            X = features.values[-1:] 
            
            if key in self.scalers:
                X = self.scalers[key].transform(X)
            
            model = self.models[key]
            prediction = model.predict(X)[0]
            
            # الثقة
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                confidence = max(proba)
            else:
                confidence = 0.65
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"خطأ في التنبؤ: {str(e)}")
            return self.simple_prediction(df)
    
    def simple_prediction(self, df):
        """تنبؤ بسيط بناءً على المتوسطات"""
        try:
            latest = df.iloc[-1]
            sma20 = df['close'].rolling(20).mean().iloc[-1]
            sma50 = df['close'].rolling(50).mean().iloc[-1]
            
            if latest['close'] > sma20 > sma50:
                return 0, 0.65  # Buy
            elif latest['close'] < sma20 < sma50:
                return 1, 0.65  # Sell
            else:
                return 2, 0.5   # Hold
        except:
            return 2, 0.5

# إنشاء مثيل النظام
system = OptimizedForexSystem()

@app.route('/status', methods=['GET'])
def status():
    """حالة السيرفر"""
    return jsonify({
        'status': 'running',
        'version': '5.0-optimized',
        'models_loaded': len(system.models),
        'available_pairs': len(system.available_pairs),
        'ml_available': ML_AVAILABLE,
        'pairs_sample': system.available_pairs[:10]
    })

@app.route('/predict', methods=['POST'])
def predict():
    """التنبؤ"""
    try:
        # معالجة JSON
        raw_data = request.get_data(as_text=True)
        
        # محاولة إصلاح JSON
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError as e:
            # محاولة الإصلاح
            if e.pos > 0 and e.pos < len(raw_data):
                try:
                    # قطع عند موضع الخطأ وإغلاق JSON
                    fixed = raw_data[:e.pos]
                    open_brackets = fixed.count('[') - fixed.count(']')
                    open_braces = fixed.count('{') - fixed.count('}')
                    fixed += ']' * open_brackets + '}' * open_braces
                    data = json.loads(fixed)
                    logger.info("✅ تم إصلاح JSON")
                except:
                    return jsonify({'error': 'Invalid JSON', 'action': 'NONE', 'confidence': 0})
            else:
                return jsonify({'error': 'Invalid JSON', 'action': 'NONE', 'confidence': 0})
        
        # استخراج البيانات
        symbol = data.get('symbol', 'UNKNOWN')
        timeframe = data.get('timeframe', 'M15')
        candles = data.get('candles', [])
        
        # تنظيف اسم الرمز
        clean_symbol = symbol.replace('m', '').replace('.ecn', '').replace('_pro', '')
        
        logger.info(f"📊 طلب: {symbol} ({clean_symbol}) {timeframe} - {len(candles)} شمعة")
        
        if len(candles) < 50:
            return jsonify({
                'symbol': symbol,
                'action': 'NONE',
                'confidence': 0,
                'error': 'Not enough candles'
            })
        
        # تحويل لـ DataFrame
        df = pd.DataFrame(candles)
        for col in ['open', 'high', 'low', 'close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        
        df = df.dropna()
        
        # التنبؤ
        prediction, confidence = system.predict(clean_symbol, timeframe, df)
        
        # تحديد الإشارة
        current_price = float(df['close'].iloc[-1])
        pip_value = 0.01 if 'JPY' in symbol else 0.0001
        
        if prediction == 0 and confidence >= 0.65:
            action = 'BUY'
        elif prediction == 1 and confidence >= 0.65:
            action = 'SELL'
        else:
            action = 'NONE'
        
        # حساب SL/TP
        atr = df['high'].rolling(14).max().iloc[-1] - df['low'].rolling(14).min().iloc[-1]
        sl_distance = max(min(atr * 1.5, 100 * pip_value), 30 * pip_value)
        
        if action == 'BUY':
            sl_price = current_price - sl_distance
            tp1_price = current_price + (sl_distance * 2)
            tp2_price = current_price + (sl_distance * 3)
        elif action == 'SELL':
            sl_price = current_price + sl_distance
            tp1_price = current_price - (sl_distance * 2)
            tp2_price = current_price - (sl_distance * 3)
        else:
            sl_price = tp1_price = tp2_price = current_price
        
        response = {
            'symbol': symbol,
            'timeframe': timeframe,
            'action': action,
            'confidence': float(confidence),
            'current_price': current_price,
            'sl_price': float(sl_price),
            'tp1_price': float(tp1_price),
            'tp2_price': float(tp2_price),
            'sl_pips': float(sl_distance / pip_value),
            'tp1_pips': float(sl_distance / pip_value * 2),
            'tp2_pips': float(sl_distance / pip_value * 3),
            'model_exists': f"{clean_symbol}_{timeframe}" in system.models
        }
        
        logger.info(f"   ✅ {action} بثقة {confidence:.1%}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"خطأ: {str(e)}")
        return jsonify({
            'error': str(e),
            'action': 'NONE',
            'confidence': 0
        })

@app.route('/train', methods=['POST'])
def train():
    """تدريب نموذج"""
    try:
        data = request.json
        symbol = data.get('symbol', '').replace('m', '').replace('.ecn', '')
        timeframe = data.get('timeframe', 'M15')
        
        success = system.train_model(symbol, timeframe)
        
        return jsonify({
            'success': success,
            'message': f'Trained {symbol} {timeframe}' if success else 'Training failed'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models', methods=['GET'])
def models():
    """قائمة النماذج"""
    return jsonify({
        'total': len(system.models),
        'models': list(system.models.keys()),
        'available_pairs': system.available_pairs
    })

@app.route('/train_all', methods=['POST'])
def train_all():
    """تدريب جميع الأزواج"""
    def train_task():
        for symbol in system.available_pairs[:20]:  # أول 20 زوج
            try:
                system.train_model(symbol)
            except:
                pass
    
    thread = threading.Thread(target=train_task)
    thread.start()
    
    return jsonify({
        'message': 'Training started in background',
        'pairs': len(system.available_pairs)
    })

if __name__ == '__main__':
    logger.info("\n" + "="*60)
    logger.info("🚀 OPTIMIZED FOREX ML SERVER")
    logger.info("📊 Working with real price_data table")
    logger.info("🌐 Server: http://0.0.0.0:5000")
    logger.info("="*60 + "\n")
    
    # تدريب بعض النماذج عند البدء
    logger.info("🤖 تدريب النماذج الأساسية...")
    for symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']:
        if symbol in system.available_pairs:
            system.train_model(symbol)
    
    # تشغيل السيرفر
    app.run(host='0.0.0.0', port=5000, debug=False)