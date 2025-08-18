#!/usr/bin/env python3
"""
🔍 فحص وتدريب من جدول price_data
📊 معالجة 7.8 مليون سجل
"""

import os
import sys
import sqlite3
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import joblib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)

def inspect_price_data():
    """فحص بنية جدول price_data"""
    db_path = './data/forex_ml.db'
    logger.info(f"🔍 فحص قاعدة البيانات: {db_path}")
    
    try:
        conn = sqlite3.connect(db_path)
        
        # فحص بنية الجدول
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(price_data)")
        columns = cursor.fetchall()
        
        logger.info("\n📋 أعمدة جدول price_data:")
        for col in columns:
            logger.info(f"   - {col[1]} ({col[2]})")
        
        # عينة من البيانات
        query = "SELECT * FROM price_data LIMIT 10"
        df_sample = pd.read_sql_query(query, conn)
        logger.info(f"\n📊 عينة من البيانات:")
        logger.info(df_sample)
        
        # إحصائيات الأزواج
        if 'symbol' in df_sample.columns:
            query = "SELECT symbol, COUNT(*) as count FROM price_data GROUP BY symbol ORDER BY count DESC"
            pairs_stats = pd.read_sql_query(query, conn)
            logger.info(f"\n📊 إحصائيات الأزواج:")
            for _, row in pairs_stats.head(20).iterrows():
                logger.info(f"   - {row['symbol']}: {row['count']:,} سجل")
        
        # إحصائيات الفريمات
        if 'timeframe' in df_sample.columns:
            query = "SELECT timeframe, COUNT(*) as count FROM price_data GROUP BY timeframe"
            tf_stats = pd.read_sql_query(query, conn)
            logger.info(f"\n⏰ إحصائيات الفريمات:")
            for _, row in tf_stats.iterrows():
                logger.info(f"   - {row['timeframe']}: {row['count']:,} سجل")
        
        conn.close()
        return df_sample.columns.tolist()
        
    except Exception as e:
        logger.error(f"❌ خطأ في الفحص: {str(e)}")
        return []

def train_from_price_data():
    """تدريب النماذج من جدول price_data"""
    db_path = './data/forex_ml.db'
    
    try:
        # استيراد النظام
        from optimized_forex_server import OptimizedForexSystem
        system = OptimizedForexSystem()
        system.db_path = db_path
        
        conn = sqlite3.connect(db_path)
        
        # جلب الأزواج المتاحة
        query = """
        SELECT symbol, COUNT(*) as count 
        FROM price_data 
        WHERE symbol NOT LIKE '%BRL%' 
        AND symbol NOT LIKE '%RUB%'
        AND symbol NOT LIKE '%ZAR%'
        GROUP BY symbol 
        HAVING count > 1000
        ORDER BY count DESC
        LIMIT 20
        """
        
        pairs = pd.read_sql_query(query, conn)
        logger.info(f"\n🎯 وجدت {len(pairs)} زوج للتدريب")
        
        trained_count = 0
        
        for _, pair_row in pairs.iterrows():
            symbol = pair_row['symbol']
            count = pair_row['count']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🎯 تدريب {symbol} ({count:,} سجل)")
            
            try:
                # جلب بيانات الزوج
                query = f"""
                SELECT * FROM price_data 
                WHERE symbol = '{symbol}'
                ORDER BY time DESC
                LIMIT 10000
                """
                
                df = pd.read_sql_query(query, conn)
                
                if df.empty:
                    logger.warning(f"⚠️ لا توجد بيانات لـ {symbol}")
                    continue
                
                # تحويل الوقت
                if 'time' in df.columns:
                    df['time'] = pd.to_datetime(df['time'])
                    df.set_index('time', inplace=True)
                
                # التأكد من وجود الأعمدة المطلوبة
                required_cols = ['open', 'high', 'low', 'close']
                if all(col in df.columns for col in required_cols):
                    # تحويل للأرقام
                    for col in required_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # إضافة الحجم إذا لم يكن موجود
                    if 'volume' not in df.columns:
                        df['volume'] = 1000
                    
                    # إزالة NaN
                    df = df.dropna()
                    
                    if len(df) < 1000:
                        logger.warning(f"⚠️ بيانات غير كافية بعد التنظيف: {len(df)}")
                        continue
                    
                    # حفظ البيانات المعالجة
                    processed_table = f"{symbol}_processed"
                    df.to_sql(processed_table, conn, if_exists='replace', index=True)
                    logger.info(f"✅ تم حفظ البيانات المعالجة في {processed_table}")
                    
                    # تدريب لفريمات مختلفة
                    timeframes = ['M15', 'M30', 'H1', 'H4', 'D1']
                    
                    for tf in timeframes:
                        try:
                            # محاولة التدريب
                            logger.info(f"   🤖 تدريب {symbol} {tf}...")
                            
                            # إنشاء نموذج بسيط إذا فشل التدريب الكامل
                            success = train_simple_model(system, symbol, tf, df)
                            
                            if success:
                                trained_count += 1
                                logger.info(f"   ✅ تم تدريب {symbol} {tf}")
                            else:
                                logger.warning(f"   ⚠️ فشل تدريب {symbol} {tf}")
                                
                        except Exception as e:
                            logger.error(f"   ❌ خطأ في {symbol} {tf}: {str(e)}")
                            
                else:
                    logger.warning(f"⚠️ أعمدة ناقصة في {symbol}")
                    
            except Exception as e:
                logger.error(f"❌ خطأ في معالجة {symbol}: {str(e)}")
        
        conn.close()
        
        logger.info(f"\n✅ تم تدريب {trained_count} نموذج")
        return trained_count
        
    except Exception as e:
        logger.error(f"❌ خطأ عام: {str(e)}")
        return 0

def train_simple_model(system, symbol, timeframe, df):
    """تدريب نموذج بسيط"""
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import RobustScaler
        from sklearn.model_selection import train_test_split
        
        # التأكد من أن df له index
        if not isinstance(df.index, pd.DatetimeIndex) and 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        
        # حساب ميزات بسيطة مع الحفاظ على نفس الـ index
        features = pd.DataFrame(index=df.index)
        
        # المتوسطات المتحركة
        features['sma_20'] = df['close'].rolling(20).mean()
        features['sma_50'] = df['close'].rolling(50).mean()
        features['sma_200'] = df['close'].rolling(200).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
        # التغيرات
        features['price_change'] = df['close'].pct_change()
        features['high_low_ratio'] = df['high'] / df['low']
        
        # إضافة الأسعار الأساسية
        features['open'] = df['open']
        features['high'] = df['high']
        features['low'] = df['low']
        features['close'] = df['close']
        
        # إزالة NaN
        features = features.dropna()
        
        if len(features) < 100:
            return False
        
        # إنشاء الهدف
        y = (df['close'].shift(-1) > df['close']).astype(int)
        
        # محاذاة البيانات - التأكد من أن features و y لهما نفس الـ index
        common_index = features.index.intersection(y.index).intersection(df.index)
        X = features.loc[common_index]
        y = y.loc[common_index]
        df_aligned = df.loc[common_index]
        
        if len(X) < 100:
            return False
        
        # إزالة آخر صف (لأن shift(-1) يخلق NaN في آخر صف)
        X = X[:-1]
        y = y[:-1]
        
        # التحقق من التطابق
        if len(X) != len(y):
            logger.error(f"عدم تطابق: X={len(X)}, y={len(y)}")
            return False
            
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values, test_size=0.2, random_state=42
        )
        
        # تطبيع
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # تدريب
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_scaled, y_train)
        
        # حفظ
        os.makedirs('./trained_models', exist_ok=True)
        
        model_path = f'./trained_models/{symbol}_{timeframe}_random_forest.pkl'
        joblib.dump(model, model_path)
        
        scaler_path = f'./trained_models/{symbol}_{timeframe}_scaler.pkl'
        joblib.dump(scaler, scaler_path)
        
        # حفظ في النظام
        key = f"{symbol}_{timeframe}"
        if key not in system.models:
            system.models[key] = {}
        system.models[key]['random_forest'] = model
        system.scalers[key] = scaler
        
        return True
        
    except Exception as e:
        logger.error(f"خطأ في النموذج البسيط: {str(e)}")
        return False

def create_optimized_training_script():
    """إنشاء سكريبت محسن للتدريب"""
    script = """#!/usr/bin/env python3
# سكريبت محسن لتدريب النماذج من price_data

import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import joblib
import os

# قائمة الأزواج الرئيسية
MAJOR_PAIRS = [
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD',
    'USDCAD', 'NZDUSD', 'EURJPY', 'GBPJPY', 'EURNZD'
]

def train_pair(symbol):
    print(f"Training {symbol}...")
    
    conn = sqlite3.connect('./data/forex_ml.db')
    
    # جلب البيانات
    query = f"SELECT * FROM price_data WHERE symbol = '{symbol}' ORDER BY time DESC LIMIT 5000"
    df = pd.read_sql_query(query, conn)
    
    if len(df) < 1000:
        print(f"Not enough data for {symbol}")
        return
    
    # معالجة البيانات
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    
    # حساب الميزات
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['rsi'] = calculate_rsi(df['close'])
    
    # إزالة NaN
    df = df.dropna()
    
    # الميزات والهدف
    features = ['open', 'high', 'low', 'close', 'sma_20', 'sma_50', 'rsi']
    X = df[features].values
    y = (df['close'].shift(-1) > df['close']).astype(int).values[:-1]
    X = X[:-1]
    
    # تدريب
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    
    # حفظ
    os.makedirs('./trained_models', exist_ok=True)
    joblib.dump(model, f'./trained_models/{symbol}_M15_model.pkl')
    
    print(f"✓ {symbol} trained successfully")
    conn.close()

def calculate_rsi(prices, period=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# تدريب جميع الأزواج
for pair in MAJOR_PAIRS:
    try:
        train_pair(pair)
    except Exception as e:
        print(f"Error training {pair}: {e}")

print("Training complete!")
"""
    
    with open('quick_train_models.py', 'w') as f:
        f.write(script)
    
    logger.info("✅ تم إنشاء quick_train_models.py")

def main():
    logger.info("\n" + "="*80)
    logger.info("🔍 فحص وتدريب من جدول price_data")
    logger.info("="*80)
    
    # فحص البنية
    columns = inspect_price_data()
    
    if not columns:
        logger.error("❌ فشل فحص قاعدة البيانات")
        return
    
    # تدريب النماذج
    logger.info("\n🤖 بدء التدريب...")
    trained = train_from_price_data()
    
    # إنشاء سكريبت سريع
    create_optimized_training_script()
    
    logger.info("\n" + "="*80)
    logger.info("📊 الملخص:")
    logger.info(f"✅ تم تدريب {trained} نموذج")
    logger.info(f"📁 النماذج في: ./trained_models/")
    logger.info(f"🚀 للتدريب السريع: python3 quick_train_models.py")
    logger.info("="*80)

if __name__ == "__main__":
    main()