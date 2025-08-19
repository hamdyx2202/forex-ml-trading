import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
import talib
import MetaTrader5 as mt5

warnings.filterwarnings('ignore')

# إعداد السجلات
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('forex_system.log'),
        logging.StreamHandler()
    ]
)

class AdvancedForexSystem:
    def __init__(self):
        # إعدادات النظام
        self.pairs = ['EURAUDm', 'EURCADm', 'EURCHFm', 'EURGBPm', 'EURJPYm', 'EURNZDm', 'EURUSDm',
                     'GBPAUDm', 'GBPCADm', 'GBPCHFm', 'GBPJPYm', 'GBPNZDm', 'GBPUSDm',
                     'USDCADm', 'USDCHFm', 'USDJPYm', 'AUDCADm', 'AUDCHFm', 'AUDJPYm', 
                     'AUDNZDm', 'AUDUSDm', 'CADCHFm', 'CADJPYm', 'CHFJPYm', 'NZDCADm',
                     'NZDCHFm', 'NZDJPYm', 'NZDUSDm']
        
        self.timeframes = [mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15, mt5.TIMEFRAME_M30, 
                          mt5.TIMEFRAME_H1, mt5.TIMEFRAME_H4]
        
        self.timeframe_names = {
            mt5.TIMEFRAME_M5: 'M5',
            mt5.TIMEFRAME_M15: 'M15', 
            mt5.TIMEFRAME_M30: 'M30',
            mt5.TIMEFRAME_H1: 'H1',
            mt5.TIMEFRAME_H4: 'H4'
        }
        
        # معلمات النظام
        self.lookback_candles = 24
        self.target_candles = 5
        self.commission_points = 5
        self.max_data_points = 30000
        
        # نماذج التعلم الآلي
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        
        # معلمات النماذج
        self.model_params = {
            'lightgbm': {
                'objective': 'multiclass',
                'num_class': 3,
                'metric': 'multi_logloss',
                'boosting': 'gbdt',
                'num_leaves': 127,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'max_depth': 15,
                'min_data_in_leaf': 50,
                'lambda_l1': 0.1,
                'lambda_l2': 0.1,
                'verbose': -1,
                'n_estimators': 500,
                'early_stopping_rounds': 50
            },
            'xgboost': {
                'objective': 'multi:softprob',
                'num_class': 3,
                'max_depth': 10,
                'learning_rate': 0.05,
                'n_estimators': 500,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'reg_alpha': 0.1,
                'reg_lambda': 1,
                'eval_metric': 'mlogloss'
            },
            'random_forest': {
                'n_estimators': 300,
                'max_depth': 20,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'max_features': 'sqrt',
                'bootstrap': True,
                'class_weight': 'balanced',
                'n_jobs': -1
            },
            'gradient_boosting': {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 10,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'subsample': 0.8,
                'max_features': 'sqrt'
            },
            'extra_trees': {
                'n_estimators': 300,
                'max_depth': 20,
                'min_samples_split': 20,
                'min_samples_leaf': 10,
                'max_features': 'sqrt',
                'bootstrap': False,
                'class_weight': 'balanced',
                'n_jobs': -1
            },
            'neural_network': {
                'hidden_layer_sizes': (200, 150, 100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'alpha': 0.001,
                'learning_rate': 'adaptive',
                'max_iter': 1000,
                'early_stopping': True,
                'validation_fraction': 0.1
            }
        }
        
    def connect_mt5(self):
        """الاتصال بـ MetaTrader 5"""
        if not mt5.initialize():
            logging.error("فشل في تهيئة MT5")
            return False
        logging.info("✅ تم الاتصال بـ MT5 بنجاح")
        return True
        
    def get_data(self, symbol, timeframe, count=30000):
        """الحصول على البيانات من MT5"""
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        
        if rates is None or len(rates) == 0:
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
        
    def calculate_all_features(self, df):
        """حساب جميع الميزات المتقدمة (200+ ميزة)"""
        features = pd.DataFrame(index=df.index)
        
        # 1. الميزات الأساسية
        features['returns'] = df['close'].pct_change()
        features['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        features['high_low_ratio'] = df['high'] / df['low']
        features['close_open_ratio'] = df['close'] / df['open']
        
        # 2. المتوسطات المتحركة (20 ميزة)
        for period in [5, 10, 20, 50, 100, 200]:
            features[f'sma_{period}'] = talib.SMA(df['close'], period)
            features[f'ema_{period}'] = talib.EMA(df['close'], period)
            features[f'price_sma_{period}_ratio'] = df['close'] / features[f'sma_{period}']
            
        # 3. مؤشرات الزخم (30 ميزة)
        features['rsi_14'] = talib.RSI(df['close'], 14)
        features['rsi_28'] = talib.RSI(df['close'], 28)
        features['rsi_7'] = talib.RSI(df['close'], 7)
        
        macd, macd_signal, macd_hist = talib.MACD(df['close'])
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_hist'] = macd_hist
        
        features['cci_14'] = talib.CCI(df['high'], df['low'], df['close'], 14)
        features['cci_28'] = talib.CCI(df['high'], df['low'], df['close'], 28)
        
        features['mom_10'] = talib.MOM(df['close'], 10)
        features['mom_20'] = talib.MOM(df['close'], 20)
        
        features['roc_10'] = talib.ROC(df['close'], 10)
        features['roc_20'] = talib.ROC(df['close'], 20)
        
        # 4. مؤشرات التذبذب (20 ميزة)
        features['atr_14'] = talib.ATR(df['high'], df['low'], df['close'], 14)
        features['atr_28'] = talib.ATR(df['high'], df['low'], df['close'], 28)
        
        features['natr_14'] = talib.NATR(df['high'], df['low'], df['close'], 14)
        features['trange'] = talib.TRANGE(df['high'], df['low'], df['close'])
        
        # Bollinger Bands
        upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
        features['bb_upper'] = upper
        features['bb_middle'] = middle
        features['bb_lower'] = lower
        features['bb_width'] = upper - lower
        features['bb_position'] = (df['close'] - lower) / (upper - lower)
        
        # 5. مؤشرات الحجم (15 ميزة)
        if 'tick_volume' in df.columns:
            features['volume_sma_10'] = talib.SMA(df['tick_volume'], 10)
            features['volume_ratio'] = df['tick_volume'] / features['volume_sma_10']
            features['obv'] = talib.OBV(df['close'], df['tick_volume'])
            features['ad'] = talib.AD(df['high'], df['low'], df['close'], df['tick_volume'])
            features['adosc'] = talib.ADOSC(df['high'], df['low'], df['close'], df['tick_volume'])
            
        # 6. الأنماط الشمعية (30 ميزة)
        candle_patterns = [
            'CDL2CROWS', 'CDL3BLACKCROWS', 'CDL3INSIDE', 'CDL3LINESTRIKE',
            'CDL3OUTSIDE', 'CDL3STARSINSOUTH', 'CDL3WHITESOLDIERS', 'CDLABANDONEDBABY',
            'CDLADVANCEBLOCK', 'CDLBELTHOLD', 'CDLBREAKAWAY', 'CDLCLOSINGMARUBOZU',
            'CDLCONCEALBABYSWALL', 'CDLCOUNTERATTACK', 'CDLDARKCLOUDCOVER', 'CDLDOJI',
            'CDLDOJISTAR', 'CDLDRAGONFLYDOJI', 'CDLENGULFING', 'CDLEVENINGDOJISTAR',
            'CDLEVENINGSTAR', 'CDLGAPSIDESIDEWHITE', 'CDLGRAVESTONEDOJI', 'CDLHAMMER',
            'CDLHANGINGMAN', 'CDLHARAMI', 'CDLHARAMICROSS', 'CDLHIGHWAVE',
            'CDLHIKKAKE', 'CDLHIKKAKEMOD'
        ]
        
        for pattern in candle_patterns:
            features[f'pattern_{pattern}'] = getattr(talib, pattern)(df['open'], df['high'], df['low'], df['close'])
            
        # 7. مؤشرات دورة السوق (15 ميزة)
        features['ht_dcperiod'] = talib.HT_DCPERIOD(df['close'])
        features['ht_dcphase'] = talib.HT_DCPHASE(df['close'])
        features['ht_trendmode'] = talib.HT_TRENDMODE(df['close'])
        
        sine, leadsine = talib.HT_SINE(df['close'])
        features['ht_sine'] = sine
        features['ht_leadsine'] = leadsine
        
        # 8. إحصائيات النافذة المتحركة (25 ميزة)
        for window in [5, 10, 20, 50]:
            features[f'rolling_mean_{window}'] = df['close'].rolling(window).mean()
            features[f'rolling_std_{window}'] = df['close'].rolling(window).std()
            features[f'rolling_skew_{window}'] = df['close'].rolling(window).skew()
            features[f'rolling_kurt_{window}'] = df['close'].rolling(window).kurt()
            features[f'rolling_min_{window}'] = df['close'].rolling(window).min()
            features[f'rolling_max_{window}'] = df['close'].rolling(window).max()
            
        # 9. النسب والفروقات (20 ميزة)
        features['high_low_spread'] = df['high'] - df['low']
        features['close_open_spread'] = df['close'] - df['open']
        features['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
        features['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']
        features['body_size'] = np.abs(df['close'] - df['open'])
        features['shadow_body_ratio'] = (features['upper_shadow'] + features['lower_shadow']) / (features['body_size'] + 0.001)
        
        # 10. مؤشرات القوة النسبية (15 ميزة)
        features['mfi_14'] = talib.MFI(df['high'], df['low'], df['close'], df['tick_volume'] if 'tick_volume' in df else df['close'], 14)
        features['willr_14'] = talib.WILLR(df['high'], df['low'], df['close'], 14)
        features['ultosc'] = talib.ULTOSC(df['high'], df['low'], df['close'])
        
        # 11. مؤشرات الاتجاه (20 ميزة)
        features['adx_14'] = talib.ADX(df['high'], df['low'], df['close'], 14)
        features['adx_28'] = talib.ADX(df['high'], df['low'], df['close'], 28)
        features['plus_di_14'] = talib.PLUS_DI(df['high'], df['low'], df['close'], 14)
        features['minus_di_14'] = talib.MINUS_DI(df['high'], df['low'], df['close'], 14)
        features['plus_dm_14'] = talib.PLUS_DM(df['high'], df['low'], 14)
        features['minus_dm_14'] = talib.MINUS_DM(df['high'], df['low'], 14)
        
        features['aroon_up'], features['aroon_down'] = talib.AROON(df['high'], df['low'], 14)
        features['aroonosc_14'] = talib.AROONOSC(df['high'], df['low'], 14)
        
        # 12. مستويات الدعم والمقاومة (10 ميزة)
        features['pivot'] = (df['high'] + df['low'] + df['close']) / 3
        features['r1'] = 2 * features['pivot'] - df['low']
        features['s1'] = 2 * features['pivot'] - df['high']
        features['r2'] = features['pivot'] + (df['high'] - df['low'])
        features['s2'] = features['pivot'] - (df['high'] - df['low'])
        
        # 13. التحليل الزمني (15 ميزة)
        features['hour'] = df.index.hour
        features['day_of_week'] = df.index.dayofweek
        features['day_of_month'] = df.index.day
        features['month'] = df.index.month
        features['quarter'] = df.index.quarter
        
        features['is_london_session'] = ((features['hour'] >= 8) & (features['hour'] < 16)).astype(int)
        features['is_ny_session'] = ((features['hour'] >= 13) & (features['hour'] < 21)).astype(int)
        features['is_tokyo_session'] = ((features['hour'] >= 0) & (features['hour'] < 8)).astype(int)
        features['is_sydney_session'] = ((features['hour'] >= 22) | (features['hour'] < 6)).astype(int)
        
        # 14. التغيرات والفروقات الزمنية (20 ميزة)
        for i in [1, 2, 3, 5, 10]:
            features[f'price_change_{i}'] = df['close'].diff(i)
            features[f'price_pct_change_{i}'] = df['close'].pct_change(i)
            features[f'volume_change_{i}'] = df['tick_volume'].diff(i) if 'tick_volume' in df else 0
            
        # 15. نسب فيبوناتشي (8 ميزة)
        period_high = df['high'].rolling(20).max()
        period_low = df['low'].rolling(20).min()
        diff = period_high - period_low
        
        features['fib_236'] = period_high - 0.236 * diff
        features['fib_382'] = period_high - 0.382 * diff
        features['fib_500'] = period_high - 0.500 * diff
        features['fib_618'] = period_high - 0.618 * diff
        
        # 16. مؤشرات مخصصة (15 ميزة)
        features['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
        features['trend_strength'] = features['adx_14'] * np.sign(features['macd'])
        features['momentum_oscillator'] = features['rsi_14'] * features['mfi_14'] / 100
        features['volatility_ratio'] = features['atr_14'] / features['rolling_mean_20']
        
        # التعامل مع القيم المفقودة
        features = features.fillna(method='ffill').fillna(0)
        features = features.replace([np.inf, -np.inf], 0)
        
        # حفظ أسماء الميزات
        self.feature_names = features.columns.tolist()
        
        return features
        
    def create_targets(self, df, future_candles=5, commission=5):
        """إنشاء التصنيفات المستهدفة"""
        targets = []
        
        for i in range(len(df) - future_candles):
            current_price = df['close'].iloc[i]
            future_prices = df['close'].iloc[i+1:i+future_candles+1]
            
            max_price = future_prices.max()
            min_price = future_prices.min()
            
            profit_buy = max_price - current_price - commission
            profit_sell = current_price - min_price - commission
            
            if profit_buy > commission and profit_buy > profit_sell:
                targets.append(0)  # Buy
            elif profit_sell > commission:
                targets.append(1)  # Sell
            else:
                targets.append(2)  # Hold
                
        # إضافة تصنيفات Hold للصفوف الأخيرة
        targets.extend([2] * future_candles)
        
        return np.array(targets)
        
    def prepare_data(self, df):
        """إعداد البيانات للتدريب"""
        # حساب الميزات
        features = self.calculate_all_features(df)
        
        # إنشاء التصنيفات
        targets = self.create_targets(df, self.target_candles, self.commission_points)
        
        # التأكد من تطابق الأطوال
        min_len = min(len(features), len(targets))
        features = features.iloc[:min_len]
        targets = targets[:min_len]
        
        # إزالة الصفوف الأولى التي تحتوي على NaN
        start_idx = 200  # بعد حساب أطول مؤشر
        features = features.iloc[start_idx:]
        targets = targets[start_idx:]
        
        return features, targets
        
    def train_models(self, features, targets, pair, timeframe):
        """تدريب جميع النماذج"""
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            features, targets, test_size=0.2, random_state=42, stratify=targets
        )
        
        # تطبيع البيانات
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model_key = f"{pair}_{timeframe}"
        self.scalers[model_key] = scaler
        
        models = {}
        accuracies = {}
        
        # 1. LightGBM
        logging.info("  Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(**self.model_params['lightgbm'])
        lgb_model.fit(X_train_scaled, y_train, 
                      eval_set=[(X_test_scaled, y_test)],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        
        lgb_pred = lgb_model.predict(X_test_scaled)
        lgb_acc = accuracy_score(y_test, lgb_pred)
        models['lightgbm'] = lgb_model
        accuracies['lightgbm'] = lgb_acc
        logging.info(f"    LightGBM Accuracy: {lgb_acc:.2%}")
        
        # 2. XGBoost
        logging.info("  Training XGBoost...")
        xgb_model = xgb.XGBClassifier(**self.model_params['xgboost'])
        xgb_model.fit(X_train_scaled, y_train,
                      eval_set=[(X_test_scaled, y_test)],
                      verbose=False)
        
        xgb_pred = xgb_model.predict(X_test_scaled)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        models['xgboost'] = xgb_model
        accuracies['xgboost'] = xgb_acc
        logging.info(f"    XGBoost Accuracy: {xgb_acc:.2%}")
        
        # 3. Random Forest
        logging.info("  Training Random Forest...")
        rf_model = RandomForestClassifier(**self.model_params['random_forest'])
        rf_model.fit(X_train_scaled, y_train)
        
        rf_pred = rf_model.predict(X_test_scaled)
        rf_acc = accuracy_score(y_test, rf_pred)
        models['random_forest'] = rf_model
        accuracies['random_forest'] = rf_acc
        logging.info(f"    Random Forest Accuracy: {rf_acc:.2%}")
        
        # 4. Gradient Boosting
        logging.info("  Training Gradient Boosting...")
        gb_model = GradientBoostingClassifier(**self.model_params['gradient_boosting'])
        gb_model.fit(X_train_scaled, y_train)
        
        gb_pred = gb_model.predict(X_test_scaled)
        gb_acc = accuracy_score(y_test, gb_pred)
        models['gradient_boosting'] = gb_model
        accuracies['gradient_boosting'] = gb_acc
        logging.info(f"    Gradient Boosting Accuracy: {gb_acc:.2%}")
        
        # 5. Extra Trees
        logging.info("  Training Extra Trees...")
        et_model = ExtraTreesClassifier(**self.model_params['extra_trees'])
        et_model.fit(X_train_scaled, y_train)
        
        et_pred = et_model.predict(X_test_scaled)
        et_acc = accuracy_score(y_test, et_pred)
        models['extra_trees'] = et_model
        accuracies['extra_trees'] = et_acc
        logging.info(f"    Extra Trees Accuracy: {et_acc:.2%}")
        
        # 6. Neural Network
        logging.info("  Training Neural Network...")
        nn_model = MLPClassifier(**self.model_params['neural_network'])
        nn_model.fit(X_train_scaled, y_train)
        
        nn_pred = nn_model.predict(X_test_scaled)
        nn_acc = accuracy_score(y_test, nn_pred)
        models['neural_network'] = nn_model
        accuracies['neural_network'] = nn_acc
        logging.info(f"    Neural Network Accuracy: {nn_acc:.2%}")
        
        # حساب متوسط الدقة
        avg_accuracy = np.mean(list(accuracies.values()))
        logging.info(f"  ✅ Average Model Accuracy: {avg_accuracy:.2%}")
        
        # حفظ النماذج
        self.models[model_key] = models
        
        # إنشاء تقرير التصنيف للنموذج الأفضل
        best_model_name = max(accuracies, key=accuracies.get)
        best_model = models[best_model_name]
        best_pred = best_model.predict(X_test_scaled)
        
        logging.info(f"\n📊 Classification Report for {best_model_name} (Best Model):")
        report = classification_report(y_test, best_pred, 
                                     target_names=['Buy', 'Sell', 'Hold'],
                                     output_dict=True)
        
        for label, metrics in report.items():
            if label in ['Buy', 'Sell', 'Hold']:
                logging.info(f"  {label}: Precision={metrics['precision']:.2%}, "
                           f"Recall={metrics['recall']:.2%}, F1={metrics['f1-score']:.2%}")
        
        return avg_accuracy
        
    def save_models(self, pair, timeframe):
        """حفظ النماذج والمعايرات"""
        model_key = f"{pair}_{timeframe}"
        
        # إنشاء مجلد النماذج
        os.makedirs('models', exist_ok=True)
        
        # حفظ النماذج
        model_path = f"models/{model_key}_models.pkl"
        joblib.dump(self.models[model_key], model_path)
        
        # حفظ المعاير
        scaler_path = f"models/{model_key}_scaler.pkl"
        joblib.dump(self.scalers[model_key], scaler_path)
        
        # حفظ أسماء الميزات
        features_path = f"models/{model_key}_features.pkl"
        joblib.dump(self.feature_names, features_path)
        
        logging.info(f"💾 Models saved: {model_path}")
        
    def train_all_pairs(self):
        """تدريب جميع الأزواج والأطر الزمنية"""
        if not self.connect_mt5():
            return
            
        total_combinations = len(self.pairs) * len(self.timeframes)
        completed = 0
        
        for pair in self.pairs:
            for timeframe in self.timeframes:
                completed += 1
                tf_name = self.timeframe_names[timeframe]
                
                logging.info(f"\n🎯 Training {pair} {tf_name} ({completed}/{total_combinations})")
                
                # الحصول على البيانات
                df = self.get_data(pair, timeframe, self.max_data_points)
                
                if df is None or len(df) < 1000:
                    logging.warning(f"  ⚠️ Not enough data for {pair} {tf_name}")
                    continue
                    
                logging.info(f"✅ Loaded {len(df)} records for {pair} {tf_name}")
                
                try:
                    # إعداد البيانات
                    features, targets = self.prepare_data(df)
                    
                    # عرض توزيع التصنيفات
                    unique, counts = np.unique(targets, return_counts=True)
                    target_dist = dict(zip(['Buy', 'Sell', 'Hold'], counts))
                    logging.info(f"  Targets: Buy={target_dist.get('Buy', 0)}, "
                               f"Sell={target_dist.get('Sell', 0)}, "
                               f"Hold={target_dist.get('Hold', 0)}")
                    
                    # تدريب النماذج
                    accuracy = self.train_models(features, targets, pair, tf_name)
                    
                    # حفظ النماذج
                    self.save_models(pair, tf_name)
                    
                except Exception as e:
                    logging.error(f"❌ Error training {pair} {tf_name}: {str(e)}")
                    continue
                    
        mt5.shutdown()
        logging.info("\n✅ Training completed for all pairs!")
        
    def predict(self, pair, timeframe, features):
        """التنبؤ باستخدام جميع النماذج"""
        model_key = f"{pair}_{timeframe}"
        
        if model_key not in self.models:
            logging.error(f"No models found for {model_key}")
            return None
            
        # تطبيع البيانات
        features_scaled = self.scalers[model_key].transform(features)
        
        # التنبؤ من جميع النماذج
        predictions = {}
        probabilities = {}
        
        for model_name, model in self.models[model_key].items():
            pred = model.predict(features_scaled)
            prob = model.predict_proba(features_scaled)
            
            predictions[model_name] = pred
            probabilities[model_name] = prob
            
        # التصويت بالأغلبية
        all_predictions = np.array(list(predictions.values()))
        ensemble_prediction = stats.mode(all_predictions, axis=0)[0].flatten()
        
        # حساب متوسط الاحتماليات
        ensemble_probabilities = np.mean(list(probabilities.values()), axis=0)
        
        return ensemble_prediction, ensemble_probabilities
        
def main():
    system = AdvancedForexSystem()
    system.train_all_pairs()
    
if __name__ == "__main__":
    main()