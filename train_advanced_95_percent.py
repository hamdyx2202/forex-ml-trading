#!/usr/bin/env python3
"""
Advanced Training System for 95% Accuracy
نظام التدريب المتقدم للوصول لدقة 95%
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False
    
try:
    import optuna
    OPTUNA_AVAILABLE = True
except:
    OPTUNA_AVAILABLE = False

import joblib
import sqlite3
from datetime import datetime
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import our feature engineer
from feature_engineer_fixed_v2 import FeatureEngineer

class AdvancedModelTrainer:
    """نظام التدريب المتقدم للوصول لدقة عالية جداً"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        Path("models/advanced").mkdir(parents=True, exist_ok=True)
        Path("logs").mkdir(exist_ok=True)
        
    def prepare_data(self, symbol, timeframe):
        """تحضير البيانات مع جميع المؤشرات المتقدمة"""
        print(f"\n📊 Preparing data for {symbol} {timeframe}...")
        
        # جلب البيانات
        conn = sqlite3.connect("data/forex_ml.db")
        query = """
            SELECT time, open, high, low, close, volume, spread
            FROM price_data
            WHERE symbol = ? AND timeframe = ?
            ORDER BY time
        """
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe))
        conn.close()
        
        if len(df) < 1000:
            print(f"⚠️ Not enough data for {symbol} {timeframe}")
            return None, None, None
            
        # تحويل الوقت
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('datetime', inplace=True)
        
        # إضافة المؤشرات
        print("  • Adding advanced technical indicators...")
        engineer = FeatureEngineer()
        df_features = engineer.create_features(
            df, 
            target_config={'lookahead': 5, 'threshold': 0.001}
        )
        
        if df_features.empty or 'target' not in df_features.columns:
            return None, None, None
            
        # إزالة الصفوف بدون هدف واضح
        df_features = df_features[df_features['target'] != 0]
        
        # تحويل الهدف لثنائي (1 للصعود، 0 للهبوط)
        df_features['target_binary'] = (df_features['target'] > 0).astype(int)
        
        # اختيار الميزات
        feature_cols = [col for col in df_features.columns 
                       if col not in ['target', 'target_binary', 'target_3class', 
                                     'future_return', 'time', 'open', 'high', 
                                     'low', 'close', 'volume', 'spread']]
        
        X = df_features[feature_cols]
        y = df_features['target_binary']
        
        print(f"  • Total samples: {len(X)}")
        print(f"  • Features: {len(feature_cols)}")
        print(f"  • Class distribution: {y.value_counts().to_dict()}")
        
        return X, y, feature_cols
        
    def optimize_lgb_params(self, X_train, y_train, X_val, y_val):
        """تحسين معاملات LightGBM تلقائياً"""
        if not OPTUNA_AVAILABLE:
            return {
                'n_estimators': 300,
                'learning_rate': 0.05,
                'num_leaves': 31,
                'max_depth': -1,
                'min_child_samples': 20,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
            
        print("  • Optimizing LightGBM parameters with Optuna...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': 42,
                'verbose': -1
            }
            
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            accuracy = accuracy_score(y_val, preds)
            
            return accuracy
            
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50, show_progress_bar=True)
        
        best_params = study.best_params
        best_params['random_state'] = 42
        best_params['verbose'] = -1
        
        return best_params
        
    def train_ensemble_model(self, X, y, symbol, timeframe):
        """تدريب نموذج مجمع قوي جداً"""
        print(f"\n🚀 Training advanced ensemble for {symbol} {timeframe}...")
        
        # تقسيم البيانات
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # تقسيم إضافي للتحقق
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # تحجيم البيانات
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # حفظ المحجم
        self.scalers[f"{symbol}_{timeframe}"] = scaler
        
        # 1. LightGBM مع معاملات محسنة
        print("\n  📈 Training LightGBM...")
        lgb_params = self.optimize_lgb_params(X_train, y_train, X_val, y_val)
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train)
        
        # 2. XGBoost
        print("  📊 Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        
        # 3. CatBoost (إذا كان متاحاً)
        models_list = [
            ('lgb', lgb_model),
            ('xgb', xgb_model)
        ]
        
        if CATBOOST_AVAILABLE:
            print("  🐱 Training CatBoost...")
            cb_model = cb.CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                random_state=42,
                verbose=False
            )
            cb_model.fit(X_train, y_train)
            models_list.append(('catboost', cb_model))
        
        # 4. Random Forest
        print("  🌲 Training Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train_scaled, y_train)
        models_list.append(('rf', rf_model))
        
        # 5. النموذج المجمع النهائي
        print("  🎯 Creating final ensemble...")
        ensemble = VotingClassifier(
            estimators=models_list,
            voting='soft'
        )
        
        # تدريب النموذج المجمع
        ensemble.fit(X_train, y_train)
        
        # تقييم الأداء
        print("\n📊 Evaluating performance...")
        predictions = ensemble.predict(X_test)
        proba = ensemble.predict_proba(X_test)
        
        # حساب المقاييس
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        f1 = f1_score(y_test, predictions)
        
        # حساب الثقة
        confidence_scores = np.max(proba, axis=1)
        high_confidence_mask = confidence_scores > 0.7
        high_conf_accuracy = accuracy_score(
            y_test[high_confidence_mask], 
            predictions[high_confidence_mask]
        ) if sum(high_confidence_mask) > 0 else 0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'high_confidence_accuracy': high_conf_accuracy,
            'high_confidence_trades': sum(high_confidence_mask),
            'total_test_samples': len(y_test)
        }
        
        print(f"\n✅ Results for {symbol} {timeframe}:")
        print(f"  • Overall Accuracy: {accuracy:.2%}")
        print(f"  • Precision: {precision:.2%}")
        print(f"  • Recall: {recall:.2%}")
        print(f"  • F1 Score: {f1:.2%}")
        print(f"  • High Confidence Accuracy: {high_conf_accuracy:.2%}")
        print(f"  • High Confidence Trades: {sum(high_confidence_mask)}/{len(y_test)}")
        
        # حفظ النموذج والمقاييس
        self.models[f"{symbol}_{timeframe}"] = ensemble
        self.performance_metrics[f"{symbol}_{timeframe}"] = metrics
        
        # حساب أهمية الميزات
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': lgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance[f"{symbol}_{timeframe}"] = feature_importance
        
        return ensemble, metrics
        
    def save_models(self):
        """حفظ جميع النماذج والبيانات"""
        print("\n💾 Saving advanced models...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # حفظ النماذج
        for key, model in self.models.items():
            model_path = f"models/advanced/{key}_ensemble_{timestamp}.pkl"
            joblib.dump({
                'model': model,
                'scaler': self.scalers.get(key),
                'metrics': self.performance_metrics.get(key),
                'feature_importance': self.feature_importance.get(key),
                'timestamp': timestamp
            }, model_path)
            print(f"  • Saved: {model_path}")
        
        # حفظ ملخص الأداء
        summary = {
            'timestamp': timestamp,
            'models': list(self.models.keys()),
            'overall_metrics': self.performance_metrics,
            'average_accuracy': np.mean([m['accuracy'] for m in self.performance_metrics.values()]),
            'average_high_conf_accuracy': np.mean([m['high_confidence_accuracy'] for m in self.performance_metrics.values()])
        }
        
        with open(f"models/advanced/training_summary_{timestamp}.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n🎯 Average Accuracy: {summary['average_accuracy']:.2%}")
        print(f"🎯 Average High-Confidence Accuracy: {summary['average_high_conf_accuracy']:.2%}")
        
    def train_all_pairs(self):
        """تدريب جميع الأزواج"""
        print("="*80)
        print("🧠 ADVANCED TRAINING SYSTEM FOR 95% ACCURACY")
        print("="*80)
        
        # الحصول على جميع الأزواج
        conn = sqlite3.connect("data/forex_ml.db")
        cursor = conn.execute("""
            SELECT DISTINCT symbol, timeframe, COUNT(*) as count
            FROM price_data
            GROUP BY symbol, timeframe
            HAVING count > 1000
            ORDER BY count DESC
        """)
        pairs = cursor.fetchall()
        conn.close()
        
        print(f"\n📊 Found {len(pairs)} pairs to train")
        
        success_count = 0
        for symbol, timeframe, count in pairs:
            print(f"\n{'='*60}")
            print(f"Processing {symbol} {timeframe} ({count:,} bars)")
            print(f"{'='*60}")
            
            try:
                # تحضير البيانات
                X, y, features = self.prepare_data(symbol, timeframe)
                
                if X is not None and len(X) > 500:
                    # تدريب النموذج
                    model, metrics = self.train_ensemble_model(X, y, symbol, timeframe)
                    success_count += 1
                else:
                    print("❌ Insufficient data for training")
                    
            except Exception as e:
                print(f"❌ Error training {symbol} {timeframe}: {e}")
                
        # حفظ جميع النماذج
        if success_count > 0:
            self.save_models()
            
        print("\n" + "="*80)
        print(f"✅ Training completed: {success_count}/{len(pairs)} models")
        print("="*80)

def main():
    trainer = AdvancedModelTrainer()
    trainer.train_all_pairs()

if __name__ == "__main__":
    main()