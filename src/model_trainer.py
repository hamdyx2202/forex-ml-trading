import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import VotingClassifier
import joblib
from loguru import logger
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """تدريب وتقييم نماذج التعلم الآلي"""
    
    def __init__(self, config_path: str = "config/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.models_dir = "data/models"
        os.makedirs(self.models_dir, exist_ok=True)
        
        logger.add("logs/model_trainer.log", rotation="1 day", retention="30 days")
        
        self.scaler = None
        self.feature_columns = None
        self.best_model = None
        self.model_metrics = {}
        
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'target') -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """تحضير البيانات للتدريب"""
        # Remove non-feature columns
        exclude_cols = ['time', 'symbol', 'timeframe', 'target', 'target_3class', 
                       'open', 'high', 'low', 'close', 'volume']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        # Prepare features and target
        X = df[feature_cols].values
        y = df[target_col].values
        
        logger.info(f"Prepared data with shape: X={X.shape}, y={y.shape}")
        logger.info(f"Features: {len(feature_cols)}")
        
        return X, y, feature_cols
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """تطبيع الميزات"""
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_val: np.ndarray, y_val: np.ndarray) -> lgb.LGBMClassifier:
        """تدريب نموذج LightGBM"""
        logger.info("Training LightGBM model...")
        
        # Base parameters
        base_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'random_state': self.config['model']['random_state'],
            'n_jobs': -1,
            'verbose': -1
        }
        
        # Grid search parameters
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 100],
            'min_child_samples': [20, 30, 40]
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Create model
        lgb_model = lgb.LGBMClassifier(**base_params)
        
        # Grid search
        grid_search = GridSearchCV(
            lgb_model, param_grid, cv=tscv, 
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        # Fit model
        grid_search.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        best_model = grid_search.best_estimator_
        logger.info(f"Best LightGBM parameters: {grid_search.best_params_}")
        
        return best_model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray, y_val: np.ndarray) -> xgb.XGBClassifier:
        """تدريب نموذج XGBoost"""
        logger.info("Training XGBoost model...")
        
        # Base parameters
        base_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'random_state': self.config['model']['random_state'],
            'n_jobs': -1,
            'use_label_encoder': False
        }
        
        # Grid search parameters
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Create model
        xgb_model = xgb.XGBClassifier(**base_params)
        
        # Grid search
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=tscv, 
            scoring='roc_auc', n_jobs=-1, verbose=1
        )
        
        # Fit model
        eval_set = [(X_train, y_train), (X_val, y_val)]
        grid_search.fit(
            X_train, y_train,
            eval_set=eval_set,
            early_stopping_rounds=50,
            verbose=False
        )
        
        best_model = grid_search.best_estimator_
        logger.info(f"Best XGBoost parameters: {grid_search.best_params_}")
        
        return best_model
    
    def create_ensemble(self, models: List[Tuple[str, Any]]) -> VotingClassifier:
        """إنشاء نموذج مجمع"""
        logger.info("Creating ensemble model...")
        
        ensemble = VotingClassifier(
            estimators=models,
            voting='soft',
            n_jobs=-1
        )
        
        return ensemble
    
    def evaluate_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, float]:
        """تقييم النموذج"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted'),
            'auc_roc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) == 2 else 0
        }
        
        logger.info(f"{model_name} performance:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        self.model_metrics[model_name] = metrics
        
        return metrics
    
    def train_all_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تدريب جميع النماذج"""
        logger.info("Starting model training process...")
        
        # Prepare data
        X, y, feature_cols = self.prepare_data(df)
        
        # Split data
        train_size = int(len(X) * self.config['model']['train_test_split'])
        val_size = int(len(X) * self.config['model']['validation_split'])
        
        X_train = X[:train_size]
        y_train = y[:train_size]
        X_val = X[train_size:train_size + val_size]
        y_val = y[train_size:train_size + val_size]
        X_test = X[train_size + val_size:]
        y_test = y[train_size + val_size:]
        
        logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Scale features
        X_train_scaled, X_val_scaled = self.scale_features(X_train, X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train models
        models = {}
        
        # LightGBM
        lgb_model = self.train_lightgbm(X_train_scaled, y_train, X_val_scaled, y_val)
        models['lightgbm'] = lgb_model
        self.evaluate_model(lgb_model, X_test_scaled, y_test, 'LightGBM')
        
        # XGBoost
        xgb_model = self.train_xgboost(X_train_scaled, y_train, X_val_scaled, y_val)
        models['xgboost'] = xgb_model
        self.evaluate_model(xgb_model, X_test_scaled, y_test, 'XGBoost')
        
        # Ensemble
        ensemble_model = self.create_ensemble([
            ('lightgbm', lgb_model),
            ('xgboost', xgb_model)
        ])
        ensemble_model.fit(X_train_scaled, y_train)
        models['ensemble'] = ensemble_model
        self.evaluate_model(ensemble_model, X_test_scaled, y_test, 'Ensemble')
        
        # Select best model
        best_model_name = max(self.model_metrics.items(), 
                             key=lambda x: x[1]['f1_score'])[0]
        self.best_model = models[best_model_name.lower().replace(' ', '_')]
        
        logger.info(f"Best model: {best_model_name}")
        
        return models
    
    def save_models(self, models: Dict[str, Any], symbol: str, timeframe: str):
        """حفظ النماذج"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for model_name, model in models.items():
            filename = f"{symbol}_{timeframe}_{model_name}_{timestamp}.joblib"
            filepath = os.path.join(self.models_dir, filename)
            
            # Save model
            joblib.dump(model, filepath)
            logger.info(f"Saved {model_name} to {filepath}")
        
        # Save scaler
        scaler_file = f"{symbol}_{timeframe}_scaler_{timestamp}.joblib"
        scaler_path = os.path.join(self.models_dir, scaler_file)
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature columns
        features_file = f"{symbol}_{timeframe}_features_{timestamp}.json"
        features_path = os.path.join(self.models_dir, features_file)
        with open(features_path, 'w') as f:
            json.dump(self.feature_columns, f)
        
        # Save metrics
        metrics_file = f"{symbol}_{timeframe}_metrics_{timestamp}.json"
        metrics_path = os.path.join(self.models_dir, metrics_file)
        with open(metrics_path, 'w') as f:
            json.dump(self.model_metrics, f, indent=2)
        
        logger.info("All models and artifacts saved successfully")
    
    def load_model(self, symbol: str, timeframe: str, model_type: str = 'ensemble') -> Tuple[Any, Any, List[str]]:
        """تحميل النموذج المحفوظ"""
        # Find latest model files
        pattern = f"{symbol}_{timeframe}_{model_type}_*.joblib"
        model_files = [f for f in os.listdir(self.models_dir) if f.startswith(f"{symbol}_{timeframe}_{model_type}_")]
        
        if not model_files:
            raise FileNotFoundError(f"No model found for {symbol} {timeframe}")
        
        # Get latest file
        latest_file = sorted(model_files)[-1]
        timestamp = latest_file.split('_')[-1].replace('.joblib', '')
        
        # Load model
        model_path = os.path.join(self.models_dir, latest_file)
        model = joblib.load(model_path)
        
        # Load scaler
        scaler_file = f"{symbol}_{timeframe}_scaler_{timestamp}.joblib"
        scaler_path = os.path.join(self.models_dir, scaler_file)
        scaler = joblib.load(scaler_path)
        
        # Load features
        features_file = f"{symbol}_{timeframe}_features_{timestamp}.json"
        features_path = os.path.join(self.models_dir, features_file)
        with open(features_path, 'r') as f:
            features = json.load(f)
        
        logger.info(f"Loaded model from {model_path}")
        
        return model, scaler, features
    
    def calculate_feature_importance(self, model: Any, feature_names: List[str]) -> pd.DataFrame:
        """حساب أهمية الميزات"""
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'estimators_'):
            # For ensemble models
            importance = np.mean([
                est.feature_importances_ for est in model.estimators_
                if hasattr(est, 'feature_importances_')
            ], axis=0)
        else:
            logger.warning("Model does not support feature importance")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df


if __name__ == "__main__":
    # مثال على الاستخدام
    trainer = ModelTrainer()
    
    # Load prepared data
    # df = pd.read_csv("data/processed/EURUSD_H1_features.csv")
    
    # Train models
    # models = trainer.train_all_models(df)
    
    # Save models
    # trainer.save_models(models, "EURUSD", "H1")