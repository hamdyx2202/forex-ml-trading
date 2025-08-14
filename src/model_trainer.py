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
import glob
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
    
    def save_models(self, models: Dict[str, Any], symbol: str, timeframe: str, 
                   save_format: str = 'joblib', save_dir: str = None, 
                   use_advanced_format: bool = False):
        """حفظ النماذج"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use provided directory or default
        save_directory = save_dir or self.models_dir
        
        # Create advanced directory if using advanced format
        if use_advanced_format:
            save_directory = os.path.join(save_directory, "advanced")
            os.makedirs(save_directory, exist_ok=True)
        
        # Determine file extension
        file_ext = '.pkl' if save_format == 'pkl' else '.joblib'
        
        for model_name, model in models.items():
            if use_advanced_format:
                # Save in advanced format with all data in one file
                filename = f"{symbol}_{timeframe}_{model_name}_{timestamp}{file_ext}"
                filepath = os.path.join(save_directory, filename)
                
                # Create comprehensive model data dictionary
                model_data = {
                    'model': model,
                    'scaler': self.scaler,
                    'features': self.feature_columns,
                    'metrics': self.model_metrics.get(model_name, {}),
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'model_type': model_name
                }
                
                joblib.dump(model_data, filepath)
                logger.info(f"Saved {model_name} with all data to {filepath}")
            else:
                # Save in traditional format with separate files
                filename = f"{symbol}_{timeframe}_{model_name}_{timestamp}{file_ext}"
                filepath = os.path.join(save_directory, filename)
                
                # Save model
                joblib.dump(model, filepath)
                logger.info(f"Saved {model_name} to {filepath}")
        
        if not use_advanced_format:
            # Save scaler separately (only for traditional format)
            scaler_file = f"{symbol}_{timeframe}_scaler_{timestamp}{file_ext}"
            scaler_path = os.path.join(save_directory, scaler_file)
            joblib.dump(self.scaler, scaler_path)
            
            # Save feature columns
            features_file = f"{symbol}_{timeframe}_features_{timestamp}.json"
            features_path = os.path.join(save_directory, features_file)
            with open(features_path, 'w') as f:
                json.dump(self.feature_columns, f)
            
            # Save metrics
            metrics_file = f"{symbol}_{timeframe}_metrics_{timestamp}.json"
            metrics_path = os.path.join(save_directory, metrics_file)
            with open(metrics_path, 'w') as f:
                json.dump(self.model_metrics, f, indent=2)
        
        logger.info(f"All models and artifacts saved successfully to {save_directory}")
        
        # Also save a copy in the standard format for backward compatibility
        if use_advanced_format and save_format == 'pkl':
            logger.info("Creating backward compatible copies...")
            self.save_models(models, symbol, timeframe, save_format='joblib', 
                           save_dir=self.models_dir, use_advanced_format=False)
    
    def load_model(self, symbol: str, timeframe: str, model_type: str = 'ensemble') -> Tuple[Any, Any, List[str]]:
        """تحميل النموذج المحفوظ"""
        # Search in multiple directories for models
        search_dirs = [
            self.models_dir,  # data/models
            "models/advanced",  # advanced models directory
            "models/unified",   # unified models directory
            "models"  # root models directory
        ]
        
        model_files = []
        found_model_dir = None
        
        # Search for model files in each directory
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                # Search for both .pkl and .joblib files with timestamped patterns
                patterns = [
                    f"{symbol}_{timeframe}_{model_type}_*.pkl",
                    f"{symbol}_{timeframe}_{model_type}_*.joblib",
                    f"{symbol}m_PERIOD_{timeframe}_{model_type}_*.pkl",  # MT5 naming convention
                    f"{symbol}m_PERIOD_{timeframe}_{model_type}_*.joblib",
                ]
                
                for pattern in patterns:
                    found_files = glob.glob(os.path.join(search_dir, pattern))
                    if found_files:
                        model_files.extend([(f, search_dir) for f in found_files])
                        found_model_dir = search_dir
                        logger.info(f"Found {len(found_files)} models matching pattern {pattern} in {search_dir}")
        
        if not model_files:
            # Try simplified search without model_type for backwards compatibility
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    simplified_patterns = [
                        f"{symbol}_{timeframe}_*.pkl",
                        f"{symbol}_{timeframe}_*.joblib",
                        f"{symbol}m_PERIOD_{timeframe}_*.pkl",
                        f"{symbol}m_PERIOD_{timeframe}_*.joblib",
                    ]
                    
                    for pattern in simplified_patterns:
                        found_files = glob.glob(os.path.join(search_dir, pattern))
                        if found_files:
                            model_files.extend([(f, search_dir) for f in found_files])
                            found_model_dir = search_dir
                            logger.info(f"Found {len(found_files)} models with simplified pattern {pattern}")
        
        if not model_files:
            raise FileNotFoundError(f"No model found for {symbol} {timeframe} (searched in: {search_dirs})")
        
        # Sort by filename to get the most recent (timestamps in filename)
        model_files.sort(key=lambda x: x[0])  # Sort by full file path
        latest_file_path, model_dir = model_files[-1]  # Get the most recent
        latest_file = os.path.basename(latest_file_path)
        
        logger.info(f"Loading most recent model: {latest_file} from {model_dir}")
        
        # Extract timestamp from filename
        filename_without_ext = os.path.splitext(latest_file)[0]
        file_extension = os.path.splitext(latest_file)[1]
        
        # Try to extract timestamp from various filename patterns
        timestamp = None
        if '_ensemble_' in filename_without_ext:
            timestamp_part = filename_without_ext.split('_ensemble_')[-1]
            timestamp = timestamp_part
        elif len(filename_without_ext.split('_')) >= 2:
            # Try to get timestamp from the last parts
            parts = filename_without_ext.split('_')
            if len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit():
                timestamp = f"{parts[-2]}_{parts[-1]}"
            elif parts[-1].replace('_', '').replace('.', '').isdigit():
                timestamp = parts[-1]
        
        # Load model - handle both pkl and joblib files
        try:
            model_path = latest_file_path
            
            # For .pkl files that contain dictionaries (like advanced models)
            if file_extension == '.pkl':
                model_data = joblib.load(model_path)
                if isinstance(model_data, dict):
                    # Advanced model format with dictionary structure
                    model = model_data.get('model')
                    scaler = model_data.get('scaler')
                    features = model_data.get('features', [])
                    if features and not isinstance(features, list):
                        features = list(features) if hasattr(features, '__iter__') else []
                else:
                    # Simple model format
                    model = model_data
                    scaler = None
                    features = []
            else:
                # .joblib files
                model = joblib.load(model_path)
                scaler = None
                features = []
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            model = joblib.load(model_path)  # Fallback to simple loading
            scaler = None
            features = []
        
        # Try to load scaler and features separately if not already loaded
        if scaler is None and timestamp:
            scaler_patterns = [
                f"{symbol}_{timeframe}_scaler_{timestamp}{file_extension}",
                f"{symbol}m_PERIOD_{timeframe}_scaler_{timestamp}{file_extension}",
                f"{symbol}_{timeframe}_scaler_{timestamp}.joblib",
                f"{symbol}m_PERIOD_{timeframe}_scaler_{timestamp}.joblib",
            ]
            
            for pattern in scaler_patterns:
                scaler_path = os.path.join(model_dir, pattern)
                if os.path.exists(scaler_path):
                    try:
                        scaler = joblib.load(scaler_path)
                        logger.info(f"Loaded scaler from {scaler_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load scaler from {scaler_path}: {e}")
        
        if not features and timestamp:
            features_patterns = [
                f"{symbol}_{timeframe}_features_{timestamp}.json",
                f"{symbol}m_PERIOD_{timeframe}_features_{timestamp}.json",
            ]
            
            for pattern in features_patterns:
                features_path = os.path.join(model_dir, pattern)
                if os.path.exists(features_path):
                    try:
                        with open(features_path, 'r') as f:
                            features = json.load(f)
                        logger.info(f"Loaded features from {features_path}")
                        break
                    except Exception as e:
                        logger.warning(f"Failed to load features from {features_path}: {e}")
        
        # Fallback: if no separate scaler/features found, create empty defaults
        if scaler is None:
            logger.warning(f"No scaler found for {symbol} {timeframe}, using None")
        if not features:
            logger.warning(f"No feature list found for {symbol} {timeframe}, using empty list")
            features = []
        
        logger.info(f"Successfully loaded model from {model_path}")
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