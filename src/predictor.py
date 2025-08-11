import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import joblib
from loguru import logger
import json
from datetime import datetime
import os


class Predictor:
    """التنبؤ باستخدام النماذج المدربة"""
    
    def __init__(self, config_path: str = "config/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.models = {}
        self.scalers = {}
        self.feature_columns = {}
        
        logger.add("logs/predictor.log", rotation="1 day", retention="30 days")
    
    def load_model_for_pair(self, symbol: str, timeframe: str, model_type: str = 'ensemble'):
        """تحميل النموذج لزوج عملات محدد"""
        try:
            from src.model_trainer import ModelTrainer
            trainer = ModelTrainer()
            
            model, scaler, features = trainer.load_model(symbol, timeframe, model_type)
            
            key = f"{symbol}_{timeframe}"
            self.models[key] = model
            self.scalers[key] = scaler
            self.feature_columns[key] = features
            
            logger.info(f"Loaded model for {symbol} {timeframe}")
            
        except Exception as e:
            logger.error(f"Failed to load model for {symbol} {timeframe}: {str(e)}")
            raise
    
    def prepare_features(self, df: pd.DataFrame, symbol: str, timeframe: str) -> np.ndarray:
        """تحضير الميزات للتنبؤ"""
        key = f"{symbol}_{timeframe}"
        
        if key not in self.feature_columns:
            raise ValueError(f"No features loaded for {symbol} {timeframe}")
        
        # Get required features
        required_features = self.feature_columns[key]
        
        # Check if all features exist
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            # Create missing features with zeros
            for feature in missing_features:
                df[feature] = 0
        
        # Select and order features
        X = df[required_features].values
        
        # Scale features
        if key in self.scalers:
            X = self.scalers[key].transform(X)
        
        return X
    
    def predict(self, df: pd.DataFrame, symbol: str, timeframe: str) -> pd.DataFrame:
        """إجراء التنبؤات"""
        key = f"{symbol}_{timeframe}"
        
        if key not in self.models:
            self.load_model_for_pair(symbol, timeframe)
        
        # Prepare features
        X = self.prepare_features(df, symbol, timeframe)
        
        # Make predictions
        model = self.models[key]
        
        # Get predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # Create results dataframe
        results = pd.DataFrame({
            'time': df['time'] if 'time' in df.columns else pd.Timestamp.now(),
            'symbol': symbol,
            'timeframe': timeframe,
            'prediction': predictions,
            'probability_down': probabilities[:, 0],
            'probability_up': probabilities[:, 1],
            'confidence': np.max(probabilities, axis=1)
        })
        
        # Add signal strength
        results['signal_strength'] = self._calculate_signal_strength(probabilities)
        
        # Add recommendation
        results['recommendation'] = self._generate_recommendation(results)
        
        logger.info(f"Generated {len(results)} predictions for {symbol} {timeframe}")
        
        return results
    
    def predict_latest(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """التنبؤ بأحدث نقطة"""
        from src.data_collector import MT5DataCollector
        from src.feature_engineer import FeatureEngineer
        
        # Get latest data
        collector = MT5DataCollector()
        df = collector.get_latest_data(symbol, timeframe, limit=500)
        
        if df.empty:
            logger.error(f"No data available for {symbol} {timeframe}")
            return {}
        
        # Create features
        engineer = FeatureEngineer()
        df_features = engineer.create_features(df)
        
        if df_features.empty:
            logger.error("Failed to create features")
            return {}
        
        # Get latest row
        latest_row = df_features.iloc[[-1]]
        
        # Make prediction
        results = self.predict(latest_row, symbol, timeframe)
        
        if results.empty:
            return {}
        
        # Convert to dictionary
        prediction = results.iloc[0].to_dict()
        
        # Add current price info
        prediction['current_price'] = float(df.iloc[-1]['close'])
        prediction['timestamp'] = datetime.now().isoformat()
        
        return prediction
    
    def predict_multiple_timeframes(self, symbol: str) -> pd.DataFrame:
        """التنبؤ عبر إطارات زمنية متعددة"""
        timeframes = self.config['trading']['timeframes']
        all_predictions = []
        
        for timeframe in timeframes:
            try:
                prediction = self.predict_latest(symbol, timeframe)
                if prediction:
                    all_predictions.append(prediction)
            except Exception as e:
                logger.error(f"Failed to predict {symbol} {timeframe}: {str(e)}")
        
        if not all_predictions:
            return pd.DataFrame()
        
        # Combine predictions
        results = pd.DataFrame(all_predictions)
        
        # Calculate overall signal
        results['combined_signal'] = self._calculate_combined_signal(results)
        
        return results
    
    def _calculate_signal_strength(self, probabilities: np.ndarray) -> np.ndarray:
        """حساب قوة الإشارة"""
        # Calculate how confident the model is
        max_prob = np.max(probabilities, axis=1)
        
        # Map to signal strength (0-100)
        signal_strength = (max_prob - 0.5) * 200
        signal_strength = np.clip(signal_strength, 0, 100)
        
        return signal_strength
    
    def _generate_recommendation(self, results: pd.DataFrame) -> List[str]:
        """توليد التوصيات"""
        recommendations = []
        
        for _, row in results.iterrows():
            confidence = row['confidence']
            prediction = row['prediction']
            signal_strength = row['signal_strength']
            
            if confidence < self.config['trading']['min_confidence']:
                rec = "NO_TRADE"
            elif prediction == 1 and signal_strength > 70:
                rec = "STRONG_BUY"
            elif prediction == 1 and signal_strength > 50:
                rec = "BUY"
            elif prediction == 0 and signal_strength > 70:
                rec = "STRONG_SELL"
            elif prediction == 0 and signal_strength > 50:
                rec = "SELL"
            else:
                rec = "HOLD"
            
            recommendations.append(rec)
        
        return recommendations
    
    def _calculate_combined_signal(self, df: pd.DataFrame) -> str:
        """حساب الإشارة المجمعة من إطارات زمنية متعددة"""
        if df.empty:
            return "NEUTRAL"
        
        # Weight by timeframe (longer timeframes have more weight)
        timeframe_weights = {
            'M5': 0.1,
            'M15': 0.15,
            'H1': 0.25,
            'H4': 0.3,
            'D1': 0.2
        }
        
        weighted_score = 0
        total_weight = 0
        
        for _, row in df.iterrows():
            timeframe = row['timeframe']
            weight = timeframe_weights.get(timeframe, 0.1)
            
            # Convert prediction and confidence to score
            score = row['prediction'] * row['confidence'] * row['signal_strength'] / 100
            
            weighted_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            final_score = weighted_score / total_weight
        else:
            final_score = 0.5
        
        # Generate final signal
        if final_score > 0.7:
            return "STRONG_BUY"
        elif final_score > 0.6:
            return "BUY"
        elif final_score < 0.3:
            return "STRONG_SELL"
        elif final_score < 0.4:
            return "SELL"
        else:
            return "NEUTRAL"
    
    def get_prediction_history(self, symbol: str, timeframe: str, days: int = 7) -> pd.DataFrame:
        """الحصول على تاريخ التنبؤات"""
        # This would typically load from a database
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def evaluate_predictions(self, predictions: pd.DataFrame, actual: pd.DataFrame) -> Dict[str, float]:
        """تقييم دقة التنبؤات"""
        if predictions.empty or actual.empty:
            return {}
        
        # Merge predictions with actual
        merged = pd.merge(
            predictions[['time', 'symbol', 'prediction', 'confidence']],
            actual[['time', 'symbol', 'actual_movement']],
            on=['time', 'symbol'],
            how='inner'
        )
        
        if merged.empty:
            return {}
        
        # Calculate metrics
        correct = (merged['prediction'] == merged['actual_movement']).sum()
        total = len(merged)
        
        # High confidence predictions
        high_conf = merged[merged['confidence'] > 0.7]
        high_conf_correct = (high_conf['prediction'] == high_conf['actual_movement']).sum()
        
        metrics = {
            'overall_accuracy': correct / total if total > 0 else 0,
            'high_confidence_accuracy': high_conf_correct / len(high_conf) if len(high_conf) > 0 else 0,
            'total_predictions': total,
            'high_confidence_count': len(high_conf)
        }
        
        return metrics


if __name__ == "__main__":
    # مثال على الاستخدام
    predictor = Predictor()
    
    # Get latest prediction
    # prediction = predictor.predict_latest("EURUSD", "H1")
    # print(json.dumps(prediction, indent=2))
    
    # Get predictions for multiple timeframes
    # multi_predictions = predictor.predict_multiple_timeframes("EURUSD")
    # print(multi_predictions)