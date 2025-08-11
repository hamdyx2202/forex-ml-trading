#!/usr/bin/env python3
"""
Script for training all models
"""

from src.data_collector import MT5DataCollector
from src.feature_engineer import FeatureEngineer
from src.model_trainer import ModelTrainer
from loguru import logger
import json


def main():
    """Train models for all configured pairs and timeframes"""
    logger.add("logs/training.log", rotation="1 day")
    
    # Load configuration
    with open("config/config.json", 'r') as f:
        config = json.load(f)
    
    # Initialize components
    collector = MT5DataCollector()
    engineer = FeatureEngineer()
    trainer = ModelTrainer()
    
    # Get pairs and timeframes
    pairs = config['trading']['pairs']
    timeframes = ['H1', 'H4']  # Focus on higher timeframes for better signals
    
    logger.info(f"Starting model training for {len(pairs)} pairs and {len(timeframes)} timeframes")
    
    for symbol in pairs:
        for timeframe in timeframes:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Training model for {symbol} {timeframe}")
                logger.info(f"{'='*50}")
                
                # Get historical data
                logger.info("Fetching historical data...")
                df = collector.get_latest_data(symbol, timeframe, limit=20000)
                
                if df.empty or len(df) < 1000:
                    logger.warning(f"Insufficient data for {symbol} {timeframe}")
                    continue
                
                logger.info(f"Loaded {len(df)} rows of data")
                
                # Create features
                logger.info("Creating features...")
                df_features = engineer.create_features(
                    df,
                    target_config={
                        'target_type': 'classification',
                        'lookahead': 5,
                        'threshold': 0.001
                    }
                )
                
                if df_features.empty:
                    logger.warning(f"Failed to create features for {symbol} {timeframe}")
                    continue
                
                logger.info(f"Created {len(df_features.columns)} features")
                
                # Train models
                logger.info("Training models...")
                models = trainer.train_all_models(df_features)
                
                # Save models
                logger.info("Saving models...")
                trainer.save_models(models, symbol, timeframe)
                
                # Display results
                logger.info("\nModel Performance:")
                for model_name, metrics in trainer.model_metrics.items():
                    logger.info(f"\n{model_name}:")
                    for metric, value in metrics.items():
                        logger.info(f"  {metric}: {value:.4f}")
                
                # Feature importance
                if trainer.best_model:
                    importance_df = trainer.calculate_feature_importance(
                        trainer.best_model,
                        trainer.feature_columns
                    )
                    if not importance_df.empty:
                        logger.info("\nTop 10 Most Important Features:")
                        for idx, row in importance_df.head(10).iterrows():
                            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
                
                logger.success(f"✅ Successfully trained models for {symbol} {timeframe}")
                
            except Exception as e:
                logger.error(f"❌ Error training {symbol} {timeframe}: {str(e)}")
                continue
    
    logger.info("\n" + "="*50)
    logger.info("🎉 Model training completed!")
    logger.info("="*50)


if __name__ == "__main__":
    main()