#!/usr/bin/env python3
"""
Configuration file for the Ultimate Forex ML System
"""

# Database configuration
DATABASE_PATH = "data/forex_ml.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Data settings
MIN_DATA_POINTS = 1000
MAX_DATA_POINTS = 100000

# Model settings
MODEL_DIR = "models"
CHECKPOINT_DIR = "checkpoints"

# Training settings
TRAINING_CONFIG = {
    'test_size': 0.2,
    'validation_size': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': 1
}

# Feature settings
FEATURE_CONFIG = {
    'use_technical_indicators': True,
    'use_price_action': True,
    'use_volume': True,
    'use_market_structure': True,
    'max_features': 200
}

# Risk management
RISK_CONFIG = {
    'max_risk_per_trade': 0.02,
    'max_drawdown': 0.20,
    'position_sizing': 'fixed',
    'use_stop_loss': True,
    'use_take_profit': True
}

# API settings (if needed)
API_CONFIG = {
    'timeout': 30,
    'retry_count': 3,
    'rate_limit': 10  # requests per second
}

# Logging
LOG_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s [%(levelname)s] %(message)s',
    'file': 'system.log'
}