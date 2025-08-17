#!/bin/bash
# تثبيت المتطلبات الأساسية للنظام

echo "🔧 Installing basic requirements for Ultimate Forex ML System..."

# Core packages
pip install numpy pandas scikit-learn joblib scipy

# ML Models
pip install xgboost lightgbm

# Try to install CatBoost (might fail on some systems)
pip install catboost || echo "⚠️ CatBoost installation failed, continuing without it"

# Database
pip install sqlalchemy

# Technical Analysis
pip install ta pandas-ta || echo "⚠️ TA libraries installation failed"

# Basic utilities
pip install requests python-dateutil pytz matplotlib

echo "✅ Basic requirements installed successfully!"
echo ""
echo "📌 Note: Some optional packages were skipped:"
echo "   - aiohttp/aiofiles (for async operations)"
echo "   - tensorflow (for deep learning)"
echo "   - optuna/shap (for optimization)"
echo ""
echo "To install all optional packages, run:"
echo "pip install -r requirements_ultimate.txt"