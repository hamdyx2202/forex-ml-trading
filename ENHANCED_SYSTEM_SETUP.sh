#!/bin/bash

echo "=========================================="
echo "🚀 ENHANCED FOREX ML TRADING SYSTEM SETUP"
echo "💰 نظام التداول الذكي المتكامل"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "\n${YELLOW}📋 Checking requirements...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 not found${NC}"
    exit 1
fi

# Install required packages
echo -e "\n${YELLOW}📦 Installing required packages...${NC}"
pip3 install -r requirements.txt

# Create necessary directories
echo -e "\n${YELLOW}📁 Creating directories...${NC}"
mkdir -p data
mkdir -p trained_models
mkdir -p logs
mkdir -p archive_old_files

# Check database
if [ ! -f "data/forex_ml.db" ]; then
    echo -e "${YELLOW}⚠️  Database not found. Please ensure you have price data.${NC}"
fi

# Make scripts executable
chmod +x enhanced_ml_server.py
chmod +x market_analysis_engine.py
chmod +x risk_management_system.py
chmod +x test_enhanced_system.py

echo -e "\n${GREEN}✅ Setup complete!${NC}"
echo -e "\n${YELLOW}📖 System Components:${NC}"
echo "1. Market Analysis Engine - تحليل السوق الشامل"
echo "2. Risk Management System - إدارة المخاطر الذكية"
echo "3. Enhanced ML Server - 6 نماذج ML متقدمة"
echo "4. Dynamic SL/TP - وقف خسارة وأهداف ديناميكية"
echo "5. Trade Validation - تحقق متعدد المستويات"

echo -e "\n${YELLOW}🚀 To start the system:${NC}"
echo "python3 enhanced_ml_server.py"

echo -e "\n${YELLOW}🧪 To test the system:${NC}"
echo "python3 test_enhanced_system.py"

echo -e "\n${GREEN}💰 System ready for profitable trading!${NC}"
echo "=========================================="