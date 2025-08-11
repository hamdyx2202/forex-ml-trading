#!/bin/bash

# Deployment script for updating the Forex Trading Bot on VPS
# Run this script from your local machine

set -e

# Configuration
VPS_USER="your_username"
VPS_HOST="your_vps_ip"
VPS_PATH="/home/$VPS_USER/forex-ml-trading"
LOCAL_PATH="."

echo "==================================="
echo "Deploying Forex ML Trading Bot"
echo "==================================="

# Check if rsync is installed
if ! command -v rsync &> /dev/null; then
    echo "rsync is required but not installed. Please install it."
    exit 1
fi

# Sync files to VPS
echo "1. Syncing files to VPS..."
rsync -avz --exclude-from='.gitignore' \
    --exclude '.git' \
    --exclude 'venv' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'data/raw/*' \
    --exclude 'data/processed/*' \
    --exclude 'data/models/*' \
    --exclude 'logs/*' \
    --exclude '.env' \
    $LOCAL_PATH/ $VPS_USER@$VPS_HOST:$VPS_PATH/

# Run remote commands
echo "2. Running deployment commands on VPS..."
ssh $VPS_USER@$VPS_HOST << 'EOF'
    cd /home/$USER/forex-ml-trading
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install/update dependencies
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    # Run migrations if needed
    # python manage.py migrate
    
    # Restart services
    echo "Restarting services..."
    sudo systemctl restart forex-trading-bot
    sudo systemctl restart forex-monitoring
    
    # Check service status
    echo "Checking service status..."
    sudo systemctl status forex-trading-bot --no-pager
    sudo systemctl status forex-monitoring --no-pager
    
    echo "Deployment completed!"
EOF

echo "==================================="
echo "Deployment finished successfully!"
echo "==================================="