#!/bin/bash

# Setup script for Hostinger VPS
# Run this script on a fresh Ubuntu 22.04 installation

set -e  # Exit on error

echo "==================================="
echo "Forex ML Trading Bot - VPS Setup"
echo "==================================="

# Update system
echo "1. Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install required system packages
echo "2. Installing system dependencies..."
sudo apt install -y \
    python3.9 \
    python3.9-dev \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    htop \
    screen \
    supervisor \
    nginx \
    ufw \
    sqlite3 \
    build-essential \
    libssl-dev \
    libffi-dev

# Install Wine for MT5 (optional - if running MT5 on same server)
echo "3. Installing Wine for MT5 (optional)..."
read -p "Do you want to install Wine for MT5? (y/n): " install_wine
if [ "$install_wine" = "y" ]; then
    sudo dpkg --add-architecture i386
    wget -nc https://dl.winehq.org/wine-builds/winehq.key
    sudo apt-key add winehq.key
    sudo add-apt-repository 'deb https://dl.winehq.org/wine-builds/ubuntu/ focal main'
    sudo apt update
    sudo apt install -y wine-stable
fi

# Setup project directory
echo "4. Setting up project directory..."
cd /home/$USER
git clone https://github.com/YOUR_USERNAME/forex-ml-trading.git
cd forex-ml-trading

# Create Python virtual environment
echo "5. Creating Python virtual environment..."
python3.9 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "6. Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "7. Creating necessary directories..."
mkdir -p logs
mkdir -p data/{raw,processed,models}

# Setup environment variables
echo "8. Setting up environment variables..."
cp .env.example .env
echo "Please edit the .env file with your credentials:"
echo "nano .env"
read -p "Press enter when you've updated the .env file..."

# Setup firewall
echo "9. Configuring firewall..."
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 8501  # Streamlit dashboard port
sudo ufw --force enable

# Create systemd service
echo "10. Creating systemd service..."
sudo tee /etc/systemd/system/forex-trading-bot.service > /dev/null <<EOF
[Unit]
Description=Forex ML Trading Bot
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/forex-ml-trading
Environment="PATH=/home/$USER/forex-ml-trading/venv/bin"
ExecStart=/home/$USER/forex-ml-trading/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Create monitoring service
sudo tee /etc/systemd/system/forex-monitoring.service > /dev/null <<EOF
[Unit]
Description=Forex Trading Monitor
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/forex-ml-trading
Environment="PATH=/home/$USER/forex-ml-trading/venv/bin"
ExecStart=/home/$USER/forex-ml-trading/venv/bin/python -m src.monitor
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable services
echo "11. Enabling services..."
sudo systemctl daemon-reload
sudo systemctl enable forex-trading-bot.service
sudo systemctl enable forex-monitoring.service

# Setup log rotation
echo "12. Setting up log rotation..."
sudo tee /etc/logrotate.d/forex-trading > /dev/null <<EOF
/home/$USER/forex-ml-trading/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $USER $USER
}
EOF

# Setup backup script
echo "13. Creating backup script..."
tee /home/$USER/forex-ml-trading/backup.sh > /dev/null <<'EOF'
#!/bin/bash
# Backup script for Forex Trading Bot

BACKUP_DIR="/home/$USER/backups"
DATE=$(date +%Y%m%d_%H%M%S)
PROJECT_DIR="/home/$USER/forex-ml-trading"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup database
cp $PROJECT_DIR/data/trading.db $BACKUP_DIR/trading_db_$DATE.db

# Backup models
tar -czf $BACKUP_DIR/models_$DATE.tar.gz $PROJECT_DIR/data/models/

# Backup logs
tar -czf $BACKUP_DIR/logs_$DATE.tar.gz $PROJECT_DIR/logs/

# Keep only last 7 days of backups
find $BACKUP_DIR -name "*.db" -mtime +7 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $DATE"
EOF

chmod +x /home/$USER/forex-ml-trading/backup.sh

# Add backup to crontab
echo "14. Setting up automated backups..."
(crontab -l 2>/dev/null; echo "0 2 * * * /home/$USER/forex-ml-trading/backup.sh") | crontab -

# Setup monitoring dashboard (optional)
echo "15. Setting up monitoring dashboard..."
read -p "Do you want to setup the web dashboard? (y/n): " setup_dashboard
if [ "$setup_dashboard" = "y" ]; then
    # Create dashboard service
    sudo tee /etc/systemd/system/forex-dashboard.service > /dev/null <<EOF
[Unit]
Description=Forex Trading Dashboard
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=/home/$USER/forex-ml-trading
Environment="PATH=/home/$USER/forex-ml-trading/venv/bin"
ExecStart=/home/$USER/forex-ml-trading/venv/bin/streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

    sudo systemctl enable forex-dashboard.service
    
    # Setup Nginx reverse proxy
    sudo tee /etc/nginx/sites-available/forex-dashboard > /dev/null <<EOF
server {
    listen 80;
    server_name your_domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF

    sudo ln -s /etc/nginx/sites-available/forex-dashboard /etc/nginx/sites-enabled/
    sudo nginx -t
    sudo systemctl restart nginx
fi

echo "==================================="
echo "Setup completed!"
echo "==================================="
echo ""
echo "Next steps:"
echo "1. Edit the .env file with your MT5 and Telegram credentials"
echo "2. Run initial data collection: python -m src.data_collector"
echo "3. Train models: python train_models.py"
echo "4. Start services:"
echo "   sudo systemctl start forex-trading-bot"
echo "   sudo systemctl start forex-monitoring"
echo "5. Check logs: tail -f logs/*.log"
echo ""
echo "Useful commands:"
echo "- Check service status: sudo systemctl status forex-trading-bot"
echo "- View logs: journalctl -u forex-trading-bot -f"
echo "- Stop service: sudo systemctl stop forex-trading-bot"
echo "- Restart service: sudo systemctl restart forex-trading-bot"