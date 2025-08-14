#!/bin/bash
# Install systemd services
# تثبيت خدمات systemd

echo "🚀 Installing Forex ML Trading systemd services..."

# التحقق من صلاحيات root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root (use sudo)"
    exit 1
fi

# نسخ ملفات الخدمة
echo "📁 Copying service files..."
cp forex-ml-*.service /etc/systemd/system/

# إعادة تحميل systemd
echo "🔄 Reloading systemd..."
systemctl daemon-reload

# تفعيل الخدمات
echo "✅ Enabling services..."
systemctl enable forex-ml-server.service
systemctl enable forex-ml-dashboard.service
systemctl enable forex-ml-monitor.service

# بدء الخدمات
echo "▶️ Starting services..."
systemctl start forex-ml-server.service
sleep 5
systemctl start forex-ml-dashboard.service
systemctl start forex-ml-monitor.service

# عرض الحالة
echo ""
echo "📊 Service Status:"
echo "=================="
systemctl status forex-ml-server.service --no-pager | grep -E "(Active:|Main PID:)"
systemctl status forex-ml-dashboard.service --no-pager | grep -E "(Active:|Main PID:)"
systemctl status forex-ml-monitor.service --no-pager | grep -E "(Active:|Main PID:)"

echo ""
echo "✅ Installation complete!"
echo ""
echo "📋 Useful commands:"
echo "  systemctl status forex-ml-server      # حالة السيرفر"
echo "  systemctl restart forex-ml-server     # إعادة تشغيل السيرفر"
echo "  systemctl stop forex-ml-server        # إيقاف السيرفر"
echo "  journalctl -u forex-ml-server -f      # عرض السجلات"
echo ""
echo "🌐 Dashboard URL: http://$(hostname -I | awk '{print $1}'):8080"