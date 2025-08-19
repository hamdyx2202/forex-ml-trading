#!/bin/bash
# 🔄 استرجاع سريع لآخر نسخة احتياطية

echo "🔄 Quick Restore - استرجاع سريع"
echo "================================"

# البحث عن آخر نسخة احتياطية
LATEST_BACKUP=$(ls -t ./backups/*.db* 2>/dev/null | head -1)

if [ -z "$LATEST_BACKUP" ]; then
    echo "❌ No backups found!"
    exit 1
fi

echo "📦 Found latest backup: $LATEST_BACKUP"
echo "📅 Date: $(stat -c %y "$LATEST_BACKUP" | cut -d' ' -f1,2)"
echo ""
echo "⚠️  This will restore the database from this backup!"
echo "Press Enter to continue or Ctrl+C to cancel..."
read

python3 restore_backup.py "$LATEST_BACKUP"