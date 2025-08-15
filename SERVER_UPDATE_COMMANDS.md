# 🚀 أوامر تحديث النظام على السيرفر

## ⚠️ تحذير مهم
قبل البدء، تأكد من:
1. إيقاف جميع العمليات الجارية
2. عمل نسخة احتياطية كاملة
3. التأكد من وجود مساحة كافية

## 📝 الخطوة 1: الاتصال بالسيرفر وإيقاف العمليات

```bash
# الاتصال بالسيرفر
ssh username@69.62.121.53

# إيقاف العمليات الجارية
tmux ls
tmux kill-session -t learning
tmux kill-session -t server
pkill -f advanced_learner_unified.py
pkill -f continuous_learner_unified.py
pkill -f mt5_bridge_server_advanced.py
```

## 📦 الخطوة 2: إنشاء نسخة احتياطية

```bash
cd ~/forex-ml-trading
mkdir -p backups/pre_sr_update_$(date +%Y%m%d_%H%M%S)

# نسخ احتياطي للملفات المهمة
cp -r src backups/pre_sr_update_*/
cp -r models backups/pre_sr_update_*/
cp -r config backups/pre_sr_update_*/
cp *.py backups/pre_sr_update_*/
cp -r trading_data.db backups/pre_sr_update_*/

# ضغط النسخة الاحتياطية
cd backups
tar -czf pre_sr_update_$(date +%Y%m%d_%H%M%S).tar.gz pre_sr_update_*/
cd ..
```

## 🔄 الخطوة 3: تحديث الملفات من GitHub

```bash
# سحب آخر التحديثات
git pull origin main

# أو إذا كان لديك تغييرات محلية
git stash
git pull origin main
git stash pop
```

## 📥 الخطوة 4: رفع الملفات الجديدة (إذا لزم الأمر)

إذا كانت الملفات الجديدة غير موجودة في GitHub، ارفعها يدوياً:

```bash
# من جهازك المحلي
scp feature_engineer_adaptive_75.py username@69.62.121.53:~/forex-ml-trading/
scp support_resistance.py username@69.62.121.53:~/forex-ml-trading/
scp dynamic_sl_tp_system.py username@69.62.121.53:~/forex-ml-trading/
scp instrument_manager.py username@69.62.121.53:~/forex-ml-trading/
scp update_learning_system.py username@69.62.121.53:~/forex-ml-trading/
scp complete_system_update.py username@69.62.121.53:~/forex-ml-trading/
scp ForexMLBot_MultiTF_SR.mq5 username@69.62.121.53:~/forex-ml-trading/
```

## 🛠️ الخطوة 5: تشغيل التحديث الشامل

```bash
# تثبيت المتطلبات الجديدة (إن وجدت)
pip install scipy loguru

# تشغيل التحديث
python3 complete_system_update.py

# أو التحديث اليدوي
python3 update_learning_system.py
```

## 🔧 الخطوة 6: تحديث ملفات الخادم يدوياً (إذا لزم)

### تحديث src/mt5_bridge_server_advanced.py:

```bash
nano src/mt5_bridge_server_advanced.py

# غيّر السطر:
from feature_engineer_adaptive_70 import AdaptiveFeatureEngineer
# إلى:
from feature_engineer_adaptive_75 import AdaptiveFeatureEngineer75

# غيّر السطر:
self.feature_engineer = AdaptiveFeatureEngineer(target_features=70)
# إلى:
self.feature_engineer = AdaptiveFeatureEngineer75(target_features=75)

# في دالة process_prediction، غيّر:
df_features = self.feature_engineer.engineer_features(df)
# إلى:
symbol = data.get('symbol', 'UNKNOWN')
df_features = self.feature_engineer.engineer_features(df, symbol)
```

### تحديث src/advanced_learner_unified.py:

```bash
nano src/advanced_learner_unified.py

# نفس التغييرات السابقة
# بالإضافة إلى تغيير:
expected_features = 70
# إلى:
expected_features = 75
```

## 🏃 الخطوة 7: إعادة تدريب النماذج

```bash
# إنشاء مجلد للنماذج الجديدة
mkdir -p models/unified_75

# تدريب نماذج الأزواج الأساسية أولاً
tmux new -s training_phase1
python3 train_all_pairs_75.py --pairs "EURUSD,GBPUSD,USDJPY,AUDUSD,NZDUSD,USDCAD,USDCHF,XAUUSD"

# في جلسة tmux أخرى لباقي الأزواج
tmux new -s training_phase2
python3 train_all_pairs_75.py --pairs "EURJPY,GBPJPY,EURGBP,XAGUSD,USOIL,US30,NAS100"
```

## 📋 الخطوة 8: إنشاء سكريبت التدريب للأزواج الجديدة

```bash
cat > train_all_pairs_75.py << 'EOF'
#!/usr/bin/env python3
import sys
import argparse
from advanced_learner_unified import AdvancedLearner

def train_pairs(pairs_list):
    """تدريب قائمة من الأزواج"""
    learner = AdvancedLearner()
    
    for pair in pairs_list:
        print(f"\n{'='*60}")
        print(f"Training models for {pair}")
        print('='*60)
        
        for timeframe in ['M5', 'M15', 'H1', 'H4']:
            try:
                print(f"\nTraining {pair} {timeframe}...")
                learner.train_model(pair, timeframe)
                print(f"✅ Completed {pair} {timeframe}")
            except Exception as e:
                print(f"❌ Failed {pair} {timeframe}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', type=str, required=True, help='Comma-separated pairs')
    args = parser.parse_args()
    
    pairs = [p.strip() for p in args.pairs.split(',')]
    train_pairs(pairs)
EOF

chmod +x train_all_pairs_75.py
```

## ✅ الخطوة 9: التحقق من النظام

```bash
# اختبار النماذج الجديدة
python3 << 'EOF'
import joblib
import os

models_dir = "models/unified"
for file in os.listdir(models_dir):
    if file.endswith('.pkl'):
        model_data = joblib.load(os.path.join(models_dir, file))
        n_features = model_data.get('n_features', 0)
        print(f"{file}: {n_features} features")
EOF

# اختبار الخادم
python3 src/mt5_bridge_server_advanced.py --test
```

## 🚀 الخطوة 10: إعادة تشغيل الخدمات

```bash
# تشغيل الخادم
tmux new -s server
cd ~/forex-ml-trading
python3 src/mt5_bridge_server_advanced.py

# Ctrl+B ثم D للخروج من tmux

# تشغيل التعلم المستمر
tmux new -s learning
cd ~/forex-ml-trading
python3 src/continuous_learner_unified.py

# Ctrl+B ثم D للخروج من tmux

# التحقق من العمليات
tmux ls
ps aux | grep python
```

## 📊 الخطوة 11: مراقبة النظام

```bash
# مشاهدة السجلات
tail -f logs/server_*.log
tail -f logs/learning_*.log

# مراقبة استخدام الموارد
htop

# التحقق من قاعدة البيانات
sqlite3 trading_data.db "SELECT COUNT(*) FROM trades;"
sqlite3 trading_data.db "SELECT COUNT(*) FROM signals;"
```

## 🔄 الخطوة 12: تحديث MetaTrader

1. انسخ ملف `ForexMLBot_MultiTF_SR.mq5` إلى مجلد Experts في MetaTrader
2. أعد تجميع الـ EA في MetaEditor
3. أضف EA للشارتات المطلوبة مع الإعدادات الجديدة

## ⚠️ في حالة حدوث مشاكل

```bash
# العودة للنسخة السابقة
cd ~/forex-ml-trading
rm -rf src models config *.py
tar -xzf backups/pre_sr_update_*.tar.gz -C .
mv backups/pre_sr_update_*/* .

# إعادة تشغيل الخدمات القديمة
tmux new -s server
python3 src/mt5_bridge_server_advanced.py
```

## 📝 ملاحظات مهمة

1. **الذاكرة**: النظام الجديد قد يحتاج ذاكرة أكثر (75 ميزة بدلاً من 70)
2. **وقت التدريب**: تدريب جميع الأزواج قد يستغرق 6-12 ساعة
3. **التوافق**: تأكد من أن جميع النماذج تستخدم نفس عدد الميزات
4. **الأداء**: راقب أداء النظام في الأيام الأولى

## 🎯 الأزواج الموصى بها للبداية

### المرحلة 1 (فورية):
- EURUSD, GBPUSD, USDJPY
- AUDUSD, NZDUSD, USDCAD, USDCHF
- XAUUSD, XAGUSD
- USOIL
- US30, NAS100

### المرحلة 2 (بعد أسبوع):
- EURJPY, GBPJPY, EURGBP
- SP500, DAX
- BTCUSD, ETHUSD

### المرحلة 3 (اختيارية):
- باقي الأزواج حسب الأداء والطلب

---

✅ **بعد إكمال جميع الخطوات، النظام جاهز للعمل بالميزات الجديدة!**