#!/usr/bin/env python3
"""
🧠 Ultimate Continuous Learning System - نظام التعلم المستمر المتقدم
📈 يتعلم من كل صفقة ويحسن الأداء تلقائياً
🔄 يدعم جميع العملات والاستراتيجيات
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sqlite3
from loguru import logger
import schedule
import time
from typing import Dict, List, Optional, Tuple
import os
from pathlib import Path
import joblib
from sklearn.preprocessing import RobustScaler
import threading
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# للتحليل المتقدم
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import talib

class UltimateContinuousLearner:
    """نظام التعلم المستمر المتقدم - يتعلم ويتحسن باستمرار"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """تهيئة نظام التعلم المستمر"""
        # تحميل الإعدادات
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            self.config = self._get_default_config()
        
        self.db_path = self.config.get("database", {}).get("path", "data/forex_data.db")
        
        # إعداد التسجيل
        logger.add("logs/continuous_learning_ultimate.log", rotation="1 day", retention="30 days")
        
        # إنشاء قواعد البيانات
        self._init_learning_database()
        
        # ذاكرة التعلم المتقدمة
        self.learning_memory = {
            'pattern_performance': defaultdict(lambda: {'success': 0, 'fail': 0, 'confidence': 0.5}),
            'market_conditions': defaultdict(list),
            'time_patterns': defaultdict(lambda: {'success': 0, 'total': 0}),
            'strategy_performance': defaultdict(lambda: {'trades': 0, 'wins': 0, 'total_pips': 0}),
            'symbol_characteristics': defaultdict(dict),
            'adaptive_parameters': defaultdict(dict),
            'hypothesis_tracker': defaultdict(list),
            'risk_adjustments': defaultdict(float)
        }
        
        # معلمات التعلم
        self.learning_rate = 0.01
        self.adaptation_threshold = 0.1
        self.memory_size = 10000
        self.update_frequency = 100  # كل 100 صفقة
        
        # تحميل الذاكرة السابقة
        self._load_learning_memory()
        
        # بدء خيط التعلم المستمر
        self.learning_thread = threading.Thread(target=self._continuous_learning_loop, daemon=True)
        self.learning_thread.start()
        
        logger.info("✅ تم تشغيل نظام التعلم المستمر المتقدم")

    def _get_default_config(self):
        """الإعدادات الافتراضية"""
        return {
            "database": {"path": "data/forex_data.db"},
            "learning": {
                "min_trades_for_update": 50,
                "confidence_decay": 0.995,
                "max_memory_days": 90
            }
        }

    def _init_learning_database(self):
        """إنشاء قواعد بيانات التعلم المستمر"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جدول التعلم من الصفقات
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS continuous_learning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                
                -- تفاصيل الصفقة
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                entry_price REAL,
                exit_price REAL,
                direction TEXT,
                lot_size REAL,
                
                -- النتائج
                result TEXT,
                pnl_pips REAL,
                pnl_money REAL,
                
                -- التحليل
                pattern_detected TEXT,
                indicators_snapshot TEXT,
                market_context TEXT,
                confidence_before REAL,
                confidence_after REAL,
                
                -- الدروس المستفادة
                lessons_learned TEXT,
                improvements_suggested TEXT,
                hypothesis_tested TEXT,
                hypothesis_result TEXT,
                
                -- المخاطر
                risk_score REAL,
                max_drawdown REAL,
                recovery_time INTEGER,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # جدول أداء الأنماط
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_performance (
                pattern_key TEXT PRIMARY KEY,
                symbol TEXT,
                timeframe TEXT,
                
                -- الإحصائيات
                total_occurrences INTEGER DEFAULT 0,
                successful_trades INTEGER DEFAULT 0,
                failed_trades INTEGER DEFAULT 0,
                
                -- الأداء
                total_pips REAL DEFAULT 0,
                avg_pips_per_trade REAL DEFAULT 0,
                max_win_pips REAL DEFAULT 0,
                max_loss_pips REAL DEFAULT 0,
                
                -- الثقة والجودة
                success_rate REAL DEFAULT 0,
                confidence_score REAL DEFAULT 0.5,
                quality_score REAL DEFAULT 0.5,
                
                -- التوقيت
                best_hours TEXT,
                best_days TEXT,
                best_sessions TEXT,
                
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # جدول تكيف الاستراتيجيات
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS strategy_adaptations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                strategy TEXT NOT NULL,
                
                -- التعديلات
                parameter_name TEXT,
                old_value REAL,
                new_value REAL,
                reason TEXT,
                
                -- الأداء
                performance_before REAL,
                performance_after REAL,
                improvement REAL,
                
                -- التطبيق
                applied BOOLEAN DEFAULT FALSE,
                applied_date TIMESTAMP,
                rollback_date TIMESTAMP,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # جدول الفرضيات
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS hypothesis_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id TEXT UNIQUE,
                symbol TEXT,
                timeframe TEXT,
                
                -- الفرضية
                hypothesis_text TEXT,
                hypothesis_type TEXT,
                confidence_level REAL,
                
                -- الاختبار
                tests_conducted INTEGER DEFAULT 0,
                tests_successful INTEGER DEFAULT 0,
                tests_failed INTEGER DEFAULT 0,
                
                -- النتائج
                avg_performance REAL,
                statistical_significance REAL,
                conclusion TEXT,
                
                -- الحالة
                status TEXT DEFAULT 'testing',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                concluded_at TIMESTAMP
            )
        """)
        
        # جدول تحليل السوق
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS market_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                analysis_time TIMESTAMP,
                
                -- حالة السوق
                market_phase TEXT,
                trend_strength REAL,
                volatility_level REAL,
                volume_profile TEXT,
                
                -- المستويات
                support_levels TEXT,
                resistance_levels TEXT,
                fibonacci_levels TEXT,
                pivot_points TEXT,
                
                -- التوقعات
                predicted_direction TEXT,
                prediction_confidence REAL,
                target_levels TEXT,
                risk_levels TEXT,
                
                -- الأحداث
                upcoming_events TEXT,
                market_sentiment TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()

    def learn_from_trade(self, trade_result: Dict) -> Dict:
        """التعلم من نتيجة صفقة مع تحليل متقدم"""
        try:
            symbol = trade_result.get('symbol', '')
            timeframe = trade_result.get('timeframe', '')
            strategy = trade_result.get('strategy', '')
            pnl_pips = trade_result.get('pnl_pips', 0)
            
            logger.info(f"📚 التعلم من صفقة: {symbol} - {strategy} - {pnl_pips:.1f} pips")
            
            # تحليل نجاح أو فشل الصفقة
            success = pnl_pips > 0
            
            # استخراج الأنماط والمؤشرات
            patterns = self._extract_patterns(trade_result)
            market_context = self._analyze_market_context(trade_result)
            time_analysis = self._analyze_time_patterns(trade_result)
            
            # تحديث ذاكرة التعلم
            self._update_pattern_performance(patterns, success, pnl_pips)
            self._update_strategy_performance(strategy, symbol, success, pnl_pips)
            self._update_time_patterns(time_analysis, success)
            
            # تحليل الفرضيات
            hypothesis_results = self._test_hypothesis(trade_result)
            
            # اقتراحات التحسين
            improvements = self._generate_improvements(trade_result, patterns, market_context)
            
            # حساب تعديل الثقة
            confidence_adjustment = self._calculate_confidence_adjustment(
                success, pnl_pips, patterns, market_context
            )
            
            # حفظ في قاعدة البيانات
            self._save_learning_record(
                trade_result, patterns, market_context, 
                improvements, hypothesis_results, confidence_adjustment
            )
            
            # التحقق من الحاجة لتحديث النماذج
            if self._should_update_models(symbol, strategy):
                self._trigger_model_update(symbol, timeframe, strategy)
            
            # إرجاع التوصيات
            return {
                'learned': True,
                'confidence_adjustment': confidence_adjustment,
                'improvements': improvements,
                'hypothesis_results': hypothesis_results,
                'next_trade_suggestions': self._get_next_trade_suggestions(symbol, strategy)
            }
            
        except Exception as e:
            logger.error(f"❌ خطأ في التعلم من الصفقة: {e}")
            return {'learned': False, 'error': str(e)}

    def _extract_patterns(self, trade_result: Dict) -> Dict:
        """استخراج الأنماط من بيانات الصفقة"""
        patterns = {
            'technical_patterns': [],
            'candlestick_patterns': [],
            'indicator_signals': {},
            'price_action': {}
        }
        
        # استخراج المؤشرات الفنية
        indicators = trade_result.get('indicators', {})
        
        # تحليل RSI
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            patterns['indicator_signals']['rsi'] = 'oversold'
        elif rsi > 70:
            patterns['indicator_signals']['rsi'] = 'overbought'
        else:
            patterns['indicator_signals']['rsi'] = 'neutral'
        
        # تحليل MACD
        macd_hist = indicators.get('macd_hist', 0)
        patterns['indicator_signals']['macd'] = 'bullish' if macd_hist > 0 else 'bearish'
        
        # تحليل Moving Averages
        ma_analysis = self._analyze_moving_averages(indicators)
        patterns['indicator_signals']['ma_trend'] = ma_analysis
        
        # أنماط الشموع
        candle_patterns = trade_result.get('candle_patterns', [])
        patterns['candlestick_patterns'] = candle_patterns
        
        # Price Action
        patterns['price_action'] = {
            'trend': self._identify_trend(trade_result),
            'support_resistance': self._find_sr_levels(trade_result),
            'volatility': indicators.get('atr', 0)
        }
        
        return patterns

    def _analyze_market_context(self, trade_result: Dict) -> Dict:
        """تحليل سياق السوق"""
        context = {
            'market_phase': 'unknown',
            'volatility': 'normal',
            'volume': 'average',
            'session': 'unknown',
            'trend_strength': 0
        }
        
        # تحديد المرحلة السوقية
        indicators = trade_result.get('indicators', {})
        
        # تحليل التقلب
        atr = indicators.get('atr', 0)
        atr_avg = indicators.get('atr_avg', 1)
        if atr > atr_avg * 1.5:
            context['volatility'] = 'high'
        elif atr < atr_avg * 0.5:
            context['volatility'] = 'low'
        
        # تحليل الحجم
        volume = trade_result.get('volume', 0)
        volume_avg = trade_result.get('volume_avg', 1)
        if volume > volume_avg * 1.5:
            context['volume'] = 'high'
        elif volume < volume_avg * 0.5:
            context['volume'] = 'low'
        
        # تحديد الجلسة
        hour = trade_result.get('hour', 0)
        context['session'] = self._get_trading_session(hour)
        
        # قوة الاتجاه
        adx = indicators.get('adx', 0)
        if adx > 40:
            context['trend_strength'] = 'strong'
        elif adx > 25:
            context['trend_strength'] = 'moderate'
        else:
            context['trend_strength'] = 'weak'
        
        return context

    def _analyze_time_patterns(self, trade_result: Dict) -> Dict:
        """تحليل الأنماط الزمنية"""
        entry_time = pd.to_datetime(trade_result.get('entry_time'))
        
        return {
            'hour': entry_time.hour,
            'day_of_week': entry_time.dayofweek,
            'day_of_month': entry_time.day,
            'month': entry_time.month,
            'session': self._get_trading_session(entry_time.hour),
            'is_news_time': self._is_news_time(entry_time),
            'is_market_open': self._is_major_market_open(entry_time)
        }

    def _update_pattern_performance(self, patterns: Dict, success: bool, pnl_pips: float):
        """تحديث أداء الأنماط"""
        # تحديث أداء كل نمط
        for pattern_type, pattern_data in patterns.items():
            if isinstance(pattern_data, dict):
                for key, value in pattern_data.items():
                    pattern_key = f"{pattern_type}_{key}_{value}"
                    
                    if success:
                        self.learning_memory['pattern_performance'][pattern_key]['success'] += 1
                    else:
                        self.learning_memory['pattern_performance'][pattern_key]['fail'] += 1
                    
                    # تحديث الثقة
                    total = (self.learning_memory['pattern_performance'][pattern_key]['success'] + 
                            self.learning_memory['pattern_performance'][pattern_key]['fail'])
                    
                    if total > 0:
                        success_rate = self.learning_memory['pattern_performance'][pattern_key]['success'] / total
                        self.learning_memory['pattern_performance'][pattern_key]['confidence'] = success_rate

    def _update_strategy_performance(self, strategy: str, symbol: str, success: bool, pnl_pips: float):
        """تحديث أداء الاستراتيجية"""
        key = f"{symbol}_{strategy}"
        
        self.learning_memory['strategy_performance'][key]['trades'] += 1
        if success:
            self.learning_memory['strategy_performance'][key]['wins'] += 1
        self.learning_memory['strategy_performance'][key]['total_pips'] += pnl_pips

    def _update_time_patterns(self, time_analysis: Dict, success: bool):
        """تحديث أنماط التوقيت"""
        hour = time_analysis['hour']
        day = time_analysis['day_of_week']
        session = time_analysis['session']
        
        # تحديث أداء الساعات
        hour_key = f"hour_{hour}"
        self.learning_memory['time_patterns'][hour_key]['total'] += 1
        if success:
            self.learning_memory['time_patterns'][hour_key]['success'] += 1
        
        # تحديث أداء الأيام
        day_key = f"day_{day}"
        self.learning_memory['time_patterns'][day_key]['total'] += 1
        if success:
            self.learning_memory['time_patterns'][day_key]['success'] += 1
        
        # تحديث أداء الجلسات
        session_key = f"session_{session}"
        self.learning_memory['time_patterns'][session_key]['total'] += 1
        if success:
            self.learning_memory['time_patterns'][session_key]['success'] += 1

    def _test_hypothesis(self, trade_result: Dict) -> Dict:
        """اختبار الفرضيات"""
        hypothesis_results = {}
        
        # اختبار فرضيات محددة مسبقاً
        active_hypotheses = self._get_active_hypotheses(
            trade_result.get('symbol'),
            trade_result.get('strategy')
        )
        
        for hypothesis in active_hypotheses:
            result = self._evaluate_hypothesis(hypothesis, trade_result)
            hypothesis_results[hypothesis['id']] = result
            
            # تحديث تتبع الفرضية
            self._update_hypothesis_tracking(hypothesis['id'], result)
        
        return hypothesis_results

    def _generate_improvements(self, trade_result: Dict, patterns: Dict, market_context: Dict) -> List[Dict]:
        """توليد اقتراحات التحسين"""
        improvements = []
        
        # تحليل نقاط الضعف
        if trade_result.get('pnl_pips', 0) < 0:
            # تحليل سبب الخسارة
            loss_reasons = self._analyze_loss_reasons(trade_result, patterns, market_context)
            
            for reason in loss_reasons:
                improvement = {
                    'type': reason['type'],
                    'description': reason['description'],
                    'action': reason['suggested_action'],
                    'priority': reason['priority']
                }
                improvements.append(improvement)
        
        # اقتراحات عامة للتحسين
        general_improvements = self._get_general_improvements(patterns, market_context)
        improvements.extend(general_improvements)
        
        # ترتيب حسب الأولوية
        improvements.sort(key=lambda x: x['priority'], reverse=True)
        
        return improvements[:5]  # أهم 5 تحسينات

    def _calculate_confidence_adjustment(self, success: bool, pnl_pips: float, 
                                       patterns: Dict, market_context: Dict) -> float:
        """حساب تعديل الثقة"""
        base_adjustment = 0.0
        
        # تعديل أساسي بناءً على النتيجة
        if success:
            base_adjustment = min(0.1, abs(pnl_pips) / 100)
        else:
            base_adjustment = -min(0.1, abs(pnl_pips) / 100)
        
        # تعديل بناءً على جودة الأنماط
        pattern_quality = self._evaluate_pattern_quality(patterns)
        base_adjustment *= pattern_quality
        
        # تعديل بناءً على سياق السوق
        if market_context.get('volatility') == 'high':
            base_adjustment *= 0.7  # تقليل الثقة في الأسواق المتقلبة
        
        return np.clip(base_adjustment, -0.2, 0.2)

    def _save_learning_record(self, trade_result: Dict, patterns: Dict, 
                            market_context: Dict, improvements: List, 
                            hypothesis_results: Dict, confidence_adjustment: float):
        """حفظ سجل التعلم في قاعدة البيانات"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO continuous_learning (
                    trade_id, symbol, timeframe, strategy,
                    entry_time, exit_time, entry_price, exit_price,
                    direction, lot_size, result, pnl_pips, pnl_money,
                    pattern_detected, indicators_snapshot, market_context,
                    confidence_before, confidence_after,
                    lessons_learned, improvements_suggested,
                    hypothesis_tested, hypothesis_result,
                    risk_score, max_drawdown, recovery_time
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_result.get('trade_id'),
                trade_result.get('symbol'),
                trade_result.get('timeframe'),
                trade_result.get('strategy'),
                trade_result.get('entry_time'),
                trade_result.get('exit_time'),
                trade_result.get('entry_price'),
                trade_result.get('exit_price'),
                trade_result.get('direction'),
                trade_result.get('lot_size'),
                'win' if trade_result.get('pnl_pips', 0) > 0 else 'loss',
                trade_result.get('pnl_pips'),
                trade_result.get('pnl_money'),
                json.dumps(patterns),
                json.dumps(trade_result.get('indicators', {})),
                json.dumps(market_context),
                trade_result.get('confidence', 0.5),
                trade_result.get('confidence', 0.5) + confidence_adjustment,
                json.dumps(self._extract_lessons(trade_result, patterns)),
                json.dumps(improvements),
                json.dumps(list(hypothesis_results.keys())),
                json.dumps(hypothesis_results),
                trade_result.get('risk_score', 0),
                trade_result.get('max_drawdown', 0),
                trade_result.get('recovery_time', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ خطأ في حفظ سجل التعلم: {e}")

    def _should_update_models(self, symbol: str, strategy: str) -> bool:
        """التحقق من الحاجة لتحديث النماذج"""
        key = f"{symbol}_{strategy}"
        trades_count = self.learning_memory['strategy_performance'][key]['trades']
        
        # تحديث كل 100 صفقة أو عند انخفاض الأداء
        if trades_count % self.update_frequency == 0:
            return True
        
        # تحقق من انخفاض الأداء
        if trades_count > 20:
            win_rate = self.learning_memory['strategy_performance'][key]['wins'] / trades_count
            if win_rate < 0.45:  # أقل من 45% نجاح
                return True
        
        return False

    def _trigger_model_update(self, symbol: str, timeframe: str, strategy: str):
        """تشغيل تحديث النموذج"""
        logger.info(f"🔄 تشغيل تحديث النموذج: {symbol} - {timeframe} - {strategy}")
        
        # إنشاء مهمة تحديث
        update_task = {
            'symbol': symbol,
            'timeframe': timeframe,
            'strategy': strategy,
            'reason': 'continuous_learning',
            'timestamp': datetime.now()
        }
        
        # حفظ المهمة في قائمة انتظار التحديثات
        self._queue_model_update(update_task)

    def _get_next_trade_suggestions(self, symbol: str, strategy: str) -> Dict:
        """الحصول على اقتراحات للصفقة التالية"""
        suggestions = {
            'best_hours': [],
            'avoid_hours': [],
            'recommended_patterns': [],
            'risk_adjustments': {},
            'confidence_modifiers': {}
        }
        
        # تحليل أفضل الأوقات
        time_performance = self._analyze_time_performance()
        suggestions['best_hours'] = time_performance['best_hours'][:3]
        suggestions['avoid_hours'] = time_performance['worst_hours'][:3]
        
        # الأنماط الموصى بها
        pattern_performance = self._analyze_pattern_performance(symbol)
        suggestions['recommended_patterns'] = pattern_performance['best_patterns'][:5]
        
        # تعديلات المخاطر
        suggestions['risk_adjustments'] = self._calculate_risk_adjustments(symbol, strategy)
        
        return suggestions

    def _continuous_learning_loop(self):
        """حلقة التعلم المستمر في الخلفية"""
        while True:
            try:
                # تحديث دوري للتحليلات
                self._periodic_analysis()
                
                # تنظيف الذاكرة القديمة
                self._cleanup_old_memory()
                
                # حفظ الذاكرة
                self._save_learning_memory()
                
                # الانتظار قبل التحديث التالي
                time.sleep(3600)  # كل ساعة
                
            except Exception as e:
                logger.error(f"❌ خطأ في حلقة التعلم المستمر: {e}")
                time.sleep(60)

    def _periodic_analysis(self):
        """تحليل دوري للأداء والتحسينات"""
        logger.info("🔄 بدء التحليل الدوري...")
        
        # تحليل أداء جميع الاستراتيجيات
        for key, performance in self.learning_memory['strategy_performance'].items():
            if performance['trades'] > 50:
                win_rate = performance['wins'] / performance['trades']
                avg_pips = performance['total_pips'] / performance['trades']
                
                logger.info(f"📊 {key}: معدل النجاح {win_rate:.2%}, متوسط النقاط {avg_pips:.1f}")
                
                # توليد توصيات تحسين
                if win_rate < 0.5:
                    self._generate_strategy_improvements(key, performance)

    def _cleanup_old_memory(self):
        """تنظيف الذاكرة القديمة"""
        max_age_days = self.config.get('learning', {}).get('max_memory_days', 90)
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # حذف السجلات القديمة
            cursor.execute("""
                DELETE FROM continuous_learning 
                WHERE created_at < ?
            """, (cutoff_date,))
            
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                logger.info(f"🧹 تم حذف {deleted_count} سجل قديم")
                
        except Exception as e:
            logger.error(f"❌ خطأ في تنظيف الذاكرة: {e}")

    def _save_learning_memory(self):
        """حفظ ذاكرة التعلم"""
        try:
            memory_path = Path("data/learning_memory.json")
            memory_path.parent.mkdir(exist_ok=True)
            
            # تحويل defaultdict إلى dict عادي للحفظ
            memory_to_save = {
                key: dict(value) if isinstance(value, defaultdict) else value
                for key, value in self.learning_memory.items()
            }
            
            with open(memory_path, 'w') as f:
                json.dump(memory_to_save, f, indent=2, default=str)
                
            logger.info("💾 تم حفظ ذاكرة التعلم")
            
        except Exception as e:
            logger.error(f"❌ خطأ في حفظ الذاكرة: {e}")

    def _load_learning_memory(self):
        """تحميل ذاكرة التعلم السابقة"""
        try:
            memory_path = Path("data/learning_memory.json")
            
            if memory_path.exists():
                with open(memory_path, 'r') as f:
                    loaded_memory = json.load(f)
                
                # تحديث الذاكرة الحالية
                for key, value in loaded_memory.items():
                    if key in self.learning_memory:
                        if isinstance(self.learning_memory[key], defaultdict):
                            self.learning_memory[key].update(value)
                        else:
                            self.learning_memory[key] = value
                
                logger.info("✅ تم تحميل ذاكرة التعلم السابقة")
                
        except Exception as e:
            logger.error(f"❌ خطأ في تحميل الذاكرة: {e}")

    def generate_performance_report(self, symbol: Optional[str] = None, 
                                  strategy: Optional[str] = None) -> Dict:
        """توليد تقرير أداء شامل"""
        report = {
            'overall_performance': {},
            'pattern_analysis': {},
            'time_analysis': {},
            'improvements': [],
            'recommendations': []
        }
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # الأداء العام
            query = """
                SELECT 
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl_pips > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(pnl_pips) as total_pips,
                    AVG(pnl_pips) as avg_pips,
                    MAX(pnl_pips) as max_win,
                    MIN(pnl_pips) as max_loss
                FROM continuous_learning
            """
            
            params = []
            if symbol:
                query += " WHERE symbol = ?"
                params.append(symbol)
            if strategy:
                query += " AND strategy = ?" if symbol else " WHERE strategy = ?"
                params.append(strategy)
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                row = df.iloc[0]
                report['overall_performance'] = {
                    'total_trades': int(row['total_trades']),
                    'wins': int(row['wins']),
                    'win_rate': row['wins'] / row['total_trades'] if row['total_trades'] > 0 else 0,
                    'total_pips': float(row['total_pips']),
                    'avg_pips': float(row['avg_pips']),
                    'max_win': float(row['max_win']),
                    'max_loss': float(row['max_loss'])
                }
            
            # تحليل الأنماط
            report['pattern_analysis'] = self._generate_pattern_report()
            
            # تحليل الوقت
            report['time_analysis'] = self._generate_time_report()
            
            # التوصيات
            report['recommendations'] = self._generate_recommendations(report)
            
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ خطأ في توليد التقرير: {e}")
        
        return report

    # وظائف مساعدة
    def _get_trading_session(self, hour: int) -> str:
        """تحديد جلسة التداول"""
        if 0 <= hour < 8:
            return 'tokyo'
        elif 8 <= hour < 16:
            return 'london'
        elif 13 <= hour < 21:
            return 'newyork'
        else:
            return 'sydney'

    def _is_news_time(self, timestamp: pd.Timestamp) -> bool:
        """التحقق من وقت الأخبار"""
        # أوقات الأخبار الشائعة (UTC)
        news_hours = [8, 10, 12, 14, 18, 20]
        return timestamp.hour in news_hours and timestamp.minute < 30

    def _is_major_market_open(self, timestamp: pd.Timestamp) -> bool:
        """التحقق من فتح الأسواق الرئيسية"""
        hour = timestamp.hour
        return 8 <= hour <= 21  # London to NY close

    def _identify_trend(self, trade_result: Dict) -> str:
        """تحديد الاتجاه"""
        indicators = trade_result.get('indicators', {})
        
        sma_50 = indicators.get('sma_50', 0)
        sma_200 = indicators.get('sma_200', 0)
        
        if sma_50 > sma_200:
            return 'uptrend'
        elif sma_50 < sma_200:
            return 'downtrend'
        else:
            return 'sideways'

    def _find_sr_levels(self, trade_result: Dict) -> Dict:
        """إيجاد مستويات الدعم والمقاومة"""
        # تنفيذ مبسط
        return {
            'support': trade_result.get('support_level', 0),
            'resistance': trade_result.get('resistance_level', 0)
        }

    def _analyze_loss_reasons(self, trade_result: Dict, patterns: Dict, 
                            market_context: Dict) -> List[Dict]:
        """تحليل أسباب الخسارة"""
        reasons = []
        
        # تحقق من التوقيت السيء
        if market_context.get('volatility') == 'high':
            reasons.append({
                'type': 'high_volatility',
                'description': 'دخول في سوق عالي التقلب',
                'suggested_action': 'تجنب التداول عند ارتفاع ATR فوق المتوسط بـ 50%',
                'priority': 8
            })
        
        # تحقق من اتجاه خاطئ
        if patterns.get('price_action', {}).get('trend') != trade_result.get('direction'):
            reasons.append({
                'type': 'against_trend',
                'description': 'تداول عكس الاتجاه العام',
                'suggested_action': 'التداول مع الاتجاه فقط عند قوة ADX > 25',
                'priority': 9
            })
        
        return reasons

    def _evaluate_pattern_quality(self, patterns: Dict) -> float:
        """تقييم جودة الأنماط"""
        quality_score = 0.5
        
        # تقييم بناءً على عدد المؤشرات المتوافقة
        confirming_indicators = 0
        total_indicators = 0
        
        for indicator, signal in patterns.get('indicator_signals', {}).items():
            total_indicators += 1
            if signal in ['bullish', 'oversold'] and patterns.get('direction') == 'buy':
                confirming_indicators += 1
            elif signal in ['bearish', 'overbought'] and patterns.get('direction') == 'sell':
                confirming_indicators += 1
        
        if total_indicators > 0:
            quality_score = confirming_indicators / total_indicators
        
        return quality_score

    def _extract_lessons(self, trade_result: Dict, patterns: Dict) -> List[str]:
        """استخراج الدروس المستفادة"""
        lessons = []
        
        if trade_result.get('pnl_pips', 0) > 0:
            lessons.append(f"نجحت عند {patterns.get('price_action', {}).get('trend', 'unknown')} trend")
        else:
            lessons.append(f"فشلت في {trade_result.get('market_context', {}).get('session', 'unknown')} session")
        
        return lessons

    def _get_active_hypotheses(self, symbol: str, strategy: str) -> List[Dict]:
        """الحصول على الفرضيات النشطة"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT * FROM hypothesis_tracking
                WHERE (symbol = ? OR symbol IS NULL)
                AND status = 'testing'
                ORDER BY confidence_level DESC
                LIMIT 5
            """, (symbol,))
            
            hypotheses = []
            for row in cursor.fetchall():
                hypotheses.append({
                    'id': row[1],
                    'text': row[4],
                    'type': row[5],
                    'confidence': row[6]
                })
            
            conn.close()
            return hypotheses
            
        except Exception as e:
            logger.error(f"❌ خطأ في جلب الفرضيات: {e}")
            return []

    def _evaluate_hypothesis(self, hypothesis: Dict, trade_result: Dict) -> Dict:
        """تقييم فرضية"""
        # تنفيذ مبسط
        return {
            'hypothesis_id': hypothesis['id'],
            'result': 'confirmed' if trade_result.get('pnl_pips', 0) > 0 else 'rejected',
            'confidence_change': 0.1 if trade_result.get('pnl_pips', 0) > 0 else -0.1
        }

    def _update_hypothesis_tracking(self, hypothesis_id: str, result: Dict):
        """تحديث تتبع الفرضية"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if result['result'] == 'confirmed':
                cursor.execute("""
                    UPDATE hypothesis_tracking
                    SET tests_conducted = tests_conducted + 1,
                        tests_successful = tests_successful + 1,
                        confidence_level = confidence_level + ?
                    WHERE hypothesis_id = ?
                """, (result['confidence_change'], hypothesis_id))
            else:
                cursor.execute("""
                    UPDATE hypothesis_tracking
                    SET tests_conducted = tests_conducted + 1,
                        tests_failed = tests_failed + 1,
                        confidence_level = confidence_level + ?
                    WHERE hypothesis_id = ?
                """, (result['confidence_change'], hypothesis_id))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحديث الفرضية: {e}")

    def _queue_model_update(self, update_task: Dict):
        """إضافة مهمة تحديث النموذج لقائمة الانتظار"""
        # حفظ في ملف المهام المعلقة
        tasks_file = Path("data/pending_model_updates.json")
        
        try:
            if tasks_file.exists():
                with open(tasks_file, 'r') as f:
                    tasks = json.load(f)
            else:
                tasks = []
            
            tasks.append(update_task)
            
            with open(tasks_file, 'w') as f:
                json.dump(tasks, f, indent=2, default=str)
                
            logger.info(f"📝 تمت إضافة مهمة تحديث النموذج لقائمة الانتظار")
            
        except Exception as e:
            logger.error(f"❌ خطأ في حفظ مهمة التحديث: {e}")

    def _analyze_time_performance(self) -> Dict:
        """تحليل أداء الأوقات"""
        performance = {'best_hours': [], 'worst_hours': []}
        
        # حساب أداء كل ساعة
        hour_performance = []
        for hour in range(24):
            hour_key = f"hour_{hour}"
            data = self.learning_memory['time_patterns'][hour_key]
            
            if data['total'] > 10:  # عينة كافية
                success_rate = data['success'] / data['total']
                hour_performance.append((hour, success_rate))
        
        # ترتيب الساعات
        hour_performance.sort(key=lambda x: x[1], reverse=True)
        
        performance['best_hours'] = [h[0] for h in hour_performance[:5]]
        performance['worst_hours'] = [h[0] for h in hour_performance[-5:]]
        
        return performance

    def _analyze_pattern_performance(self, symbol: str) -> Dict:
        """تحليل أداء الأنماط"""
        performance = {'best_patterns': [], 'worst_patterns': []}
        
        # حساب أداء كل نمط
        pattern_scores = []
        for pattern_key, data in self.learning_memory['pattern_performance'].items():
            if symbol in pattern_key:
                total = data['success'] + data['fail']
                if total > 5:  # عينة كافية
                    success_rate = data['success'] / total
                    pattern_scores.append((pattern_key, success_rate, data['confidence']))
        
        # ترتيب الأنماط
        pattern_scores.sort(key=lambda x: x[2], reverse=True)
        
        performance['best_patterns'] = [p[0] for p in pattern_scores[:10]]
        performance['worst_patterns'] = [p[0] for p in pattern_scores[-5:]]
        
        return performance

    def _calculate_risk_adjustments(self, symbol: str, strategy: str) -> Dict:
        """حساب تعديلات المخاطر"""
        key = f"{symbol}_{strategy}"
        performance = self.learning_memory['strategy_performance'][key]
        
        adjustments = {
            'lot_size_modifier': 1.0,
            'sl_modifier': 1.0,
            'tp_modifier': 1.0,
            'confidence_threshold': 0.85
        }
        
        if performance['trades'] > 50:
            win_rate = performance['wins'] / performance['trades']
            
            # تعديل حجم الصفقة
            if win_rate > 0.6:
                adjustments['lot_size_modifier'] = 1.2
            elif win_rate < 0.4:
                adjustments['lot_size_modifier'] = 0.8
            
            # تعديل وقف الخسارة
            if win_rate < 0.45:
                adjustments['sl_modifier'] = 0.8  # وقف خسارة أقرب
            
            # تعديل الهدف
            avg_pips = performance['total_pips'] / performance['trades']
            if avg_pips < 5:
                adjustments['tp_modifier'] = 1.2  # أهداف أبعد
        
        return adjustments

    def _get_general_improvements(self, patterns: Dict, market_context: Dict) -> List[Dict]:
        """الحصول على تحسينات عامة"""
        improvements = []
        
        # تحسينات بناءً على السياق
        if market_context.get('volume') == 'low':
            improvements.append({
                'type': 'volume_filter',
                'description': 'تجنب التداول في أوقات الحجم المنخفض',
                'action': 'إضافة فلتر حجم > المتوسط × 0.8',
                'priority': 6
            })
        
        return improvements

    def _generate_strategy_improvements(self, key: str, performance: Dict):
        """توليد تحسينات للاستراتيجية"""
        logger.info(f"🔧 توليد تحسينات لـ {key}")
        
        # تحليل نقاط الضعف وتوليد حلول
        improvements = []
        
        win_rate = performance['wins'] / performance['trades']
        avg_pips = performance['total_pips'] / performance['trades']
        
        if win_rate < 0.5:
            improvements.append("زيادة معايير التأكيد قبل الدخول")
            improvements.append("تحسين توقيت الدخول بناءً على الجلسات")
        
        if avg_pips < 0:
            improvements.append("مراجعة مستويات وقف الخسارة")
            improvements.append("تحسين نسبة المخاطرة للعائد")
        
        # حفظ التحسينات
        self._save_strategy_improvements(key, improvements)

    def _save_strategy_improvements(self, key: str, improvements: List[str]):
        """حفظ تحسينات الاستراتيجية"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            parts = key.split('_')
            symbol = parts[0]
            strategy = '_'.join(parts[1:])
            
            for improvement in improvements:
                cursor.execute("""
                    INSERT INTO strategy_adaptations (
                        symbol, timeframe, strategy, parameter_name,
                        old_value, new_value, reason
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (symbol, 'H1', strategy, 'improvement', 0, 0, improvement))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ خطأ في حفظ التحسينات: {e}")

    def _generate_pattern_report(self) -> Dict:
        """توليد تقرير الأنماط"""
        report = {}
        
        # أفضل الأنماط
        best_patterns = []
        for pattern_key, data in self.learning_memory['pattern_performance'].items():
            total = data['success'] + data['fail']
            if total > 10:
                success_rate = data['success'] / total
                best_patterns.append({
                    'pattern': pattern_key,
                    'success_rate': success_rate,
                    'confidence': data['confidence'],
                    'occurrences': total
                })
        
        best_patterns.sort(key=lambda x: x['confidence'], reverse=True)
        report['top_patterns'] = best_patterns[:10]
        
        return report

    def _generate_time_report(self) -> Dict:
        """توليد تقرير الوقت"""
        report = {
            'best_hours': [],
            'best_days': [],
            'best_sessions': []
        }
        
        # أفضل الساعات
        hour_data = []
        for hour in range(24):
            hour_key = f"hour_{hour}"
            data = self.learning_memory['time_patterns'][hour_key]
            if data['total'] > 10:
                success_rate = data['success'] / data['total']
                hour_data.append({'hour': hour, 'success_rate': success_rate})
        
        hour_data.sort(key=lambda x: x['success_rate'], reverse=True)
        report['best_hours'] = hour_data[:5]
        
        return report

    def _generate_recommendations(self, report: Dict) -> List[str]:
        """توليد التوصيات النهائية"""
        recommendations = []
        
        # توصيات بناءً على الأداء
        if report['overall_performance'].get('win_rate', 0) < 0.5:
            recommendations.append("تحسين معايير دخول الصفقات لزيادة معدل النجاح")
        
        if report['overall_performance'].get('avg_pips', 0) < 5:
            recommendations.append("زيادة أهداف الربح أو تحسين نقاط الدخول")
        
        # توصيات الوقت
        if report['time_analysis'].get('best_hours'):
            best_hour = report['time_analysis']['best_hours'][0]['hour']
            recommendations.append(f"التركيز على التداول في الساعة {best_hour}:00")
        
        return recommendations


# مثال على الاستخدام
if __name__ == "__main__":
    # إنشاء نظام التعلم المستمر
    learner = UltimateContinuousLearner()
    
    # مثال على صفقة للتعلم منها
    sample_trade = {
        'trade_id': 'TRADE_001',
        'symbol': 'EURUSD',
        'timeframe': 'H1',
        'strategy': 'scalping',
        'entry_time': datetime.now() - timedelta(hours=2),
        'exit_time': datetime.now(),
        'entry_price': 1.0850,
        'exit_price': 1.0870,
        'direction': 'buy',
        'lot_size': 0.1,
        'pnl_pips': 20,
        'pnl_money': 200,
        'indicators': {
            'rsi': 65,
            'macd_hist': 0.0002,
            'atr': 0.0010,
            'adx': 35
        },
        'volume': 1500,
        'volume_avg': 1200,
        'hour': 14
    }
    
    # التعلم من الصفقة
    result = learner.learn_from_trade(sample_trade)
    logger.info(f"نتيجة التعلم: {result}")
    
    # توليد تقرير الأداء
    report = learner.generate_performance_report('EURUSD', 'scalping')
    logger.info(f"تقرير الأداء: {json.dumps(report, indent=2)}")