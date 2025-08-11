import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import sqlite3
from loguru import logger
import schedule
import time
from typing import Dict, List, Optional
import os


class ContinuousLearner:
    """نظام التعلم المستمر - يتعلم من كل صفقة ويحسن نفسه تلقائياً"""
    
    def __init__(self, config_path: str = "config/config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.db_path = self.config["database"]["path"]
        logger.add("logs/continuous_learning.log", rotation="1 day", retention="30 days")
        
        # إنشاء جداول التعلم المستمر
        self._init_learning_database()
        
        # ذاكرة التعلم
        self.learning_memory = {
            'successful_patterns': {},
            'failed_patterns': {},
            'market_conditions': {},
            'time_patterns': {},
            'improvement_suggestions': []
        }
        
        # تحميل التعلم السابق
        self._load_learning_memory()
    
    def _init_learning_database(self):
        """إنشاء قاعدة بيانات التعلم المستمر"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جدول التعلم من كل صفقة
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS continuous_learning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                
                -- نتيجة الصفقة
                entry_time TIMESTAMP,
                exit_time TIMESTAMP,
                direction TEXT,
                result TEXT,
                pnl_pips REAL,
                
                -- ما تعلمناه
                pattern_success TEXT,
                indicators_state TEXT,
                market_context TEXT,
                lessons_learned TEXT,
                confidence_adjustment REAL,
                
                -- التحسينات المقترحة
                improvements TEXT,
                
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # جدول أداء الأنماط المحدث
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_performance (
                pattern_key TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                successful_trades INTEGER DEFAULT 0,
                total_pips REAL DEFAULT 0,
                avg_pips REAL DEFAULT 0,
                success_rate REAL DEFAULT 0,
                confidence_score REAL DEFAULT 0.5,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def learn_from_trade(self, trade_result: Dict):
        """التعلم من نتيجة صفقة"""
        logger.info(f"Learning from trade: {trade_result.get('symbol')} - Result: {trade_result.get('pnl_pips', 0)} pips")
        
        # تحليل نجاح أو فشل الصفقة
        success = trade_result['pnl_pips'] > 0
        
        # استخراج الأنماط والظروف
        patterns = self._extract_patterns(trade_result)
        market_context = self._analyze_market_context(trade_result)
        
        # تحديث ذاكرة التعلم
        pattern_key = self._create_pattern_key(patterns)
        
        if success:
            if pattern_key not in self.learning_memory['successful_patterns']:
                self.learning_memory['successful_patterns'][pattern_key] = {
                    'count': 0,
                    'total_pips': 0,
                    'conditions': []
                }
            
            self.learning_memory['successful_patterns'][pattern_key]['count'] += 1
            self.learning_memory['successful_patterns'][pattern_key]['total_pips'] += trade_result['pnl_pips']
            self.learning_memory['successful_patterns'][pattern_key]['conditions'].append(market_context)
        else:
            if pattern_key not in self.learning_memory['failed_patterns']:
                self.learning_memory['failed_patterns'][pattern_key] = {
                    'count': 0,
                    'total_pips': 0,
                    'reasons': []
                }
            
            self.learning_memory['failed_patterns'][pattern_key]['count'] += 1
            self.learning_memory['failed_patterns'][pattern_key]['total_pips'] += trade_result['pnl_pips']
            
            # تحليل أسباب الفشل
            failure_reason = self._analyze_failure(trade_result)
            self.learning_memory['failed_patterns'][pattern_key]['reasons'].append(failure_reason)
        
        # حفظ التعلم في قاعدة البيانات
        self._save_learning(trade_result, patterns, market_context, success)
        
        # تحديث أداء الأنماط
        self._update_pattern_performance(pattern_key, success, trade_result['pnl_pips'])
        
        # توليد اقتراحات التحسين
        improvements = self._generate_improvements()
        if improvements:
            logger.info(f"New improvements suggested: {improvements}")
            self.learning_memory['improvement_suggestions'].extend(improvements)
        
        # حفظ الذاكرة
        self._save_learning_memory()
    
    def _extract_patterns(self, trade_result: Dict) -> Dict:
        """استخراج الأنماط من الصفقة"""
        patterns = {
            'candle_pattern': trade_result.get('candle_pattern', 'unknown'),
            'indicators': {
                'rsi_state': self._categorize_rsi(trade_result.get('rsi_value', 50)),
                'macd_state': trade_result.get('macd_signal', 'neutral'),
                'bb_position': self._categorize_bb(trade_result.get('bb_position', 0.5)),
                'trend_strength': self._categorize_trend(trade_result.get('adx_value', 0))
            },
            'volume_state': 'high' if trade_result.get('volume_ratio', 1) > 1.5 else 'normal',
            'time_pattern': {
                'hour': trade_result.get('hour', 0),
                'session': trade_result.get('trading_session', 'unknown')
            }
        }
        
        return patterns
    
    def _categorize_rsi(self, rsi: float) -> str:
        """تصنيف حالة RSI"""
        if rsi < 30:
            return 'oversold'
        elif rsi > 70:
            return 'overbought'
        elif 45 <= rsi <= 55:
            return 'neutral'
        elif rsi < 45:
            return 'bearish'
        else:
            return 'bullish'
    
    def _categorize_bb(self, position: float) -> str:
        """تصنيف موقع السعر من Bollinger Bands"""
        if position < 0.2:
            return 'near_lower'
        elif position > 0.8:
            return 'near_upper'
        else:
            return 'middle'
    
    def _categorize_trend(self, adx: float) -> str:
        """تصنيف قوة الترند"""
        if adx < 20:
            return 'no_trend'
        elif adx < 40:
            return 'weak_trend'
        elif adx < 60:
            return 'strong_trend'
        else:
            return 'very_strong_trend'
    
    def _analyze_market_context(self, trade_result: Dict) -> Dict:
        """تحليل سياق السوق"""
        return {
            'volatility': trade_result.get('volatility', 0),
            'trend_direction': trade_result.get('trend_direction', 'unknown'),
            'support_resistance_nearby': trade_result.get('near_support', False) or trade_result.get('near_resistance', False),
            'volume_profile': trade_result.get('volume_ratio', 1),
            'time_of_day': trade_result.get('hour', 0),
            'day_of_week': trade_result.get('day_of_week', 0)
        }
    
    def _create_pattern_key(self, patterns: Dict) -> str:
        """إنشاء مفتاح فريد للنمط"""
        key_parts = [
            patterns['candle_pattern'],
            patterns['indicators']['rsi_state'],
            patterns['indicators']['macd_state'],
            patterns['indicators']['trend_strength'],
            patterns['volume_state']
        ]
        return '_'.join(key_parts)
    
    def _analyze_failure(self, trade_result: Dict) -> str:
        """تحليل أسباب فشل الصفقة"""
        reasons = []
        
        # هل كان Stop Loss ضيق جداً؟
        if trade_result.get('exit_reason') == 'stop_loss':
            if trade_result.get('max_favorable_move', 0) > abs(trade_result['pnl_pips']) * 0.5:
                reasons.append('stop_loss_too_tight')
        
        # هل دخلنا عكس الترند؟
        if trade_result.get('trend_direction') == 'up' and trade_result.get('direction') == 'SELL':
            reasons.append('against_trend')
        elif trade_result.get('trend_direction') == 'down' and trade_result.get('direction') == 'BUY':
            reasons.append('against_trend')
        
        # هل كان هناك حجم ضعيف؟
        if trade_result.get('volume_ratio', 1) < 0.8:
            reasons.append('low_volume')
        
        # هل كان الوقت غير مناسب؟
        hour = trade_result.get('hour', 0)
        if hour < 8 or hour > 20:
            reasons.append('off_hours_trading')
        
        return ', '.join(reasons) if reasons else 'unknown'
    
    def _save_learning(self, trade_result: Dict, patterns: Dict, market_context: Dict, success: bool):
        """حفظ التعلم في قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        lessons = {
            'pattern_reliability': self._calculate_pattern_reliability(patterns),
            'market_conditions_impact': self._assess_market_impact(market_context, success),
            'timing_importance': self._assess_timing_impact(trade_result, success)
        }
        
        cursor.execute("""
            INSERT INTO continuous_learning (
                trade_id, symbol, timeframe, entry_time, exit_time,
                direction, result, pnl_pips, pattern_success,
                indicators_state, market_context, lessons_learned,
                confidence_adjustment, improvements
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade_result.get('trade_id', ''),
            trade_result.get('symbol', ''),
            trade_result.get('timeframe', ''),
            trade_result.get('entry_time', ''),
            trade_result.get('exit_time', ''),
            trade_result.get('direction', ''),
            'success' if success else 'failure',
            trade_result.get('pnl_pips', 0),
            json.dumps(patterns),
            json.dumps(patterns['indicators']),
            json.dumps(market_context),
            json.dumps(lessons),
            self._calculate_confidence_adjustment(patterns, success),
            json.dumps(self._suggest_improvements_for_pattern(patterns, success))
        ))
        
        conn.commit()
        conn.close()
    
    def _calculate_pattern_reliability(self, patterns: Dict) -> float:
        """حساب موثوقية النمط"""
        pattern_key = self._create_pattern_key(patterns)
        
        success_count = self.learning_memory['successful_patterns'].get(pattern_key, {}).get('count', 0)
        fail_count = self.learning_memory['failed_patterns'].get(pattern_key, {}).get('count', 0)
        
        total = success_count + fail_count
        if total == 0:
            return 0.5  # غير معروف
        
        return success_count / total
    
    def _assess_market_impact(self, market_context: Dict, success: bool) -> Dict:
        """تقييم تأثير ظروف السوق"""
        impact = {}
        
        if market_context['volatility'] > 0.02 and not success:
            impact['high_volatility'] = 'negative'
        elif market_context['volatility'] < 0.01 and success:
            impact['low_volatility'] = 'positive'
        
        if market_context['support_resistance_nearby'] and success:
            impact['sr_levels'] = 'very_positive'
        
        return impact
    
    def _assess_timing_impact(self, trade_result: Dict, success: bool) -> Dict:
        """تقييم تأثير التوقيت"""
        hour = trade_result.get('hour', 0)
        session = trade_result.get('trading_session', '')
        
        impact = {}
        
        if success:
            impact[f'hour_{hour}'] = 'positive'
            impact[f'session_{session}'] = 'positive'
        else:
            impact[f'hour_{hour}'] = 'negative'
            impact[f'session_{session}'] = 'negative'
        
        return impact
    
    def _calculate_confidence_adjustment(self, patterns: Dict, success: bool) -> float:
        """حساب تعديل الثقة للنمط"""
        base_adjustment = 0.05 if success else -0.05
        
        # تعديل إضافي بناءً على قوة المؤشرات
        if patterns['indicators']['trend_strength'] == 'strong_trend':
            base_adjustment *= 1.5
        elif patterns['indicators']['trend_strength'] == 'no_trend':
            base_adjustment *= 0.5
        
        return base_adjustment
    
    def _suggest_improvements_for_pattern(self, patterns: Dict, success: bool) -> List[str]:
        """اقتراح تحسينات للنمط"""
        improvements = []
        
        if not success:
            if patterns['indicators']['trend_strength'] == 'no_trend':
                improvements.append("avoid_trading_in_no_trend_conditions")
            
            if patterns['volume_state'] == 'normal':
                improvements.append("wait_for_high_volume_confirmation")
            
            if patterns['indicators']['rsi_state'] == 'neutral':
                improvements.append("wait_for_clearer_rsi_signal")
        
        return improvements
    
    def _update_pattern_performance(self, pattern_key: str, success: bool, pips: float):
        """تحديث أداء النمط في قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # جلب البيانات الحالية
        cursor.execute("SELECT * FROM pattern_performance WHERE pattern_key = ?", (pattern_key,))
        row = cursor.fetchone()
        
        if row:
            # تحديث البيانات الموجودة
            total_trades = row[1] + 1
            successful_trades = row[2] + (1 if success else 0)
            total_pips = row[3] + pips
            avg_pips = total_pips / total_trades
            success_rate = successful_trades / total_trades
            
            # حساب نقاط الثقة الجديدة
            confidence_score = success_rate * 0.6 + (avg_pips / 100) * 0.4
            confidence_score = max(0, min(1, confidence_score))  # بين 0 و 1
            
            cursor.execute("""
                UPDATE pattern_performance 
                SET total_trades = ?, successful_trades = ?, total_pips = ?,
                    avg_pips = ?, success_rate = ?, confidence_score = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE pattern_key = ?
            """, (total_trades, successful_trades, total_pips, avg_pips, 
                  success_rate, confidence_score, pattern_key))
        else:
            # إنشاء سجل جديد
            cursor.execute("""
                INSERT INTO pattern_performance 
                (pattern_key, total_trades, successful_trades, total_pips,
                 avg_pips, success_rate, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (pattern_key, 1, 1 if success else 0, pips, pips,
                  1.0 if success else 0.0, 0.6 if success else 0.4))
        
        conn.commit()
        conn.close()
    
    def _generate_improvements(self) -> List[str]:
        """توليد اقتراحات تحسين عامة"""
        improvements = []
        
        # تحليل الأنماط الفاشلة المتكررة
        for pattern_key, data in self.learning_memory['failed_patterns'].items():
            if data['count'] > 5:
                success_data = self.learning_memory['successful_patterns'].get(pattern_key, {'count': 0})
                if success_data['count'] < data['count'] * 0.5:
                    improvements.append(f"avoid_pattern_{pattern_key}")
        
        # تحليل أفضل الأوقات
        time_performance = self._analyze_time_performance()
        if time_performance['best_hours']:
            improvements.append(f"focus_trading_hours_{time_performance['best_hours']}")
        
        return improvements
    
    def _analyze_time_performance(self) -> Dict:
        """تحليل أداء الأوقات المختلفة"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                CAST(strftime('%H', entry_time) AS INTEGER) as hour,
                COUNT(*) as total,
                SUM(CASE WHEN result = 'success' THEN 1 ELSE 0 END) as wins,
                AVG(pnl_pips) as avg_pips
            FROM continuous_learning
            GROUP BY hour
            HAVING total > 5
            ORDER BY avg_pips DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if not df.empty:
            best_hours = df.head(3)['hour'].tolist()
            return {'best_hours': best_hours}
        
        return {'best_hours': []}
    
    def get_pattern_confidence(self, pattern_key: str) -> float:
        """الحصول على ثقة النمط"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT confidence_score FROM pattern_performance WHERE pattern_key = ?", (pattern_key,))
        row = cursor.fetchone()
        
        conn.close()
        
        return row[0] if row else 0.5
    
    def get_learning_insights(self) -> Dict:
        """الحصول على رؤى التعلم"""
        conn = sqlite3.connect(self.db_path)
        
        # إحصائيات عامة
        stats_query = """
            SELECT 
                COUNT(*) as total_trades_analyzed,
                SUM(CASE WHEN result = 'success' THEN 1 ELSE 0 END) as successful_trades,
                AVG(pnl_pips) as avg_pips,
                MAX(pnl_pips) as best_trade,
                MIN(pnl_pips) as worst_trade
            FROM continuous_learning
        """
        
        stats = pd.read_sql_query(stats_query, conn).iloc[0].to_dict()
        
        # أفضل الأنماط
        best_patterns_query = """
            SELECT pattern_key, success_rate, avg_pips, total_trades
            FROM pattern_performance
            WHERE total_trades > 10
            ORDER BY confidence_score DESC
            LIMIT 5
        """
        
        best_patterns = pd.read_sql_query(best_patterns_query, conn).to_dict('records')
        
        # أسوأ الأنماط
        worst_patterns_query = """
            SELECT pattern_key, success_rate, avg_pips, total_trades
            FROM pattern_performance
            WHERE total_trades > 10
            ORDER BY confidence_score ASC
            LIMIT 5
        """
        
        worst_patterns = pd.read_sql_query(worst_patterns_query, conn).to_dict('records')
        
        conn.close()
        
        return {
            'general_stats': stats,
            'best_patterns': best_patterns,
            'worst_patterns': worst_patterns,
            'improvements': self.learning_memory['improvement_suggestions'][-10:]  # آخر 10 اقتراحات
        }
    
    def _save_learning_memory(self):
        """حفظ ذاكرة التعلم"""
        memory_file = 'data/learning_memory.json'
        with open(memory_file, 'w') as f:
            json.dump(self.learning_memory, f, indent=2)
    
    def _load_learning_memory(self):
        """تحميل ذاكرة التعلم"""
        memory_file = 'data/learning_memory.json'
        if os.path.exists(memory_file):
            with open(memory_file, 'r') as f:
                self.learning_memory = json.load(f)
    
    def continuous_improvement_cycle(self):
        """دورة التحسين المستمر - تُشغل يومياً"""
        logger.info("Starting continuous improvement cycle...")
        
        # 1. تحليل أداء آخر 24 ساعة
        recent_performance = self._analyze_recent_performance()
        
        # 2. تحديد نقاط الضعف
        weaknesses = self._identify_weaknesses(recent_performance)
        
        # 3. توليد خطة تحسين
        improvement_plan = self._generate_improvement_plan(weaknesses)
        
        # 4. تحديث الإعدادات تلقائياً
        self._apply_improvements(improvement_plan)
        
        # 5. إرسال تقرير التحسين
        self._send_improvement_report(improvement_plan)
        
        logger.info("Continuous improvement cycle completed")
    
    def _analyze_recent_performance(self) -> Dict:
        """تحليل الأداء الأخير"""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT * FROM continuous_learning
            WHERE created_at > datetime('now', '-1 day')
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if df.empty:
            return {}
        
        return {
            'total_trades': len(df),
            'success_rate': (df['result'] == 'success').mean(),
            'avg_pips': df['pnl_pips'].mean(),
            'problem_patterns': df[df['result'] == 'failure']['pattern_success'].value_counts().to_dict()
        }
    
    def _identify_weaknesses(self, performance: Dict) -> List[str]:
        """تحديد نقاط الضعف"""
        weaknesses = []
        
        if performance.get('success_rate', 0) < 0.5:
            weaknesses.append('low_success_rate')
        
        if performance.get('avg_pips', 0) < 0:
            weaknesses.append('negative_expectancy')
        
        # تحليل الأنماط المشكلة
        for pattern in performance.get('problem_patterns', {}).keys():
            weaknesses.append(f'failing_pattern_{pattern}')
        
        return weaknesses
    
    def _generate_improvement_plan(self, weaknesses: List[str]) -> Dict:
        """توليد خطة التحسين"""
        plan = {
            'adjustments': [],
            'new_rules': [],
            'patterns_to_avoid': []
        }
        
        for weakness in weaknesses:
            if weakness == 'low_success_rate':
                plan['adjustments'].append({
                    'parameter': 'min_confidence',
                    'action': 'increase',
                    'new_value': 0.75
                })
            
            elif weakness == 'negative_expectancy':
                plan['new_rules'].append('increase_risk_reward_ratio')
            
            elif weakness.startswith('failing_pattern_'):
                pattern = weakness.replace('failing_pattern_', '')
                plan['patterns_to_avoid'].append(pattern)
        
        return plan
    
    def _apply_improvements(self, plan: Dict):
        """تطبيق التحسينات تلقائياً"""
        # هنا يمكن تحديث إعدادات النظام تلقائياً
        # مثل رفع الحد الأدنى للثقة أو تغيير معايير الدخول
        
        if plan['adjustments']:
            logger.info(f"Applying adjustments: {plan['adjustments']}")
        
        if plan['patterns_to_avoid']:
            # حفظ الأنماط التي يجب تجنبها
            with open('data/blacklisted_patterns.json', 'w') as f:
                json.dump(plan['patterns_to_avoid'], f)
    
    def _send_improvement_report(self, plan: Dict):
        """إرسال تقرير التحسين"""
        report = "🔧 تقرير التحسين المستمر\n\n"
        
        if plan['adjustments']:
            report += "📊 التعديلات:\n"
            for adj in plan['adjustments']:
                report += f"• {adj['parameter']}: {adj['new_value']}\n"
        
        if plan['patterns_to_avoid']:
            report += "\n⚠️ أنماط يجب تجنبها:\n"
            for pattern in plan['patterns_to_avoid']:
                report += f"• {pattern}\n"
        
        logger.info(report)