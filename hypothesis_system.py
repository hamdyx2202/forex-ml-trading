#!/usr/bin/env python3
"""
🔬 Advanced Hypothesis System - نظام الفرضيات المتقدم
📊 يولد ويختبر فرضيات السوق
🎯 يتعلم من النجاحات والفشل
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
import uuid
from scipy import stats
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MarketHypothesis:
    """فرضية السوق"""
    hypothesis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: str = ""  # trend, reversal, breakout, range, volatility
    conditions: Dict[str, Any] = field(default_factory=dict)
    indicators: List[str] = field(default_factory=list)
    timeframes: List[str] = field(default_factory=list)
    symbols: List[str] = field(default_factory=list)
    expected_outcome: Dict[str, Any] = field(default_factory=dict)
    entry_rules: Dict[str, Any] = field(default_factory=dict)
    exit_rules: Dict[str, Any] = field(default_factory=dict)
    risk_parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5
    priority: int = 5
    test_results: List[Dict] = field(default_factory=list)
    backtest_results: Dict[str, Any] = field(default_factory=dict)
    live_results: Dict[str, Any] = field(default_factory=dict)
    validation_score: float = 0.0
    success_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tested_count: int = 0
    successful_trades: int = 0
    failed_trades: int = 0
    is_active: bool = True
    is_validated: bool = False
    parent_hypothesis_id: Optional[str] = None
    child_hypotheses: List[str] = field(default_factory=list)
    meta_data: Dict[str, Any] = field(default_factory=dict)

class HypothesisGenerator:
    """مولد الفرضيات"""
    
    def __init__(self):
        self.hypothesis_templates = self._load_hypothesis_templates()
        self.market_patterns = self._load_market_patterns()
        self.successful_patterns = []
        self.failed_patterns = []
        
    def _load_hypothesis_templates(self) -> List[Dict]:
        """تحميل قوالب الفرضيات"""
        return [
            # فرضيات الاتجاه
            {
                'category': 'trend',
                'template': 'trend_following',
                'conditions': {
                    'ma_alignment': 'bullish',  # MA50 > MA100 > MA200
                    'momentum': 'positive',
                    'volume': 'increasing'
                },
                'expected_outcome': {
                    'direction': 'up',
                    'duration': '4-8 hours',
                    'target': '1.5% - 3%'
                }
            },
            {
                'category': 'trend',
                'template': 'trend_reversal',
                'conditions': {
                    'divergence': True,
                    'support_resistance': 'at_level',
                    'candlestick_pattern': 'reversal'
                },
                'expected_outcome': {
                    'direction': 'reverse',
                    'duration': '2-4 hours',
                    'target': '1% - 2%'
                }
            },
            # فرضيات الاختراق
            {
                'category': 'breakout',
                'template': 'range_breakout',
                'conditions': {
                    'price_action': 'consolidation',
                    'volatility': 'contracting',
                    'volume': 'decreasing'
                },
                'expected_outcome': {
                    'direction': 'breakout',
                    'duration': '1-2 hours',
                    'target': 'range_height * 1.5'
                }
            },
            {
                'category': 'breakout',
                'template': 'triangle_breakout',
                'conditions': {
                    'pattern': 'triangle',
                    'volume': 'decreasing',
                    'time_in_pattern': '>20 candles'
                },
                'expected_outcome': {
                    'direction': 'breakout_direction',
                    'duration': '2-3 hours',
                    'target': 'pattern_height'
                }
            },
            # فرضيات التذبذب
            {
                'category': 'volatility',
                'template': 'volatility_expansion',
                'conditions': {
                    'bollinger_squeeze': True,
                    'atr': 'low',
                    'news_event': 'upcoming'
                },
                'expected_outcome': {
                    'volatility': 'increase',
                    'duration': '30-60 minutes',
                    'range': '2x_atr'
                }
            },
            # فرضيات النطاق
            {
                'category': 'range',
                'template': 'range_trading',
                'conditions': {
                    'market_structure': 'ranging',
                    'support_resistance': 'defined',
                    'oscillator': 'oversold/overbought'
                },
                'expected_outcome': {
                    'direction': 'mean_reversion',
                    'duration': '2-4 hours',
                    'target': 'opposite_boundary'
                }
            },
            # فرضيات متقدمة
            {
                'category': 'complex',
                'template': 'multi_timeframe_confluence',
                'conditions': {
                    'h4_trend': 'up',
                    'h1_pullback': True,
                    'm15_reversal_signal': True
                },
                'expected_outcome': {
                    'direction': 'up',
                    'duration': '4-8 hours',
                    'target': '2% - 4%'
                }
            }
        ]
    
    def _load_market_patterns(self) -> List[Dict]:
        """تحميل أنماط السوق"""
        return [
            {'name': 'double_top', 'type': 'reversal', 'reliability': 0.75},
            {'name': 'double_bottom', 'type': 'reversal', 'reliability': 0.75},
            {'name': 'head_shoulders', 'type': 'reversal', 'reliability': 0.80},
            {'name': 'triangle', 'type': 'continuation', 'reliability': 0.70},
            {'name': 'flag', 'type': 'continuation', 'reliability': 0.72},
            {'name': 'wedge', 'type': 'reversal', 'reliability': 0.68},
            {'name': 'channel', 'type': 'continuation', 'reliability': 0.65}
        ]
    
    async def generate_hypotheses(self, market_data: Dict, 
                                historical_performance: Dict) -> List[MarketHypothesis]:
        """توليد فرضيات جديدة"""
        hypotheses = []
        
        # 1. توليد من القوالب
        template_hypotheses = await self._generate_from_templates(market_data)
        hypotheses.extend(template_hypotheses)
        
        # 2. توليد من الأنماط الناجحة
        pattern_hypotheses = await self._generate_from_successful_patterns(
            market_data, historical_performance
        )
        hypotheses.extend(pattern_hypotheses)
        
        # 3. توليد فرضيات متكيفة
        adaptive_hypotheses = await self._generate_adaptive_hypotheses(
            market_data, historical_performance
        )
        hypotheses.extend(adaptive_hypotheses)
        
        # 4. توليد فرضيات مركبة
        composite_hypotheses = await self._generate_composite_hypotheses(
            hypotheses[:10]  # أفضل 10 فرضيات
        )
        hypotheses.extend(composite_hypotheses)
        
        # 5. تقييم وترتيب الفرضيات
        scored_hypotheses = await self._score_hypotheses(hypotheses, market_data)
        
        # إرجاع أفضل الفرضيات
        return sorted(scored_hypotheses, key=lambda h: h.confidence, reverse=True)[:50]
    
    async def _generate_from_templates(self, market_data: Dict) -> List[MarketHypothesis]:
        """توليد فرضيات من القوالب"""
        hypotheses = []
        
        for template in self.hypothesis_templates:
            # تخصيص القالب للسوق الحالي
            hypothesis = MarketHypothesis(
                name=f"{template['template']}_{datetime.now().strftime('%Y%m%d_%H%M')}",
                description=f"Hypothesis based on {template['template']} pattern",
                category=template['category'],
                conditions=template['conditions'],
                expected_outcome=template['expected_outcome'],
                confidence=0.5  # ثقة أولية
            )
            
            # تخصيص للرموز المناسبة
            suitable_symbols = await self._find_suitable_symbols(
                template, market_data
            )
            hypothesis.symbols = suitable_symbols[:10]
            
            # تحديد الإطارات الزمنية المناسبة
            hypothesis.timeframes = self._determine_timeframes(template)
            
            # تحديد المؤشرات المطلوبة
            hypothesis.indicators = self._extract_required_indicators(template)
            
            # تحديد قواعد الدخول والخروج
            hypothesis.entry_rules = self._generate_entry_rules(template)
            hypothesis.exit_rules = self._generate_exit_rules(template)
            
            # تحديد معاملات المخاطرة
            hypothesis.risk_parameters = self._generate_risk_parameters(template)
            
            hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_adaptive_hypotheses(self, market_data: Dict,
                                          historical_performance: Dict) -> List[MarketHypothesis]:
        """توليد فرضيات متكيفة"""
        hypotheses = []
        
        # تحليل الأداء التاريخي
        performance_patterns = self._analyze_performance_patterns(historical_performance)
        
        for pattern in performance_patterns:
            if pattern['success_rate'] > 0.6:
                # إنشاء فرضية محسنة
                hypothesis = MarketHypothesis(
                    name=f"adaptive_{pattern['name']}_{datetime.now().strftime('%Y%m%d')}",
                    description=f"Adaptive hypothesis based on successful {pattern['name']}",
                    category='adaptive',
                    conditions=self._enhance_conditions(pattern['conditions']),
                    expected_outcome=pattern['expected_outcome'],
                    confidence=pattern['success_rate']
                )
                
                # تحسين القواعد بناءً على التعلم
                hypothesis.entry_rules = self._optimize_entry_rules(
                    pattern['entry_rules'], 
                    pattern['successful_trades']
                )
                
                hypothesis.exit_rules = self._optimize_exit_rules(
                    pattern['exit_rules'],
                    pattern['successful_trades']
                )
                
                hypotheses.append(hypothesis)
        
        return hypotheses
    
    async def _generate_composite_hypotheses(self, 
                                           base_hypotheses: List[MarketHypothesis]) -> List[MarketHypothesis]:
        """توليد فرضيات مركبة"""
        composite_hypotheses = []
        
        # دمج الفرضيات المتوافقة
        for i, h1 in enumerate(base_hypotheses):
            for j, h2 in enumerate(base_hypotheses[i+1:], i+1):
                if self._are_compatible(h1, h2):
                    composite = self._create_composite_hypothesis(h1, h2)
                    composite_hypotheses.append(composite)
        
        return composite_hypotheses
    
    def _are_compatible(self, h1: MarketHypothesis, h2: MarketHypothesis) -> bool:
        """التحقق من توافق فرضيتين"""
        # التحقق من عدم التعارض في الاتجاه
        if h1.expected_outcome.get('direction') and h2.expected_outcome.get('direction'):
            if h1.expected_outcome['direction'] != h2.expected_outcome['direction']:
                return False
        
        # التحقق من توافق الإطارات الزمنية
        common_timeframes = set(h1.timeframes) & set(h2.timeframes)
        if not common_timeframes:
            return False
        
        # التحقق من توافق الفئات
        compatible_categories = {
            'trend': ['momentum', 'breakout'],
            'breakout': ['trend', 'volatility'],
            'range': ['volatility'],
            'volatility': ['breakout', 'range']
        }
        
        if h2.category not in compatible_categories.get(h1.category, []):
            return False
        
        return True
    
    def _create_composite_hypothesis(self, h1: MarketHypothesis, 
                                   h2: MarketHypothesis) -> MarketHypothesis:
        """إنشاء فرضية مركبة"""
        composite = MarketHypothesis(
            name=f"composite_{h1.name}_{h2.name}",
            description=f"Composite of {h1.description} and {h2.description}",
            category='composite',
            parent_hypothesis_id=h1.hypothesis_id,
            child_hypotheses=[h1.hypothesis_id, h2.hypothesis_id]
        )
        
        # دمج الشروط
        composite.conditions = {**h1.conditions, **h2.conditions}
        
        # دمج المؤشرات
        composite.indicators = list(set(h1.indicators + h2.indicators))
        
        # دمج الإطارات الزمنية
        composite.timeframes = list(set(h1.timeframes) & set(h2.timeframes))
        
        # دمج الرموز
        composite.symbols = list(set(h1.symbols) & set(h2.symbols))
        
        # حساب الثقة المركبة
        composite.confidence = (h1.confidence + h2.confidence) / 2 * 1.1  # مكافأة للتأكيد المتعدد
        
        return composite

class HypothesisTester:
    """مختبر الفرضيات"""
    
    def __init__(self):
        self.test_results = []
        self.validation_criteria = {
            'min_trades': 30,
            'min_success_rate': 0.55,
            'min_profit_factor': 1.2,
            'max_drawdown': 0.15,
            'min_sharpe_ratio': 1.0
        }
    
    async def test_hypothesis(self, hypothesis: MarketHypothesis, 
                            market_data: pd.DataFrame,
                            mode: str = 'backtest') -> Dict[str, Any]:
        """اختبار فرضية"""
        logger.info(f"Testing hypothesis: {hypothesis.name}")
        
        if mode == 'backtest':
            results = await self._backtest_hypothesis(hypothesis, market_data)
        elif mode == 'paper':
            results = await self._paper_trade_hypothesis(hypothesis, market_data)
        elif mode == 'live':
            results = await self._live_test_hypothesis(hypothesis, market_data)
        else:
            raise ValueError(f"Unknown test mode: {mode}")
        
        # تحديث الفرضية بالنتائج
        hypothesis.test_results.append(results)
        hypothesis.tested_count += 1
        
        # حساب المقاييس
        self._update_hypothesis_metrics(hypothesis, results)
        
        # التحقق من الصلاحية
        hypothesis.is_validated = self._validate_hypothesis(hypothesis)
        
        return results
    
    async def _backtest_hypothesis(self, hypothesis: MarketHypothesis,
                                 market_data: pd.DataFrame) -> Dict[str, Any]:
        """اختبار رجعي للفرضية"""
        trades = []
        
        for i in range(100, len(market_data)):
            window = market_data.iloc[i-100:i]
            
            # التحقق من شروط الدخول
            if self._check_entry_conditions(hypothesis, window):
                entry_price = window.iloc[-1]['close']
                entry_time = window.iloc[-1].name
                
                # البحث عن نقطة الخروج
                exit_idx, exit_price, exit_reason = self._find_exit_point(
                    hypothesis, market_data[i:i+50], entry_price
                )
                
                if exit_idx is not None:
                    trade = {
                        'entry_time': entry_time,
                        'entry_price': entry_price,
                        'exit_time': market_data.index[i + exit_idx],
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'profit': (exit_price - entry_price) / entry_price,
                        'duration': exit_idx
                    }
                    trades.append(trade)
        
        # حساب المقاييس
        results = self._calculate_backtest_metrics(trades)
        results['trades'] = trades
        
        return results
    
    def _check_entry_conditions(self, hypothesis: MarketHypothesis,
                              window: pd.DataFrame) -> bool:
        """التحقق من شروط الدخول"""
        # هنا يتم تنفيذ منطق التحقق من الشروط
        # هذا مثال مبسط
        for condition, value in hypothesis.entry_rules.items():
            if condition == 'ma_crossover':
                if not self._check_ma_crossover(window, value):
                    return False
            elif condition == 'rsi_level':
                if not self._check_rsi_level(window, value):
                    return False
            # إضافة المزيد من الشروط حسب الحاجة
        
        return True
    
    def _calculate_backtest_metrics(self, trades: List[Dict]) -> Dict[str, Any]:
        """حساب مقاييس الاختبار الرجعي"""
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'success_rate': 0,
                'profit_factor': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'avg_duration': 0
            }
        
        profits = [t['profit'] for t in trades]
        winning_trades = [t for t in trades if t['profit'] > 0]
        losing_trades = [t for t in trades if t['profit'] <= 0]
        
        total_profit = sum([t['profit'] for t in winning_trades])
        total_loss = abs(sum([t['profit'] for t in losing_trades]))
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'success_rate': len(winning_trades) / len(trades) if trades else 0,
            'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
            'sharpe_ratio': self._calculate_sharpe_ratio(profits),
            'max_drawdown': self._calculate_max_drawdown(profits),
            'avg_profit': np.mean([t['profit'] for t in winning_trades]) if winning_trades else 0,
            'avg_loss': np.mean([t['profit'] for t in losing_trades]) if losing_trades else 0,
            'avg_duration': np.mean([t['duration'] for t in trades])
        }
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.02) -> float:
        """حساب نسبة شارب"""
        if not returns or len(returns) < 2:
            return 0
        
        returns_array = np.array(returns)
        excess_returns = returns_array - risk_free_rate / 252  # يومي
        
        if np.std(excess_returns) == 0:
            return 0
        
        return np.sqrt(252) * np.mean(excess_returns) / np.std(excess_returns)
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """حساب أقصى انخفاض"""
        if not returns:
            return 0
        
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        
        return abs(np.min(drawdown))

class HypothesisManager:
    """مدير الفرضيات"""
    
    def __init__(self):
        self.generator = HypothesisGenerator()
        self.tester = HypothesisTester()
        self.active_hypotheses: List[MarketHypothesis] = []
        self.hypothesis_history: List[MarketHypothesis] = []
        self.performance_threshold = 0.6
        self.max_active_hypotheses = 20
        
    async def update_hypotheses(self, market_data: Dict,
                              performance_data: Dict) -> List[MarketHypothesis]:
        """تحديث الفرضيات"""
        logger.info("Updating market hypotheses...")
        
        # 1. تقييم الفرضيات الحالية
        await self._evaluate_active_hypotheses(market_data)
        
        # 2. إزالة الفرضيات الضعيفة
        self._prune_weak_hypotheses()
        
        # 3. توليد فرضيات جديدة
        new_hypotheses = await self.generator.generate_hypotheses(
            market_data, performance_data
        )
        
        # 4. اختبار الفرضيات الجديدة
        tested_hypotheses = await self._test_new_hypotheses(
            new_hypotheses[:10], market_data
        )
        
        # 5. إضافة الفرضيات الواعدة
        self._add_promising_hypotheses(tested_hypotheses)
        
        # 6. تطور الفرضيات الناجحة
        evolved_hypotheses = await self._evolve_successful_hypotheses()
        self.active_hypotheses.extend(evolved_hypotheses)
        
        # 7. الحفاظ على العدد الأمثل
        self._maintain_optimal_count()
        
        return self.active_hypotheses
    
    async def _evaluate_active_hypotheses(self, market_data: Dict):
        """تقييم الفرضيات النشطة"""
        for hypothesis in self.active_hypotheses:
            # اختبار مستمر
            recent_performance = await self._test_recent_performance(
                hypothesis, market_data
            )
            
            # تحديث المقاييس
            hypothesis.validation_score = self._calculate_validation_score(
                hypothesis, recent_performance
            )
            
            # تحديث الحالة
            if hypothesis.validation_score < 0.4:
                hypothesis.is_active = False
                logger.info(f"Deactivating hypothesis: {hypothesis.name}")
    
    def _prune_weak_hypotheses(self):
        """إزالة الفرضيات الضعيفة"""
        # نقل الفرضيات الضعيفة للتاريخ
        weak_hypotheses = [h for h in self.active_hypotheses 
                          if not h.is_active or h.validation_score < 0.4]
        
        for hypothesis in weak_hypotheses:
            self.hypothesis_history.append(hypothesis)
            self.active_hypotheses.remove(hypothesis)
    
    async def _evolve_successful_hypotheses(self) -> List[MarketHypothesis]:
        """تطوير الفرضيات الناجحة"""
        evolved = []
        
        successful = [h for h in self.active_hypotheses 
                     if h.success_rate > 0.65 and h.tested_count > 50]
        
        for hypothesis in successful:
            # إنشاء متغيرات محسنة
            variations = await self._create_hypothesis_variations(hypothesis)
            
            for variation in variations[:3]:  # أفضل 3 متغيرات
                # اختبار سريع
                test_results = await self.tester.test_hypothesis(
                    variation, self._get_recent_market_data()
                )
                
                if test_results['success_rate'] > hypothesis.success_rate:
                    evolved.append(variation)
                    logger.info(f"Evolved hypothesis: {variation.name}")
        
        return evolved
    
    async def _create_hypothesis_variations(self, 
                                          base_hypothesis: MarketHypothesis) -> List[MarketHypothesis]:
        """إنشاء متغيرات من فرضية ناجحة"""
        variations = []
        
        # تغيير معاملات المخاطرة
        risk_variation = self._create_risk_variation(base_hypothesis)
        variations.append(risk_variation)
        
        # تغيير الإطار الزمني
        timeframe_variation = self._create_timeframe_variation(base_hypothesis)
        variations.append(timeframe_variation)
        
        # تغيير الشروط
        condition_variation = self._create_condition_variation(base_hypothesis)
        variations.append(condition_variation)
        
        # دمج مع فرضيات أخرى ناجحة
        hybrid_variations = await self._create_hybrid_variations(base_hypothesis)
        variations.extend(hybrid_variations)
        
        return variations
    
    def save_hypotheses(self, filepath: str):
        """حفظ الفرضيات"""
        data = {
            'active_hypotheses': [asdict(h) for h in self.active_hypotheses],
            'hypothesis_history': [asdict(h) for h in self.hypothesis_history[-100:]],
            'metadata': {
                'last_update': datetime.now().isoformat(),
                'total_generated': len(self.hypothesis_history) + len(self.active_hypotheses),
                'active_count': len(self.active_hypotheses)
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        logger.info(f"Saved {len(self.active_hypotheses)} active hypotheses")
    
    def load_hypotheses(self, filepath: str):
        """تحميل الفرضيات"""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.active_hypotheses = [
                MarketHypothesis(**h) for h in data['active_hypotheses']
            ]
            
            logger.info(f"Loaded {len(self.active_hypotheses)} active hypotheses")

# مثال على الاستخدام
async def main():
    """تشغيل نظام الفرضيات"""
    manager = HypothesisManager()
    
    # بيانات السوق التجريبية
    market_data = {
        'symbols': ['EUR/USD', 'GBP/USD', 'USD/JPY'],
        'current_prices': {'EUR/USD': 1.0850, 'GBP/USD': 1.2650, 'USD/JPY': 149.50},
        'volatility': {'EUR/USD': 0.08, 'GBP/USD': 0.10, 'USD/JPY': 0.12}
    }
    
    performance_data = {
        'total_trades': 1000,
        'winning_trades': 580,
        'success_rate': 0.58
    }
    
    # تحديث الفرضيات
    hypotheses = await manager.update_hypotheses(market_data, performance_data)
    
    print(f"Generated {len(hypotheses)} active hypotheses")
    
    # حفظ الفرضيات
    manager.save_hypotheses('market_hypotheses.json')

if __name__ == "__main__":
    asyncio.run(main())