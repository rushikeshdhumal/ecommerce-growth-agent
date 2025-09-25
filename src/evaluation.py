"""
Evaluation System for E-commerce Growth Agent
Handles performance tracking, A/B testing, and optimization analysis
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict
import json
import sqlite3
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

from config.settings import settings
from config.logging_config import get_agent_logger


@dataclass
class PerformanceMetric:
    """Performance metric structure"""
    metric_name: str
    current_value: float
    previous_value: float
    benchmark_value: float
    trend: str
    significance: float
    confidence_interval: Tuple[float, float]


@dataclass
class ABTestResult:
    """A/B test result structure"""
    test_id: str
    test_name: str
    variant_a_performance: Dict[str, float]
    variant_b_performance: Dict[str, float]
    winner: str
    confidence_level: float
    statistical_significance: bool
    lift_percentage: float
    sample_size_a: int
    sample_size_b: int


@dataclass
class AnomalyDetection:
    """Anomaly detection result"""
    metric_name: str
    current_value: float
    expected_range: Tuple[float, float]
    anomaly_score: float
    is_anomaly: bool
    severity: str
    recommended_action: str


class EvaluationSystem:
    """
    Comprehensive evaluation system for monitoring campaign performance,
    conducting A/B tests, and detecting anomalies
    """
    
    def __init__(self):
        self.logger = get_agent_logger("EvaluationSystem")
        self.db_path = "data/ecommerce_agent.db"
        self.conn = sqlite3.connect(self.db_path)
        
        # Performance tracking
        self.performance_history = []
        self.benchmark_metrics = self._load_benchmark_metrics()
        
        # A/B testing
        self.active_tests = {}
        self.completed_tests = []
        
        # Anomaly detection parameters
        self.anomaly_thresholds = {
            'roas': {'min': 1.0, 'max': 10.0, 'std_multiplier': 2.5},
            'ctr': {'min': 0.001, 'max': 0.20, 'std_multiplier': 2.0},
            'conversion_rate': {'min': 0.001, 'max': 0.50, 'std_multiplier': 2.0},
            'cpc': {'min': 0.10, 'max': 50.0, 'std_multiplier': 2.5},
            'spend': {'min': 0, 'max': settings.MAX_DAILY_BUDGET * 2, 'std_multiplier': 3.0}
        }
        
        self._setup_evaluation_tables()
        self.logger.log_action("evaluation_system_initialized", {})
    
    def _setup_evaluation_tables(self):
        """Setup database tables for evaluation tracking"""
        cursor = self.conn.cursor()
        
        # Performance metrics history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                campaign_id TEXT,
                channel TEXT,
                metric_name TEXT,
                metric_value REAL,
                benchmark_value REAL,
                context TEXT
            )
        """)
        
        # A/B test results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_test_results (
                test_id TEXT PRIMARY KEY,
                test_name TEXT,
                campaign_id TEXT,
                start_date DATETIME,
                end_date DATETIME,
                variant_a_data TEXT,
                variant_b_data TEXT,
                winner TEXT,
                confidence_level REAL,
                statistical_significance BOOLEAN,
                lift_percentage REAL
            )
        """)
        
        # Anomaly detection log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                campaign_id TEXT,
                metric_name TEXT,
                current_value REAL,
                expected_value REAL,
                anomaly_score REAL,
                severity TEXT,
                action_taken TEXT
            )
        """)
        
        self.conn.commit()
    
    def _load_benchmark_metrics(self) -> Dict[str, float]:
        """Load industry benchmark metrics"""
        return {
            'roas': 3.0,
            'ctr': 0.025,
            'conversion_rate': 0.035,
            'cpc': 2.50,
            'customer_ltv': 250.0,
            'churn_rate': 0.15,
            'email_open_rate': 0.22,
            'email_click_rate': 0.035
        }
    
    def calculate_current_metrics(self) -> Dict[str, float]:
        """Calculate current performance metrics across all campaigns"""
        cursor = self.conn.cursor()
        
        # Get recent campaign performance data
        cursor.execute("""
            SELECT channel, SUM(spend) as total_spend, SUM(revenue) as total_revenue,
                   SUM(clicks) as total_clicks, SUM(impressions) as total_impressions,
                   SUM(conversions) as total_conversions
            FROM campaign_performance 
            WHERE start_date >= date('now', '-30 days')
            GROUP BY channel
        """)
        
        channel_data = cursor.fetchall()
        
        # Calculate overall metrics
        total_spend = sum(row[1] for row in channel_data)
        total_revenue = sum(row[2] for row in channel_data)
        total_clicks = sum(row[3] for row in channel_data)
        total_impressions = sum(row[4] for row in channel_data)
        total_conversions = sum(row[5] for row in channel_data)
        
        current_metrics = {
            'overall_roas': total_revenue / total_spend if total_spend > 0 else 0,
            'overall_ctr': total_clicks / total_impressions if total_impressions > 0 else 0,
            'overall_conversion_rate': total_conversions / total_clicks if total_clicks > 0 else 0,
            'overall_cpc': total_spend / total_clicks if total_clicks > 0 else 0,
            'total_spend': total_spend,
            'total_revenue': total_revenue,
            'total_conversions': total_conversions
        }
        
        # Add channel-specific metrics
        for channel_name, spend, revenue, clicks, impressions, conversions in channel_data:
            current_metrics[f'{channel_name}_roas'] = revenue / spend if spend > 0 else 0
            current_metrics[f'{channel_name}_ctr'] = clicks / impressions if impressions > 0 else 0
            current_metrics[f'{channel_name}_conversion_rate'] = conversions / clicks if clicks > 0 else 0
        
        # Get customer metrics
        cursor.execute("SELECT COUNT(*) FROM customers WHERE days_since_last_purchase <= 30")
        active_customers = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM customers WHERE days_since_last_purchase > 90")
        churned_customers = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM customers")
        total_customers = cursor.fetchone()[0]
        
        current_metrics.update({
            'active_customers': active_customers,
            'churned_customers': churned_customers,
            'churn_rate': churned_customers / total_customers if total_customers > 0 else 0,
            'active_customer_rate': active_customers / total_customers if total_customers > 0 else 0
        })
        
        # Store metrics in history
        self._store_performance_metrics(current_metrics)
        
        return current_metrics
    
    def _store_performance_metrics(self, metrics: Dict[str, float]):
        """Store performance metrics in database"""
        cursor = self.conn.cursor()
        timestamp = datetime.now().isoformat()
        
        for metric_name, value in metrics.items():
            benchmark = self.benchmark_metrics.get(metric_name.split('_')[-1], 0)
            
            cursor.execute("""
                INSERT INTO performance_history 
                (timestamp, campaign_id, channel, metric_name, metric_value, benchmark_value, context)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (timestamp, 'overall', 'all', metric_name, value, benchmark, 'system_evaluation'))
        
        self.conn.commit()
    
    def analyze_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time"""
        cursor = self.conn.cursor()
        
        # Get historical data for trend analysis
        cursor.execute("""
            SELECT metric_name, metric_value, timestamp
            FROM performance_history 
            WHERE timestamp >= date('now', '-30 days')
            ORDER BY metric_name, timestamp
        """)
        
        historical_data = cursor.fetchall()
        
        # Group by metric
        metrics_data = {}
        for metric_name, value, timestamp in historical_data:
            if metric_name not in metrics_data:
                metrics_data[metric_name] = []
            metrics_data[metric_name].append((timestamp, value))
        
        trends = {}
        
        for metric_name, data_points in metrics_data.items():
            if len(data_points) >= 5:  # Need minimum data points for trend analysis
                values = [point[1] for point in data_points]
                
                # Calculate trend using linear regression
                x = np.arange(len(values))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
                
                # Determine trend direction and strength
                if abs(r_value) < 0.3:
                    trend_direction = "stable"
                elif slope > 0:
                    trend_direction = "increasing"
                else:
                    trend_direction = "decreasing"
                
                trend_strength = abs(r_value)
                
                # Calculate percentage change
                if len(values) >= 2:
                    recent_avg = np.mean(values[-3:])
                    earlier_avg = np.mean(values[:3])
                    pct_change = ((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg != 0 else 0
                else:
                    pct_change = 0
                
                trends[metric_name] = {
                    'direction': trend_direction,
                    'strength': round(trend_strength, 3),
                    'slope': round(slope, 4),
                    'r_squared': round(r_value ** 2, 3),
                    'p_value': round(p_value, 4),
                    'percentage_change': round(pct_change, 2),
                    'data_points': len(values),
                    'current_value': values[-1],
                    'trend_confidence': self._calculate_trend_confidence(r_value, p_value, len(values))
                }
        
        return trends
    
    def _calculate_trend_confidence(self, r_value: float, p_value: float, sample_size: int) -> str:
        """Calculate confidence level for trend analysis"""
        if sample_size < 5:
            return "insufficient_data"
        elif p_value <= 0.01 and abs(r_value) >= 0.7:
            return "very_high"
        elif p_value <= 0.05 and abs(r_value) >= 0.5:
            return "high"
        elif p_value <= 0.10 and abs(r_value) >= 0.3:
            return "medium"
        else:
            return "low"
    
    def detect_anomalies(self, current_metrics: Dict[str, float]) -> List[AnomalyDetection]:
        """Detect anomalies in current performance metrics"""
        anomalies = []
        
        # Get historical data for each metric
        cursor = self.conn.cursor()
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in self.anomaly_thresholds:
                # Get historical values for this metric
                cursor.execute("""
                    SELECT metric_value 
                    FROM performance_history 
                    WHERE metric_name = ? AND timestamp >= date('now', '-30 days')
                    ORDER BY timestamp DESC
                    LIMIT 30
                """, (metric_name,))
                
                historical_values = [row[0] for row in cursor.fetchall()]
                
                if len(historical_values) >= 5:  # Need minimum history
                    anomaly_result = self._detect_statistical_anomaly(
                        metric_name, current_value, historical_values
                    )
                    
                    if anomaly_result:
                        anomalies.append(anomaly_result)
                        
                        # Log anomaly
                        self._log_anomaly(anomaly_result)
        
        return anomalies
    
    def _detect_statistical_anomaly(self, metric_name: str, current_value: float, 
                                   historical_values: List[float]) -> Optional[AnomalyDetection]:
        """Detect statistical anomalies using z-score and business rules"""
        
        thresholds = self.anomaly_thresholds[metric_name]
        
        # Business rule checks
        if current_value < thresholds['min'] or current_value > thresholds['max']:
            return AnomalyDetection(
                metric_name=metric_name,
                current_value=current_value,
                expected_range=(thresholds['min'], thresholds['max']),
                anomaly_score=1.0,
                is_anomaly=True,
                severity="high",
                recommended_action="immediate_investigation_required"
            )
        
        # Statistical anomaly detection using z-score
        if len(historical_values) >= 5:
            mean_value = np.mean(historical_values)
            std_value = np.std(historical_values)
            
            if std_value > 0:
                z_score = abs((current_value - mean_value) / std_value)
                
                if z_score > thresholds['std_multiplier']:
                    # Calculate expected range (mean ± 2 std)
                    expected_range = (
                        mean_value - 2 * std_value,
                        mean_value + 2 * std_value
                    )
                    
                    # Determine severity
                    if z_score > thresholds['std_multiplier'] * 1.5:
                        severity = "high"
                        action = "immediate_investigation_required"
                    elif z_score > thresholds['std_multiplier'] * 1.2:
                        severity = "medium"
                        action = "monitor_closely"
                    else:
                        severity = "low"
                        action = "continue_monitoring"
                    
                    return AnomalyDetection(
                        metric_name=metric_name,
                        current_value=current_value,
                        expected_range=expected_range,
                        anomaly_score=min(z_score / thresholds['std_multiplier'], 1.0),
                        is_anomaly=True,
                        severity=severity,
                        recommended_action=action
                    )
        
        return None
    
    def _log_anomaly(self, anomaly: AnomalyDetection):
        """Log detected anomaly to database"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT INTO anomaly_log 
            (timestamp, campaign_id, metric_name, current_value, expected_value, 
             anomaly_score, severity, action_taken)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            'system',
            anomaly.metric_name,
            anomaly.current_value,
            np.mean(anomaly.expected_range),
            anomaly.anomaly_score,
            anomaly.severity,
            anomaly.recommended_action
        ))
        
        self.conn.commit()
        
        self.logger.log_observation("anomaly_detected", {
            "metric": anomaly.metric_name,
            "value": anomaly.current_value,
            "severity": anomaly.severity,
            "score": anomaly.anomaly_score
        })
    
    def create_ab_test(self, test_name: str, campaign_id: str, 
                      test_config: Dict[str, Any]) -> str:
        """Create a new A/B test"""
        test_id = f"TEST_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        ab_test = {
            'test_id': test_id,
            'test_name': test_name,
            'campaign_id': campaign_id,
            'start_date': datetime.now(),
            'end_date': datetime.now() + timedelta(days=test_config.get('duration_days', 14)),
            'traffic_split': test_config.get('traffic_split', 0.5),
            'success_metric': test_config.get('success_metric', 'conversion_rate'),
            'minimum_sample_size': test_config.get('minimum_sample_size', 1000),
            'confidence_level': test_config.get('confidence_level', 0.95),
            'variant_a_config': test_config.get('variant_a'),
            'variant_b_config': test_config.get('variant_b'),
            'status': 'running'
        }
        
        self.active_tests[test_id] = ab_test
        
        self.logger.log_action("ab_test_created", {
            "test_id": test_id,
            "test_name": test_name,
            "campaign_id": campaign_id,
            "success_metric": ab_test['success_metric']
        })
        
        return test_id
    
    def analyze_ab_test(self, test_id: str, variant_a_data: Dict[str, Any], 
                       variant_b_data: Dict[str, Any]) -> ABTestResult:
        """Analyze A/B test results for statistical significance"""
        
        if test_id not in self.active_tests:
            raise ValueError(f"Test {test_id} not found")
        
        test_config = self.active_tests[test_id]
        success_metric = test_config['success_metric']
        
        # Extract key metrics
        a_conversions = variant_a_data.get('conversions', 0)
        a_visitors = variant_a_data.get('visitors', 1)
        b_conversions = variant_b_data.get('conversions', 0)
        b_visitors = variant_b_data.get('visitors', 1)
        
        a_rate = a_conversions / a_visitors if a_visitors > 0 else 0
        b_rate = b_conversions / b_visitors if b_visitors > 0 else 0
        
        # Perform statistical significance test (Chi-square test)
        contingency_table = np.array([
            [a_conversions, a_visitors - a_conversions],
            [b_conversions, b_visitors - b_conversions]
        ])
        
        try:
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            is_significant = p_value < (1 - test_config['confidence_level'])
        except:
            is_significant = False
            p_value = 1.0
        
        # Calculate lift
        lift_percentage = ((b_rate - a_rate) / a_rate * 100) if a_rate > 0 else 0
        
        # Determine winner
        if is_significant:
            winner = "B" if b_rate > a_rate else "A"
        else:
            winner = "inconclusive"
        
        # Calculate confidence level
        confidence_level = 1 - p_value if is_significant else 0
        
        result = ABTestResult(
            test_id=test_id,
            test_name=test_config['test_name'],
            variant_a_performance={'conversion_rate': a_rate, 'conversions': a_conversions, 'visitors': a_visitors},
            variant_b_performance={'conversion_rate': b_rate, 'conversions': b_conversions, 'visitors': b_visitors},
            winner=winner,
            confidence_level=confidence_level,
            statistical_significance=is_significant,
            lift_percentage=lift_percentage,
            sample_size_a=a_visitors,
            sample_size_b=b_visitors
        )
        
        # Store result
        self._store_ab_test_result(result)
        
        # Move to completed tests
        if is_significant or (datetime.now() >= test_config['end_date']):
            self.completed_tests.append(result)
            del self.active_tests[test_id]
        
        self.logger.log_action("ab_test_analyzed", {
            "test_id": test_id,
            "winner": winner,
            "significance": is_significant,
            "lift": lift_percentage
        })
        
        return result
    
    def _store_ab_test_result(self, result: ABTestResult):
        """Store A/B test result in database"""
        cursor = self.conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO ab_test_results
            (test_id, test_name, campaign_id, start_date, end_date, 
             variant_a_data, variant_b_data, winner, confidence_level,
             statistical_significance, lift_percentage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.test_id,
            result.test_name,
            'campaign_id',  # Would get from test config
            datetime.now().isoformat(),
            datetime.now().isoformat(),
            json.dumps(result.variant_a_performance),
            json.dumps(result.variant_b_performance),
            result.winner,
            result.confidence_level,
            result.statistical_significance,
            result.lift_percentage
        ))
        
        self.conn.commit()
    
    def calculate_roi_improvement(self, baseline_metrics: Dict[str, float], 
                                 current_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Calculate ROI improvement from baseline to current performance"""
        
        improvements = {}
        
        key_metrics = ['overall_roas', 'overall_ctr', 'overall_conversion_rate', 'churn_rate']
        
        for metric in key_metrics:
            baseline_value = baseline_metrics.get(metric, 0)
            current_value = current_metrics.get(metric, 0)
            
            if baseline_value > 0:
                if metric == 'churn_rate':  # Lower is better for churn rate
                    improvement_pct = ((baseline_value - current_value) / baseline_value) * 100
                else:  # Higher is better for other metrics
                    improvement_pct = ((current_value - baseline_value) / baseline_value) * 100
                
                improvements[metric] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'improvement_percentage': round(improvement_pct, 2),
                    'absolute_change': round(current_value - baseline_value, 4)
                }
        
        # Calculate overall ROI improvement score
        roi_scores = []
        for metric, data in improvements.items():
            weight = {'overall_roas': 0.4, 'overall_ctr': 0.2, 
                     'overall_conversion_rate': 0.3, 'churn_rate': 0.1}.get(metric, 0.1)
            roi_scores.append(data['improvement_percentage'] * weight)
        
        overall_roi_improvement = sum(roi_scores)
        
        return {
            'metric_improvements': improvements,
            'overall_roi_improvement_percentage': round(overall_roi_improvement, 2),
            'performance_grade': self._calculate_improvement_grade(overall_roi_improvement),
            'recommendation': self._generate_improvement_recommendation(improvements)
        }
    
    def _calculate_improvement_grade(self, improvement_percentage: float) -> str:
        """Calculate performance improvement grade"""
        if improvement_percentage >= 20:
            return "Excellent"
        elif improvement_percentage >= 10:
            return "Good"
        elif improvement_percentage >= 5:
            return "Fair"
        elif improvement_percentage >= 0:
            return "Stable"
        else:
            return "Needs Improvement"
    
    def _generate_improvement_recommendation(self, improvements: Dict) -> str:
        """Generate recommendation based on improvement analysis"""
        recommendations = []
        
        for metric, data in improvements.items():
            improvement = data['improvement_percentage']
            
            if improvement < -10:
                recommendations.append(f"Critical: {metric} declined by {abs(improvement):.1f}% - immediate action needed")
            elif improvement < 0:
                recommendations.append(f"Warning: {metric} declined by {abs(improvement):.1f}% - monitor closely")
            elif improvement > 15:
                recommendations.append(f"Success: {metric} improved by {improvement:.1f}% - scale successful strategies")
        
        if not recommendations:
            return "Performance is stable. Continue current strategies and look for optimization opportunities."
        
        return "; ".join(recommendations[:3])  # Top 3 recommendations
    
    def generate_performance_report(self, time_period: str = "30_days") -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        # Calculate current metrics
        current_metrics = self.calculate_current_metrics()
        
        # Analyze trends
        trends = self.analyze_trends()
        
        # Detect anomalies
        anomalies = self.detect_anomalies(current_metrics)
        
        # Get A/B test results
        recent_tests = self.completed_tests[-5:]  # Last 5 tests
        
        # Calculate benchmark comparisons
        benchmark_comparisons = {}
        for metric, benchmark in self.benchmark_metrics.items():
            current_key = f'overall_{metric}' if f'overall_{metric}' in current_metrics else metric
            if current_key in current_metrics:
                current_val = current_metrics[current_key]
                benchmark_comparisons[metric] = {
                    'current': current_val,
                    'benchmark': benchmark,
                    'vs_benchmark_percentage': ((current_val - benchmark) / benchmark * 100) if benchmark > 0 else 0,
                    'status': 'above_benchmark' if current_val > benchmark else 'below_benchmark'
                }
        
        # Generate insights
        insights = self._generate_performance_insights(current_metrics, trends, anomalies)
        
        report = {
            'report_date': datetime.now().isoformat(),
            'time_period': time_period,
            'current_metrics': current_metrics,
            'benchmark_comparisons': benchmark_comparisons,
            'trends': trends,
            'anomalies': [asdict(anomaly) for anomaly in anomalies],
            'recent_ab_tests': [asdict(test) for test in recent_tests],
            'insights': insights,
            'overall_health_score': self._calculate_health_score(current_metrics, benchmark_comparisons),
            'recommendations': self._generate_actionable_recommendations(current_metrics, trends, anomalies)
        }
        
        return report
    
    def _generate_performance_insights(self, metrics: Dict, trends: Dict, 
                                     anomalies: List[AnomalyDetection]) -> List[str]:
        """Generate actionable insights from performance data"""
        insights = []
        
        # ROAS insights
        roas = metrics.get('overall_roas', 0)
        if roas > 4.0:
            insights.append("Excellent ROAS performance - consider scaling successful campaigns")
        elif roas < 2.0:
            insights.append("ROAS below target - focus on targeting and creative optimization")
        
        # Trend insights
        for metric, trend_data in trends.items():
            if trend_data['direction'] == 'increasing' and trend_data['strength'] > 0.7:
                insights.append(f"{metric} showing strong positive trend - maintain current strategies")
            elif trend_data['direction'] == 'decreasing' and trend_data['strength'] > 0.7:
                insights.append(f"{metric} declining significantly - investigate and adjust campaigns")
        
        # Anomaly insights
        high_severity_anomalies = [a for a in anomalies if a.severity == 'high']
        if high_severity_anomalies:
            insights.append(f"Critical anomalies detected in {len(high_severity_anomalies)} metrics - immediate attention required")
        
        # Channel performance insights
        channel_roas = {k: v for k, v in metrics.items() if k.endswith('_roas')}
        if channel_roas:
            best_channel = max(channel_roas, key=channel_roas.get)
            insights.append(f"{best_channel.replace('_roas', '')} is the top performing channel")
        
        return insights[:5]  # Return top 5 insights
    
    def _calculate_health_score(self, metrics: Dict, benchmark_comparisons: Dict) -> float:
        """Calculate overall system health score (0-100)"""
        score = 0
        total_weight = 0
        
        weights = {
            'roas': 0.3,
            'ctr': 0.2,
            'conversion_rate': 0.25,
            'churn_rate': 0.15,
            'active_customer_rate': 0.1
        }
        
        for metric, weight in weights.items():
            if metric in benchmark_comparisons:
                comparison = benchmark_comparisons[metric]
                if comparison['status'] == 'above_benchmark':
                    score += weight * 100
                else:
                    # Partial score based on how close to benchmark
                    ratio = comparison['current'] / comparison['benchmark'] if comparison['benchmark'] > 0 else 0
                    score += weight * min(ratio * 100, 100)
                total_weight += weight
        
        return round(score / total_weight if total_weight > 0 else 0, 1)
    
    def _generate_actionable_recommendations(self, metrics: Dict, trends: Dict, 
                                          anomalies: List[AnomalyDetection]) -> List[str]:
        """Generate specific, actionable recommendations"""
        recommendations = []
        
        # Budget optimization recommendations
        roas = metrics.get('overall_roas', 0)
        if roas > 3.5:
            recommendations.append("Scale budget on high-performing campaigns to maximize ROI")
        elif roas < 2.0:
            recommendations.append("Reduce budget on underperforming campaigns and optimize targeting")
        
        # Creative optimization recommendations
        ctr = metrics.get('overall_ctr', 0)
        if ctr < 0.02:
            recommendations.append("Refresh creative assets - current CTR indicates creative fatigue")
        
        # Channel optimization recommendations
        channel_performance = {}
        for key, value in metrics.items():
            if key.endswith('_roas'):
                channel = key.replace('_roas', '')
                channel_performance[channel] = value
        
        if channel_performance:
            best_channel = max(channel_performance, key=channel_performance.get)
            worst_channel = min(channel_performance, key=channel_performance.get)
            
            recommendations.append(f"Shift budget from {worst_channel} to {best_channel} for better ROAS")
        
        # Anomaly-based recommendations
        for anomaly in anomalies:
            if anomaly.severity == 'high':
                recommendations.append(f"Investigate {anomaly.metric_name} anomaly immediately - {anomaly.recommended_action}")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def close_connection(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()