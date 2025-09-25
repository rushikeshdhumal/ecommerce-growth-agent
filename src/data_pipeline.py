"""
Data Pipeline for E-commerce Growth Agent
Handles customer segmentation, behavioral analysis, and opportunity identification
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import sqlite3
import json
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from faker import Faker

from config.settings import settings
from config.logging_config import get_agent_logger


@dataclass
class CustomerSegment:
    """Customer segment definition"""
    segment_id: str
    name: str
    size: int
    characteristics: Dict[str, Any]
    avg_clv: float
    churn_risk: float
    recommended_channels: List[str]
    suggested_campaigns: List[str]


@dataclass
class MarketOpportunity:
    """Market opportunity identification"""
    opportunity_id: str
    opportunity_type: str
    description: str
    potential_revenue: float
    confidence_score: float
    target_segments: List[str]
    recommended_actions: List[Dict[str, Any]]
    priority: str  # high, medium, low


class DataPipeline:
    """
    Comprehensive data pipeline for customer analytics and insights generation
    """
    
    def __init__(self):
        self.logger = get_agent_logger("DataPipeline")
        self.faker = Faker()
        self.db_path = "data/ecommerce_agent.db"
        self.scaler = StandardScaler()
        
        # Initialize database and load/generate data
        self._setup_database()
        self._load_or_generate_data()
        
        self.logger.log_action("data_pipeline_initialized", {
            "customer_count": len(self.customers_df) if hasattr(self, 'customers_df') else 0
        })
    
    def _setup_database(self):
        """Setup SQLite database for data storage"""
        Path("data").mkdir(exist_ok=True)
        
        self.conn = sqlite3.connect(self.db_path)
        
        # Create tables if they don't exist
        self._create_tables()
    
    def _create_tables(self):
        """Create database tables for storing e-commerce data"""
        cursor = self.conn.cursor()
        
        # Customers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                email TEXT,
                registration_date DATE,
                last_purchase_date DATE,
                total_orders INTEGER,
                total_revenue REAL,
                avg_order_value REAL,
                days_since_last_purchase INTEGER,
                preferred_category TEXT,
                acquisition_channel TEXT,
                geographic_region TEXT,
                age_group TEXT,
                segment_id TEXT
            )
        """)
        
        # Transactions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                transaction_id TEXT PRIMARY KEY,
                customer_id TEXT,
                order_date DATE,
                order_value REAL,
                product_category TEXT,
                product_count INTEGER,
                discount_used REAL,
                channel TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            )
        """)
        
        # Campaign performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS campaign_performance (
                campaign_id TEXT PRIMARY KEY,
                campaign_name TEXT,
                channel TEXT,
                start_date DATE,
                end_date DATE,
                budget REAL,
                spend REAL,
                impressions INTEGER,
                clicks INTEGER,
                conversions INTEGER,
                revenue REAL,
                target_segment TEXT
            )
        """)
        
        self.conn.commit()
    
    def _load_or_generate_data(self):
        """Load existing data or generate mock data for the system"""
        # Check if data already exists
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM customers")
        customer_count = cursor.fetchone()[0]
        
        if customer_count < 100:  # Generate data if insufficient
            self.logger.log_action("generating_mock_data", {"target_size": settings.SAMPLE_DATA_SIZE})
            self._generate_mock_data()
        
        # Load data into DataFrames
        self.customers_df = pd.read_sql_query("SELECT * FROM customers", self.conn)
        self.transactions_df = pd.read_sql_query("SELECT * FROM transactions", self.conn)
        self.campaigns_df = pd.read_sql_query("SELECT * FROM campaign_performance", self.conn)
        
        self.logger.log_action("data_loaded", {
            "customers": len(self.customers_df),
            "transactions": len(self.transactions_df),
            "campaigns": len(self.campaigns_df)
        })
    
    def _generate_mock_data(self):
        """Generate realistic mock e-commerce data"""
        # Generate customers
        customers_data = []
        
        for i in range(settings.SAMPLE_DATA_SIZE):
            registration_date = self.faker.date_between(start_date='-2y', end_date='today')
            last_purchase = self.faker.date_between(
                start_date=registration_date, 
                end_date='today'
            ) if np.random.random() > 0.15 else None  # 15% churn rate
            
            # Calculate customer metrics
            total_orders = np.random.poisson(lam=8) + 1
            avg_order_value = np.random.normal(75, 25)
            total_revenue = total_orders * avg_order_value
            
            days_since_last = (datetime.now().date() - last_purchase).days if last_purchase else 999
            
            customer = {
                'customer_id': f"CUST_{i:06d}",
                'first_name': self.faker.first_name(),
                'last_name': self.faker.last_name(),
                'email': self.faker.email(),
                'registration_date': registration_date,
                'last_purchase_date': last_purchase,
                'total_orders': total_orders,
                'total_revenue': round(total_revenue, 2),
                'avg_order_value': round(avg_order_value, 2),
                'days_since_last_purchase': days_since_last,
                'preferred_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports']),
                'acquisition_channel': np.random.choice(['Organic', 'Paid Search', 'Social', 'Email', 'Referral'], 
                                                       p=[0.3, 0.25, 0.2, 0.15, 0.1]),
                'geographic_region': np.random.choice(['North', 'South', 'East', 'West']),
                'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+']),
                'segment_id': None  # Will be assigned during segmentation
            }
            customers_data.append(customer)
        
        # Insert customers into database
        cursor = self.conn.cursor()
        cursor.executemany("""
            INSERT OR REPLACE INTO customers VALUES (
                :customer_id, :first_name, :last_name, :email, :registration_date,
                :last_purchase_date, :total_orders, :total_revenue, :avg_order_value,
                :days_since_last_purchase, :preferred_category, :acquisition_channel,
                :geographic_region, :age_group, :segment_id
            )
        """, customers_data)
        
        # Generate transactions
        transactions_data = []
        transaction_id = 0
        
        for customer in customers_data:
            if customer['last_purchase_date']:  # Only for active customers
                for order_num in range(customer['total_orders']):
                    transaction_date = self.faker.date_between(
                        start_date=customer['registration_date'],
                        end_date=customer['last_purchase_date']
                    )
                    
                    transaction = {
                        'transaction_id': f"TXN_{transaction_id:08d}",
                        'customer_id': customer['customer_id'],
                        'order_date': transaction_date,
                        'order_value': round(np.random.normal(customer['avg_order_value'], 15), 2),
                        'product_category': customer['preferred_category'] if np.random.random() > 0.3 
                                          else np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports']),
                        'product_count': np.random.randint(1, 6),
                        'discount_used': round(np.random.exponential(5), 2) if np.random.random() > 0.7 else 0,
                        'channel': np.random.choice(['Website', 'Mobile App', 'Marketplace'])
                    }
                    transactions_data.append(transaction)
                    transaction_id += 1
        
        cursor.executemany("""
            INSERT OR REPLACE INTO transactions VALUES (
                :transaction_id, :customer_id, :order_date, :order_value,
                :product_category, :product_count, :discount_used, :channel
            )
        """, transactions_data)
        
        # Generate sample campaign data
        campaigns_data = []
        for i in range(50):  # 50 historical campaigns
            start_date = self.faker.date_between(start_date='-6m', end_date='today')
            end_date = start_date + timedelta(days=np.random.randint(7, 30))
            budget = np.random.uniform(500, 5000)
            spend = budget * np.random.uniform(0.8, 1.0)
            
            # Generate realistic metrics
            impressions = int(spend * np.random.uniform(20, 100))
            clicks = int(impressions * np.random.uniform(0.01, 0.05))
            conversions = int(clicks * np.random.uniform(0.02, 0.08))
            revenue = conversions * np.random.uniform(50, 200)
            
            campaign = {
                'campaign_id': f"CAMP_{i:04d}",
                'campaign_name': f"Campaign_{i}_{np.random.choice(['Acquisition', 'Retention', 'Winback'])}",
                'channel': np.random.choice(['Google Ads', 'Meta Ads', 'Email', 'Display']),
                'start_date': start_date,
                'end_date': end_date,
                'budget': round(budget, 2),
                'spend': round(spend, 2),
                'impressions': impressions,
                'clicks': clicks,
                'conversions': conversions,
                'revenue': round(revenue, 2),
                'target_segment': np.random.choice(['All', 'High Value', 'At Risk', 'New'])
            }
            campaigns_data.append(campaign)
        
        cursor.executemany("""
            INSERT OR REPLACE INTO campaign_performance VALUES (
                :campaign_id, :campaign_name, :channel, :start_date, :end_date,
                :budget, :spend, :impressions, :clicks, :conversions, :revenue, :target_segment
            )
        """, campaigns_data)
        
        self.conn.commit()
        self.logger.log_action("mock_data_generated", {
            "customers": len(customers_data),
            "transactions": len(transactions_data),
            "campaigns": len(campaigns_data)
        })
    
    def get_customer_segments(self) -> Dict[str, Any]:
        """Perform customer segmentation using ML clustering"""
        self.logger.log_action("customer_segmentation_started", {})
        
        # Prepare features for clustering
        features_df = self._prepare_segmentation_features()
        
        # Perform clustering
        segments = self._perform_clustering(features_df)
        
        # Analyze segments
        segment_analysis = self._analyze_segments(segments, features_df)
        
        self.logger.log_action("customer_segmentation_completed", {
            "segment_count": len(segment_analysis),
            "largest_segment_size": max([s.size for s in segment_analysis.values()])
        })
        
        return {
            "segments": {k: {
                "segment_id": v.segment_id,
                "name": v.name,
                "size": v.size,
                "characteristics": v.characteristics,
                "avg_clv": v.avg_clv,
                "churn_risk": v.churn_risk,
                "recommended_channels": v.recommended_channels,
                "suggested_campaigns": v.suggested_campaigns
            } for k, v in segment_analysis.items()},
            "total_customers": len(features_df),
            "segmentation_quality": self._calculate_segmentation_quality(features_df, segments)
        }
    
    def _prepare_segmentation_features(self) -> pd.DataFrame:
        """Prepare features for customer segmentation"""
        features = self.customers_df.copy()
        
        # Calculate RFM features (Recency, Frequency, Monetary)
        features['recency_score'] = pd.cut(features['days_since_last_purchase'], 
                                         bins=5, labels=[5,4,3,2,1]).astype(int)
        features['frequency_score'] = pd.cut(features['total_orders'], 
                                           bins=5, labels=[1,2,3,4,5]).astype(int)
        features['monetary_score'] = pd.cut(features['total_revenue'], 
                                          bins=5, labels=[1,2,3,4,5]).astype(int)
        
        # Additional behavioral features
        features['order_frequency'] = features['total_orders'] / ((datetime.now().date() - 
                                                                  pd.to_datetime(features['registration_date']).dt.date).dt.days + 1) * 365
        features['is_churned'] = (features['days_since_last_purchase'] > 90).astype(int)
        
        # Encode categorical variables
        features_encoded = pd.get_dummies(features[['recency_score', 'frequency_score', 'monetary_score',
                                                   'order_frequency', 'avg_order_value', 'is_churned',
                                                   'preferred_category', 'acquisition_channel', 'age_group']])
        
        return features_encoded
    
    def _perform_clustering(self, features_df: pd.DataFrame) -> np.ndarray:
        """Perform K-means clustering on customer features"""
        # Scale features
        features_scaled = self.scaler.fit_transform(features_df)
        
        # Determine optimal number of clusters
        optimal_k = self._find_optimal_clusters(features_scaled)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_scaled)
        
        return cluster_labels
    
    def _find_optimal_clusters(self, features_scaled: np.ndarray) -> int:
        """Find optimal number of clusters using silhouette score"""
        best_score = -1
        best_k = settings.CUSTOMER_SEGMENT_COUNT
        
        for k in range(2, min(10, len(features_scaled) // 100)):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(features_scaled)
            score = silhouette_score(features_scaled, labels)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        return min(best_k, settings.CUSTOMER_SEGMENT_COUNT)
    
    def _analyze_segments(self, segments: np.ndarray, features_df: pd.DataFrame) -> Dict[str, CustomerSegment]:
        """Analyze and characterize customer segments"""
        segment_analysis = {}
        
        for segment_id in np.unique(segments):
            segment_mask = segments == segment_id
            segment_customers = self.customers_df[segment_mask]
            segment_features = features_df[segment_mask]
            
            # Calculate segment characteristics
            characteristics = {
                'avg_recency': segment_customers['days_since_last_purchase'].mean(),
                'avg_frequency': segment_customers['total_orders'].mean(),
                'avg_monetary': segment_customers['total_revenue'].mean(),
                'avg_order_value': segment_customers['avg_order_value'].mean(),
                'dominant_category': segment_customers['preferred_category'].mode().iloc[0],
                'dominant_channel': segment_customers['acquisition_channel'].mode().iloc[0],
                'churn_rate': (segment_customers['days_since_last_purchase'] > 90).mean()
            }
            
            # Calculate Customer Lifetime Value
            avg_clv = self._calculate_clv(segment_customers)
            
            # Determine segment name based on characteristics
            segment_name = self._generate_segment_name(characteristics)
            
            # Recommend channels and campaigns
            recommended_channels = self._recommend_channels(characteristics)
            suggested_campaigns = self._suggest_campaigns(characteristics)
            
            segment_analysis[f"segment_{segment_id}"] = CustomerSegment(
                segment_id=f"segment_{segment_id}",
                name=segment_name,
                size=len(segment_customers),
                characteristics=characteristics,
                avg_clv=avg_clv,
                churn_risk=characteristics['churn_rate'],
                recommended_channels=recommended_channels,
                suggested_campaigns=suggested_campaigns
            )
            
            # Update database with segment assignments
            customer_ids = segment_customers['customer_id'].tolist()
            self._update_segment_assignments(customer_ids, f"segment_{segment_id}")
        
        return segment_analysis
    
    def _calculate_clv(self, customers: pd.DataFrame) -> float:
        """Calculate Customer Lifetime Value for a segment"""
        avg_order_value = customers['avg_order_value'].mean()
        avg_frequency = customers['total_orders'].mean()
        avg_lifespan_days = (datetime.now().date() - pd.to_datetime(customers['registration_date']).dt.date).dt.days.mean()
        avg_lifespan_years = avg_lifespan_days / 365
        
        # Simple CLV calculation
        clv = avg_order_value * avg_frequency * avg_lifespan_years
        return round(clv, 2)
    
    def _generate_segment_name(self, characteristics: Dict) -> str:
        """Generate descriptive name for customer segment"""
        if characteristics['avg_monetary'] > 1000 and characteristics['churn_rate'] < 0.2:
            return "High Value Loyal"
        elif characteristics['avg_monetary'] > 500 and characteristics['avg_recency'] < 30:
            return "Active Customers"
        elif characteristics['churn_rate'] > 0.5:
            return "At Risk / Churned"
        elif characteristics['avg_frequency'] < 2:
            return "One-time Buyers"
        elif characteristics['avg_recency'] > 60:
            return "Dormant Customers"
        else:
            return "Regular Customers"
    
    def _recommend_channels(self, characteristics: Dict) -> List[str]:
        """Recommend marketing channels based on segment characteristics"""
        channels = []
        
        if characteristics['avg_monetary'] > 500:
            channels.extend(['email', 'google_ads'])
        
        if characteristics['churn_rate'] > 0.3:
            channels.extend(['email', 'sms'])
        
        if characteristics['avg_frequency'] > 5:
            channels.append('display')
        
        if characteristics['dominant_channel'] in ['Social', 'Paid Search']:
            channels.append('meta_ads')
        
        return list(set(channels)) if channels else ['email']
    
    def _suggest_campaigns(self, characteristics: Dict) -> List[str]:
        """Suggest campaign types for segment"""
        campaigns = []
        
        if characteristics['churn_rate'] > 0.5:
            campaigns.append('winback')
        elif characteristics['avg_recency'] > 60:
            campaigns.append('retention')
        elif characteristics['avg_monetary'] > 800:
            campaigns.append('upsell')
        else:
            campaigns.append('acquisition')
        
        return campaigns
    
    def _update_segment_assignments(self, customer_ids: List[str], segment_id: str):
        """Update customer segment assignments in database"""
        cursor = self.conn.cursor()
        for customer_id in customer_ids:
            cursor.execute(
                "UPDATE customers SET segment_id = ? WHERE customer_id = ?",
                (segment_id, customer_id)
            )
        self.conn.commit()
    
    def _calculate_segmentation_quality(self, features_df: pd.DataFrame, segments: np.ndarray) -> float:
        """Calculate quality score for segmentation"""
        try:
            features_scaled = self.scaler.fit_transform(features_df)
            return round(silhouette_score(features_scaled, segments), 3)
        except:
            return 0.0
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Calculate current performance metrics across all campaigns and channels"""
        # Overall metrics
        total_spend = self.campaigns_df['spend'].sum()
        total_revenue = self.campaigns_df['revenue'].sum()
        total_conversions = self.campaigns_df['conversions'].sum()
        
        overall_roas = total_revenue / total_spend if total_spend > 0 else 0
        
        # Channel performance
        channel_performance = {}
        for channel in self.campaigns_df['channel'].unique():
            channel_data = self.campaigns_df[self.campaigns_df['channel'] == channel]
            channel_spend = channel_data['spend'].sum()
            channel_revenue = channel_data['revenue'].sum()
            
            channel_performance[channel] = {
                'spend': channel_spend,
                'revenue': channel_revenue,
                'roas': channel_revenue / channel_spend if channel_spend > 0 else 0,
                'conversions': channel_data['conversions'].sum(),
                'campaigns': len(channel_data)
            }
        
        # Customer metrics
        active_customers = len(self.customers_df[self.customers_df['days_since_last_purchase'] <= 30])
        churned_customers = len(self.customers_df[self.customers_df['days_since_last_purchase'] > 90])
        
        avg_clv = self.customers_df['total_revenue'].mean()
        avg_order_value = self.customers_df['avg_order_value'].mean()
        
        return {
            'overall_roas': round(overall_roas, 2),
            'total_spend': round(total_spend, 2),
            'total_revenue': round(total_revenue, 2),
            'total_conversions': total_conversions,
            'channel_performance': channel_performance,
            'active_customers': active_customers,
            'churned_customers': churned_customers,
            'churn_rate': round(churned_customers / len(self.customers_df), 3),
            'avg_clv': round(avg_clv, 2),
            'avg_order_value': round(avg_order_value, 2),
            'total_customers': len(self.customers_df)
        }
    
    def identify_opportunities(self) -> List[MarketOpportunity]:
        """Identify market opportunities using data analysis"""
        opportunities = []
        
        # Opportunity 1: High-value customer expansion
        high_value_segment = self.customers_df[self.customers_df['total_revenue'] > 1000]
        if len(high_value_segment) > 50:
            opportunities.append(MarketOpportunity(
                opportunity_id="OPP_001",
                opportunity_type="segment_expansion",
                description="Expand high-value customer segment through lookalike targeting",
                potential_revenue=len(high_value_segment) * 500,  # Estimated additional revenue
                confidence_score=0.8,
                target_segments=["High Value Loyal"],
                recommended_actions=[
                    {"action": "create_lookalike_campaign", "budget": 2000, "channel": "meta_ads"}
                ],
                priority="high"
            ))
        
        # Opportunity 2: Churn prevention
        at_risk_customers = self.customers_df[
            (self.customers_df['days_since_last_purchase'] > 45) & 
            (self.customers_df['days_since_last_purchase'] <= 90)
        ]
        if len(at_risk_customers) > 100:
            opportunities.append(MarketOpportunity(
                opportunity_id="OPP_002",
                opportunity_type="churn_prevention",
                description="Prevent churn of at-risk customers with targeted retention campaigns",
                potential_revenue=len(at_risk_customers) * 150,
                confidence_score=0.7,
                target_segments=["At Risk / Churned"],
                recommended_actions=[
                    {"action": "create_retention_campaign", "budget": 1500, "channel": "email"}
                ],
                priority="high"
            ))
        
        # Opportunity 3: Category cross-sell
        category_analysis = self.customers_df.groupby('preferred_category')['total_revenue'].mean()
        top_category = category_analysis.idxmax()
        opportunities.append(MarketOpportunity(
            opportunity_id="OPP_003",
            opportunity_type="cross_sell",
            description=f"Cross-sell opportunities in {top_category} category",
            potential_revenue=500 * len(self.customers_df),
            confidence_score=0.6,
            target_segments=["Active Customers", "Regular Customers"],
            recommended_actions=[
                {"action": "create_cross_sell_campaign", "budget": 1000, "category": top_category}
            ],
            priority="medium"
        ))
        
        # Opportunity 4: Seasonal optimization
        recent_campaigns = self.campaigns_df[self.campaigns_df['end_date'] >= 
                                           (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')]
        if len(recent_campaigns) > 0:
            best_performing_channel = recent_campaigns.groupby('channel')['revenue'].sum().idxmax()
            opportunities.append(MarketOpportunity(
                opportunity_id="OPP_004",
                opportunity_type="channel_optimization",
                description=f"Scale up {best_performing_channel} campaigns based on recent performance",
                potential_revenue=2000,
                confidence_score=0.75,
                target_segments=["All"],
                recommended_actions=[
                    {"action": "scale_campaign", "budget": 3000, "channel": best_performing_channel}
                ],
                priority="medium"
            ))
        
        return opportunities
    
    def get_trend_analysis(self) -> Dict[str, Any]:
        """Analyze trends in customer behavior and campaign performance"""
        # Convert date columns
        self.campaigns_df['start_date'] = pd.to_datetime(self.campaigns_df['start_date'])
        self.campaigns_df['end_date'] = pd.to_datetime(self.campaigns_df['end_date'])
        
        # Monthly performance trends
        monthly_performance = self.campaigns_df.groupby(
            self.campaigns_df['start_date'].dt.to_period('M')
        ).agg({
            'spend': 'sum',
            'revenue': 'sum',
            'conversions': 'sum'
        }).reset_index()
        
        monthly_performance['roas'] = monthly_performance['revenue'] / monthly_performance['spend']
        monthly_performance['start_date'] = monthly_performance['start_date'].astype(str)
        
        # Customer acquisition trends
        self.customers_df['registration_date'] = pd.to_datetime(self.customers_df['registration_date'])
        monthly_acquisitions = self.customers_df.groupby(
            self.customers_df['registration_date'].dt.to_period('M')
        ).size().reset_index()
        monthly_acquisitions['registration_date'] = monthly_acquisitions['registration_date'].astype(str)
        
        return {
            'monthly_performance': monthly_performance.to_dict('records'),
            'monthly_acquisitions': monthly_acquisitions.to_dict('records'),
            'performance_trend': self._calculate_trend(monthly_performance['roas'].tolist()),
            'acquisition_trend': self._calculate_trend(monthly_acquisitions[0].tolist())
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a list of values"""
        if len(values) < 2:
            return "insufficient_data"
        
        recent_avg = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
        earlier_avg = np.mean(values[:-3]) if len(values) >= 6 else values[0]
        
        change_percent = (recent_avg - earlier_avg) / earlier_avg * 100 if earlier_avg != 0 else 0
        
        if change_percent > 10:
            return "strong_positive"
        elif change_percent > 5:
            return "positive"
        elif change_percent > -5:
            return "stable"
        elif change_percent > -10:
            return "negative"
        else:
            return "strong_negative"
    
    def close_connection(self):
        """Close database connection"""
        if hasattr(self, 'conn'):
            self.conn.close()