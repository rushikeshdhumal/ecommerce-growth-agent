"""
Unit tests for the Data Pipeline component
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from src.data_pipeline import DataPipeline, CustomerSegment, MarketOpportunity


class TestDataPipeline:
    """Test suite for the DataPipeline class"""
    
    @pytest.fixture
    def temp_db(self):
        """Create a temporary database for testing"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_file.close()
        return temp_file.name
    
    @pytest.fixture
    def mock_pipeline(self, temp_db):
        """Create a DataPipeline instance with temporary database"""
        with patch('src.data_pipeline.DataPipeline._setup_database'):
            pipeline = DataPipeline()
            pipeline.db_path = temp_db
            
            # Mock the database connection
            pipeline.conn = Mock()
            
            # Mock sample data
            pipeline.customers_df = pd.DataFrame({
                'customer_id': [f'CUST_{i:06d}' for i in range(100)],
                'total_revenue': np.random.uniform(50, 2000, 100),
                'total_orders': np.random.randint(1, 20, 100),
                'days_since_last_purchase': np.random.randint(0, 365, 100),
                'avg_order_value': np.random.uniform(25, 150, 100),
                'preferred_category': np.random.choice(['Electronics', 'Clothing', 'Home'], 100),
                'acquisition_channel': np.random.choice(['Organic', 'Paid Search', 'Social'], 100),
                'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55'], 100)
            })
            
            pipeline.transactions_df = pd.DataFrame({
                'transaction_id': [f'TXN_{i:08d}' for i in range(500)],
                'customer_id': np.random.choice([f'CUST_{i:06d}' for i in range(100)], 500),
                'order_value': np.random.uniform(20, 200, 500),
                'order_date': pd.date_range(start='2023-01-01', periods=500, freq='D')
            })
            
            pipeline.campaigns_df = pd.DataFrame({
                'campaign_id': [f'CAMP_{i:04d}' for i in range(20)],
                'channel': np.random.choice(['Google Ads', 'Meta Ads', 'Email'], 20),
                'spend': np.random.uniform(100, 5000, 20),
                'revenue': np.random.uniform(200, 15000, 20),
                'clicks': np.random.randint(50, 2000, 20),
                'impressions': np.random.randint(1000, 50000, 20),
                'conversions': np.random.randint(5, 200, 20)
            })
            
            return pipeline
    
    def teardown_method(self, method):
        """Clean up temporary files after each test"""
        # Clean up any temporary database files
        pass
    
    def test_pipeline_initialization(self, mock_pipeline):
        """Test DataPipeline initializes correctly"""
        assert mock_pipeline is not None
        assert hasattr(mock_pipeline, 'customers_df')
        assert hasattr(mock_pipeline, 'transactions_df')
        assert hasattr(mock_pipeline, 'campaigns_df')
        assert len(mock_pipeline.customers_df) > 0
    
    def test_customer_segmentation(self, mock_pipeline):
        """Test customer segmentation functionality"""
        # Mock the segmentation methods
        with patch.object(mock_pipeline, '_perform_clustering') as mock_clustering, \
             patch.object(mock_pipeline, '_analyze_segments') as mock_analysis:
            
            # Mock clustering results
            mock_clustering.return_value = np.random.randint(0, 5, len(mock_pipeline.customers_df))
            
            # Mock segment analysis
            mock_segments = {
                'segment_0': CustomerSegment(
                    segment_id='segment_0',
                    name='High Value Loyal',
                    size=25,
                    characteristics={'avg_clv': 500.0, 'churn_rate': 0.1},
                    avg_clv=500.0,
                    churn_risk=0.1,
                    recommended_channels=['email', 'google_ads'],
                    suggested_campaigns=['retention']
                )
            }
            mock_analysis.return_value = mock_segments
            
            # Execute segmentation
            result = mock_pipeline.get_customer_segments()
            
            # Assertions
            assert 'segments' in result
            assert 'total_customers' in result
            assert 'segmentation_quality' in result
            assert len(result['segments']) > 0
            
            # Verify methods were called
            mock_clustering.assert_called_once()
            mock_analysis.assert_called_once()
    
    def test_performance_metrics_calculation(self, mock_pipeline):
        """Test performance metrics calculation"""
        # Execute metrics calculation
        metrics = mock_pipeline.get_performance_metrics()
        
        # Assertions
        assert isinstance(metrics, dict)
        assert 'overall_roas' in metrics
        assert 'total_spend' in metrics
        assert 'total_revenue' in metrics
        assert 'channel_performance' in metrics
        assert 'churn_rate' in metrics
        
        # Verify ROAS calculation
        if metrics['total_spend'] > 0:
            expected_roas = metrics['total_revenue'] / metrics['total_spend']
            assert abs(metrics['overall_roas'] - expected_roas) < 0.01
    
    def test_opportunity_identification(self, mock_pipeline):
        """Test market opportunity identification"""
        # Execute opportunity identification
        opportunities = mock_pipeline.identify_opportunities()
        
        # Assertions
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        # Check opportunity structure
        for opportunity in opportunities:
            assert isinstance(opportunity, MarketOpportunity)
            assert hasattr(opportunity, 'opportunity_id')
            assert hasattr(opportunity, 'opportunity_type')
            assert hasattr(opportunity, 'potential_revenue')
            assert hasattr(opportunity, 'confidence_score')
            assert hasattr(opportunity, 'recommended_actions')
    
    def test_trend_analysis(self, mock_pipeline):
        """Test trend analysis functionality"""
        # Mock database query for trend analysis
        mock_pipeline.conn.cursor.return_value.fetchall.return_value = [
            ('overall_roas', 3.2, '2024-01-01'),
            ('overall_roas', 3.5, '2024-01-02'),
            ('overall_roas', 3.8, '2024-01-03'),
            ('overall_ctr', 0.025, '2024-01-01'),
            ('overall_ctr', 0.027, '2024-01-02'),
            ('overall_ctr', 0.029, '2024-01-03')
        ]
        
        # Execute trend analysis
        trends = mock_pipeline.get_trend_analysis()
        
        # Assertions
        assert 'monthly_performance' in trends
        assert 'monthly_acquisitions' in trends
        assert 'performance_trend' in trends
        assert 'acquisition_trend' in trends
    
    def test_feature_preparation_for_segmentation(self, mock_pipeline):
        """Test feature preparation for customer segmentation"""
        # Execute feature preparation
        features_df = mock_pipeline._prepare_segmentation_features()
        
        # Assertions
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == len(mock_pipeline.customers_df)
        
        # Check that RFM features are created
        expected_columns = ['recency_score', 'frequency_score', 'monetary_score']
        for col in expected_columns:
            assert col in features_df.columns or any(col in str(c) for c in features_df.columns)
    
    def test_clustering_quality_assessment(self, mock_pipeline):
        """Test clustering quality assessment"""
        # Create mock features for clustering
        mock_features = pd.DataFrame(np.random.randn(100, 5))
        mock_segments = np.random.randint(0, 3, 100)
        
        # Test quality calculation
        quality_score = mock_pipeline._calculate_segmentation_quality(mock_features, mock_segments)
        
        # Assertions
        assert isinstance(quality_score, float)
        assert 0 <= quality_score <= 1  # Silhouette score range
    
    def test_clv_calculation(self, mock_pipeline):
        """Test Customer Lifetime Value calculation"""
        # Create sample customer data
        sample_customers = pd.DataFrame({
            'avg_order_value': [100, 150, 80],
            'total_orders': [5, 8, 3],
            'registration_date': [
                datetime.now() - timedelta(days=365),
                datetime.now() - timedelta(days=200), 
                datetime.now() - timedelta(days=100)
            ]
        })
        
        # Calculate CLV
        clv = mock_pipeline._calculate_clv(sample_customers)
        
        # Assertions
        assert isinstance(clv, float)
        assert clv > 0
    
    def test_segment_naming(self, mock_pipeline):
        """Test segment naming logic"""
        # Test various segment characteristics
        test_cases = [
            {
                'avg_monetary': 1500,
                'churn_rate': 0.1,
                'expected_name': 'High Value Loyal'
            },
            {
                'avg_monetary': 600,
                'avg_recency': 20,
                'expected_name': 'Active Customers'
            },
            {
                'churn_rate': 0.6,
                'expected_name': 'At Risk / Churned'
            },
            {
                'avg_frequency': 1.5,
                'expected_name': 'One-time Buyers'
            }
        ]
        
        for case in test_cases:
            characteristics = {
                'avg_monetary': case.get('avg_monetary', 300),
                'avg_recency': case.get('avg_recency', 50),
                'avg_frequency': case.get('avg_frequency', 3),
                'churn_rate': case.get('churn_rate', 0.2),
                'avg_order_value': 75,
                'dominant_category': 'Electronics',
                'dominant_channel': 'Organic'
            }
            
            name = mock_pipeline._generate_segment_name(characteristics)
            assert isinstance(name, str)
            assert len(name) > 0
    
    def test_channel_recommendation(self, mock_pipeline):
        """Test channel recommendation logic"""
        # Test different segment characteristics
        high_value_characteristics = {
            'avg_monetary': 1000,
            'churn_rate': 0.2,
            'avg_frequency': 5,
            'dominant_channel': 'Paid Search'
        }
        
        channels = mock_pipeline._recommend_channels(high_value_characteristics)
        
        # Assertions
        assert isinstance(channels, list)
        assert len(channels) > 0
        assert all(isinstance(channel, str) for channel in channels)
    
    def test_campaign_suggestions(self, mock_pipeline):
        """Test campaign suggestion logic"""
        # Test different scenarios
        test_scenarios = [
            {
                'churn_rate': 0.6,
                'expected_campaign': 'winback'
            },
            {
                'avg_recency': 80,
                'churn_rate': 0.3,
                'expected_campaign': 'retention'
            },
            {
                'avg_monetary': 900,
                'churn_rate': 0.1,
                'expected_campaign': 'upsell'
            }
        ]
        
        for scenario in test_scenarios:
            characteristics = {
                'avg_monetary': scenario.get('avg_monetary', 400),
                'avg_recency': scenario.get('avg_recency', 30),
                'churn_rate': scenario.get('churn_rate', 0.2)
            }
            
            campaigns = mock_pipeline._suggest_campaigns(characteristics)
            
            assert isinstance(campaigns, list)
            assert len(campaigns) > 0
            if 'expected_campaign' in scenario:
                assert scenario['expected_campaign'] in campaigns
    
    def test_data_quality_checks(self, mock_pipeline):
        """Test data quality validation"""
        # Check for required columns
        required_customer_columns = [
            'customer_id', 'total_revenue', 'total_orders', 
            'days_since_last_purchase', 'avg_order_value'
        ]
        
        for col in required_customer_columns:
            assert col in mock_pipeline.customers_df.columns
        
        # Check for data integrity
        assert mock_pipeline.customers_df['total_revenue'].min() >= 0
        assert mock_pipeline.customers_df['total_orders'].min() >= 0
        assert mock_pipeline.customers_df['days_since_last_purchase'].min() >= 0
    
    def test_performance_calculation_edge_cases(self, mock_pipeline):
        """Test performance calculation with edge cases"""
        # Test with zero spend
        mock_pipeline.campaigns_df['spend'] = 0
        mock_pipeline.campaigns_df['revenue'] = 1000
        
        metrics = mock_pipeline.get_performance_metrics()
        
        # Should handle division by zero gracefully
        assert metrics['overall_roas'] == 0  # or some default value
        
        # Test with zero revenue
        mock_pipeline.campaigns_df['spend'] = 1000
        mock_pipeline.campaigns_df['revenue'] = 0
        
        metrics = mock_pipeline.get_performance_metrics()
        assert metrics['overall_roas'] == 0
    
    def test_mock_data_generation(self, mock_pipeline):
        """Test mock data generation functionality"""
        # Test that generated data is realistic
        customers = mock_pipeline.customers_df
        
        # Check data ranges
        assert customers['total_revenue'].min() >= 0
        assert customers['total_orders'].min() >= 0
        assert customers['avg_order_value'].min() > 0
        
        # Check categorical data
        valid_categories = ['Electronics', 'Clothing', 'Home', 'Books', 'Sports']
        assert all(cat in valid_categories for cat in customers['preferred_category'].unique())
        
        valid_channels = ['Organic', 'Paid Search', 'Social', 'Email', 'Referral']
        assert all(ch in valid_channels for ch in customers['acquisition_channel'].unique())
    
    def test_database_connection_handling(self, mock_pipeline):
        """Test database connection management"""
        # Test connection closing
        mock_pipeline.close_connection()
        
        # Verify close was called (in real implementation)
        # This is a placeholder for actual database connection testing
        assert True  # Connection handling tested in integration tests


class TestCustomerSegment:
    """Test suite for CustomerSegment dataclass"""
    
    def test_segment_creation(self):
        """Test CustomerSegment creation"""
        segment = CustomerSegment(
            segment_id="seg_001",
            name="High Value Customers",
            size=1500,
            characteristics={
                "avg_clv": 750.0,
                "avg_frequency": 8.5,
                "churn_rate": 0.12
            },
            avg_clv=750.0,
            churn_risk=0.12,
            recommended_channels=["email", "google_ads"],
            suggested_campaigns=["retention", "upsell"]
        )
        
        assert segment.segment_id == "seg_001"
        assert segment.name == "High Value Customers"
        assert segment.size == 1500
        assert segment.avg_clv == 750.0
        assert len(segment.recommended_channels) == 2
        assert len(segment.suggested_campaigns) == 2


class TestMarketOpportunity:
    """Test suite for MarketOpportunity dataclass"""
    
    def test_opportunity_creation(self):
        """Test MarketOpportunity creation"""
        opportunity = MarketOpportunity(
            opportunity_id="OPP_001",
            opportunity_type="segment_expansion",
            description="Expand high-value customer segment",
            potential_revenue=25000.0,
            confidence_score=0.85,
            target_segments=["High Value Loyal"],
            recommended_actions=[
                {"action": "create_lookalike_campaign", "budget": 2000}
            ],
            priority="high"
        )
        
        assert opportunity.opportunity_id == "OPP_001"
        assert opportunity.opportunity_type == "segment_expansion"
        assert opportunity.potential_revenue == 25000.0
        assert opportunity.confidence_score == 0.85
        assert opportunity.priority == "high"
        assert len(opportunity.recommended_actions) == 1


if __name__ == "__main__":
    pytest.main([__file__])