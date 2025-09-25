"""
Unit tests for the Campaign Manager component
"""
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from src.campaign_manager import CampaignManager, Campaign, CreativeAsset
from src.integrations.google_ads_mock import GoogleAdsMock
from src.integrations.meta_ads_mock import MetaAdsMock
from src.integrations.klaviyo_mock import KlaviyoMock


class TestCampaignManager:
    """Test suite for the CampaignManager class"""
    
    @pytest.fixture
    def mock_campaign_manager(self):
        """Create a CampaignManager instance with mocked dependencies"""
        with patch('src.campaign_manager.openai') as mock_openai, \
             patch('src.campaign_manager.anthropic') as mock_anthropic, \
             patch('src.campaign_manager.GoogleAdsMock') as mock_gads, \
             patch('src.campaign_manager.MetaAdsMock') as mock_meta, \
             patch('src.campaign_manager.KlaviyoMock') as mock_klaviyo:
            
            # Mock AI clients
            mock_openai.api_key = "test_key"
            mock_anthropic.return_value = Mock()
            
            # Mock platform clients
            mock_gads.return_value = Mock()
            mock_meta.return_value = Mock()
            mock_klaviyo.return_value = Mock()
            
            manager = CampaignManager()
            return manager
    
    @pytest.fixture
    def sample_campaign_data(self):
        """Sample campaign data for testing"""
        return {
            'campaign_type': 'acquisition',
            'target_segment': 'High Value Prospects',
            'budget': 2000.0,
            'channels': ['google_ads', 'meta_ads'],
            'custom_objectives': ['increase_conversions', 'improve_roas']
        }
    
    @pytest.fixture
    def sample_creative_assets(self):
        """Sample creative assets for testing"""
        return [
            CreativeAsset(
                asset_id="ASSET_001",
                asset_type="search_ad",
                content='{"headline": "Summer Sale", "description": "Up to 50% off!"}',
                metadata={"channel": "google_ads", "campaign_type": "acquisition"},
                performance_score=0.0,
                a_b_test_variant="A"
            ),
            CreativeAsset(
                asset_id="ASSET_002", 
                asset_type="news_feed_ad",
                content='{"primary_text": "Don\'t miss out!", "headline": "Shop Now"}',
                metadata={"channel": "meta_ads", "campaign_type": "acquisition"},
                performance_score=0.0,
                a_b_test_variant="A"
            )
        ]
    
    def test_campaign_manager_initialization(self, mock_campaign_manager):
        """Test CampaignManager initializes correctly"""
        assert mock_campaign_manager is not None
        assert hasattr(mock_campaign_manager, 'active_campaigns')
        assert hasattr(mock_campaign_manager, 'campaign_templates')
        assert hasattr(mock_campaign_manager, 'platform_clients')
        assert len(mock_campaign_manager.active_campaigns) == 0
    
    def test_ai_client_setup(self, mock_campaign_manager):
        """Test AI client setup"""
        with patch('src.campaign_manager.settings') as mock_settings:
            mock_settings.OPENAI_API_KEY = "test_openai_key"
            mock_settings.ANTHROPIC_API_KEY = "test_anthropic_key"
            
            # Re-initialize to test setup
            mock_campaign_manager._setup_ai_clients()
            
            # Verify clients are set up
            assert hasattr(mock_campaign_manager, 'openai_client')
    
    @patch('src.campaign_manager.uuid.uuid4')
    def test_create_campaign_success(self, mock_uuid, mock_campaign_manager, sample_campaign_data):
        """Test successful campaign creation"""
        # Mock UUID generation
        mock_uuid.return_value.hex = "abcd1234"
        
        # Mock platform client responses
        mock_campaign_manager.google_ads.create_campaign.return_value = {
            'success': True,
            'platform_campaign_id': 'GADS_TEST_001'
        }
        mock_campaign_manager.meta_ads.create_campaign.return_value = {
            'success': True,
            'platform_campaign_id': 'META_TEST_001'
        }
        
        # Mock creative generation
        with patch.object(mock_campaign_manager, '_generate_creative_assets') as mock_creative:
            mock_creative.return_value = [
                CreativeAsset(
                    asset_id="ASSET_001",
                    asset_type="search_ad",
                    content='{"headline": "Test Ad"}',
                    metadata={},
                    performance_score=0.0,
                    a_b_test_variant="A"
                )
            ]
            
            # Mock strategy generation
            with patch.object(mock_campaign_manager, '_generate_campaign_strategy') as mock_strategy:
                mock_strategy.return_value = {
                    'positioning': {'key_message': 'Test message'},
                    'value_propositions': {'google_ads': ['quality', 'price']}
                }
                
                # Create campaign
                result = mock_campaign_manager.create_campaign(**sample_campaign_data)
                
                # Assertions
                assert result['success'] is True
                assert 'campaign_id' in result
                assert result['campaign_id'] == 'CAMP_ABCD1234'
                assert len(result['channels_created']) == 2
                assert 'google_ads' in result['channels_created']
                assert 'meta_ads' in result['channels_created']
    
    def test_create_campaign_partial_failure(self, mock_campaign_manager, sample_campaign_data):
        """Test campaign creation with partial platform failures"""
        # Mock one success, one failure
        mock_campaign_manager.google_ads.create_campaign.return_value = {
            'success': True,
            'platform_campaign_id': 'GADS_TEST_001'
        }
        mock_campaign_manager.meta_ads.create_campaign.return_value = {
            'success': False,
            'error': 'Budget too low'
        }
        
        with patch.object(mock_campaign_manager, '_generate_creative_assets') as mock_creative, \
             patch.object(mock_campaign_manager, '_generate_campaign_strategy') as mock_strategy:
            
            mock_creative.return_value = []
            mock_strategy.return_value = {'positioning': {}, 'value_propositions': {}}
            
            result = mock_campaign_manager.create_campaign(**sample_campaign_data)
            
            # Should succeed with partial results
            assert result['success'] is True
            assert len(result['channels_created']) == 1
            assert 'google_ads' in result['channels_created']
            assert 'meta_ads' not in result['channels_created']
    
    def test_create_campaign_complete_failure(self, mock_campaign_manager, sample_campaign_data):
        """Test campaign creation with complete platform failure"""
        # Mock all failures
        mock_campaign_manager.google_ads.create_campaign.return_value = {
            'success': False,
            'error': 'API error'
        }
        mock_campaign_manager.meta_ads.create_campaign.return_value = {
            'success': False,
            'error': 'Invalid credentials'
        }
        
        with patch.object(mock_campaign_manager, '_generate_creative_assets') as mock_creative, \
             patch.object(mock_campaign_manager, '_generate_campaign_strategy') as mock_strategy:
            
            mock_creative.return_value = []
            mock_strategy.return_value = {'positioning': {}, 'value_propositions': {}}
            
            result = mock_campaign_manager.create_campaign(**sample_campaign_data)
            
            # Should fail
            assert result['success'] is False
            assert 'error' in result
    
    def test_budget_allocation_calculation(self, mock_campaign_manager):
        """Test budget allocation across channels"""
        total_budget = 1000.0
        channels = ['google_ads', 'meta_ads']
        template = {
            'budget_allocation': {
                'google_ads': 0.6,
                'meta_ads': 0.4
            }
        }
        
        allocation = mock_campaign_manager._calculate_budget_allocation(
            total_budget, channels, template
        )
        
        assert allocation['google_ads'] == 600.0
        assert allocation['meta_ads'] == 400.0
        assert sum(allocation.values()) == total_budget
    
    def test_budget_allocation_no_template(self, mock_campaign_manager):
        """Test budget allocation without template"""
        total_budget = 1000.0
        channels = ['google_ads', 'meta_ads']
        template = {}
        
        with patch.object(mock_campaign_manager, '_get_channel_performance_weights') as mock_weights:
            mock_weights.return_value = {
                'google_ads': 0.7,
                'meta_ads': 0.3
            }
            
            allocation = mock_campaign_manager._calculate_budget_allocation(
                total_budget, channels, template
            )
            
            assert allocation['google_ads'] == 700.0
            assert allocation['meta_ads'] == 300.0
    
    @patch('src.campaign_manager.CampaignManager._call_llm')
    def test_campaign_strategy_generation(self, mock_llm, mock_campaign_manager):
        """Test campaign strategy generation"""
        mock_strategy = {
            'positioning': {'key_message': 'Best deals online'},
            'value_propositions': {
                'google_ads': ['quality', 'price', 'convenience'],
                'meta_ads': ['trendy', 'social', 'engaging']
            },
            'audience_insights': {
                'demographics': {'age_range': '25-45'},
                'interests': ['shopping', 'fashion'],
                'behaviors': ['online_shoppers']
            }
        }
        
        mock_llm.return_value = json.dumps(mock_strategy)
        
        result = mock_campaign_manager._generate_campaign_strategy(
            'acquisition', 'Young Adults', 2000, ['google_ads', 'meta_ads']
        )
        
        assert result['positioning']['key_message'] == 'Best deals online'
        assert len(result['value_propositions']['google_ads']) == 3
        assert 'demographics' in result['audience_insights']
    
    @patch('src.campaign_manager.CampaignManager._call_llm')
    def test_campaign_strategy_generation_fallback(self, mock_llm, mock_campaign_manager):
        """Test campaign strategy generation with LLM failure"""
        mock_llm.side_effect = Exception("API Error")
        
        result = mock_campaign_manager._generate_campaign_strategy(
            'acquisition', 'Young Adults', 2000, ['google_ads']
        )
        
        # Should return fallback strategy
        assert 'positioning' in result
        assert 'value_propositions' in result
        assert 'audience_insights' in result
    
    @patch('src.campaign_manager.CampaignManager._call_llm')
    def test_creative_asset_generation(self, mock_llm, mock_campaign_manager):
        """Test creative asset generation"""
        mock_llm.return_value = json.dumps({
            'headline': 'Amazing Summer Sale',
            'description': 'Up to 50% off everything!',
            'call_to_action': 'Shop Now'
        })
        
        strategy = {
            'positioning': {'key_message': 'Best deals'},
            'value_propositions': {'google_ads': ['quality']}
        }
        
        assets = mock_campaign_manager._generate_creative_assets(
            'google_ads', 'acquisition', 'Young Adults', strategy
        )
        
        assert len(assets) > 0
        assert assets[0].asset_type in ['search_ad', 'display_ad']
        assert assets[0].metadata['channel'] == 'google_ads'
        assert assets[0].a_b_test_variant in ['A', 'B']
    
    def test_creative_asset_generation_fallback(self, mock_campaign_manager):
        """Test creative asset generation with LLM failure"""
        with patch.object(mock_campaign_manager, '_call_llm') as mock_llm:
            mock_llm.side_effect = Exception("API Error")
            
            strategy = {'positioning': {}, 'value_propositions': {}}
            
            assets = mock_campaign_manager._generate_creative_assets(
                'google_ads', 'acquisition', 'Young Adults', strategy
            )
            
            # Should return fallback assets
            assert len(assets) > 0
            assert assets[0].metadata.get('fallback') is True
    
    def test_creative_specs_retrieval(self, mock_campaign_manager):
        """Test creative specifications for different channels"""
        # Test Google Ads specs
        google_specs = mock_campaign_manager._get_channel_creative_specs('google_ads')
        assert len(google_specs) > 0
        assert any(spec['type'] == 'search_ad' for spec in google_specs)
        
        # Test Meta Ads specs
        meta_specs = mock_campaign_manager._get_channel_creative_specs('meta_ads')
        assert len(meta_specs) > 0
        assert any(spec['type'] == 'news_feed_ad' for spec in meta_specs)
        
        # Test Email specs
        email_specs = mock_campaign_manager._get_channel_creative_specs('email')
        assert len(email_specs) > 0
        assert any(spec['type'] == 'email_campaign' for spec in email_specs)
    
    def test_targeting_parameters_creation(self, mock_campaign_manager):
        """Test targeting parameter creation"""
        strategy = {
            'audience_insights': {
                'demographics': {'age_range': '25-45'},
                'interests': ['shopping', 'fashion'],
                'behaviors': ['online_shoppers'],
                'geographic': 'United States'
            }
        }
        
        # Test Google Ads targeting
        google_targeting = mock_campaign_manager._create_targeting_parameters(
            'google_ads', 'Young Adults', strategy
        )
        
        assert 'keywords' in google_targeting
        assert 'match_types' in google_targeting
        assert google_targeting['target_segment'] == 'Young Adults'
        
        # Test Meta Ads targeting
        meta_targeting = mock_campaign_manager._create_targeting_parameters(
            'meta_ads', 'Young Adults', strategy
        )
        
        assert 'lookalike_audiences' in meta_targeting
        assert 'placement' in meta_targeting
        assert meta_targeting['target_segment'] == 'Young Adults'
    
    def test_keyword_generation(self, mock_campaign_manager):
        """Test keyword generation for Google Ads"""
        keywords = mock_campaign_manager._generate_keywords(
            'high_value_customers', 
            {'audience_insights': {'interests': ['luxury', 'premium']}}
        )
        
        assert len(keywords) > 0
        assert 'premium' in keywords or 'luxury' in keywords
        assert 'ecommerce' in keywords  # Base keyword
    
    def test_email_filters_creation(self, mock_campaign_manager):
        """Test email segmentation filters"""
        # High value segment
        hv_filters = mock_campaign_manager._create_email_filters('high_value_customers')
        assert hv_filters['purchase_history'] == 'high_value'
        
        # Churned segment
        churned_filters = mock_campaign_manager._create_email_filters('churned_customers')
        assert churned_filters['engagement_level'] == 'low'
        assert 'last_purchase' in churned_filters
    
    def test_platform_campaign_creation(self, mock_campaign_manager, sample_creative_assets):
        """Test platform-specific campaign creation"""
        mock_campaign_manager.google_ads.create_campaign.return_value = {
            'success': True,
            'platform_campaign_id': 'GADS_TEST_001'
        }
        
        campaign_data = {
            'name': 'Test Campaign',
            'budget': 1000.0,
            'targeting': {'keywords': ['test']},
            'creative_assets': [asset.__dict__ for asset in sample_creative_assets]
        }
        
        result = mock_campaign_manager._create_platform_campaign(
            'google_ads', 'CAMP_001', 1000.0, {'keywords': ['test']}, sample_creative_assets
        )
        
        assert result['success'] is True
        assert 'platform_campaign_id' in result
    
    def test_get_campaign_performance(self, mock_campaign_manager):
        """Test campaign performance retrieval"""
        # Create a test campaign
        campaign_id = 'CAMP_TEST_001'
        campaign = Campaign(
            campaign_id=campaign_id,
            name='Test Campaign',
            campaign_type='acquisition',
            channel='multi_channel',
            target_segment='Test Segment',
            budget=1000.0,
            daily_budget=33.33,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            status='active',
            objectives=['test'],
            targeting_parameters={
                'channels': {
                    'google_ads': {'platform_campaign_id': 'GADS_001'},
                    'meta_ads': {'platform_campaign_id': 'META_001'}
                }
            },
            creative_assets=[],
            performance_metrics={},
            optimization_history=[]
        )
        
        mock_campaign_manager.active_campaigns[campaign_id] = campaign
        
        # Mock platform responses
        mock_campaign_manager.google_ads.get_campaign_metrics.return_value = {
            'spend': 500.0,
            'revenue': 1500.0,
            'clicks': 100,
            'impressions': 5000,
            'conversions': 15
        }
        
        mock_campaign_manager.meta_ads.get_campaign_metrics.return_value = {
            'spend': 400.0,
            'revenue': 1200.0,
            'clicks': 80,
            'impressions': 4000,
            'conversions': 12
        }
        
        performance = mock_campaign_manager.get_campaign_performance(campaign_id)
        
        assert 'overall_metrics' in performance
        assert performance['overall_metrics']['spend'] == 900.0  # 500 + 400
        assert performance['overall_metrics']['revenue'] == 2700.0  # 1500 + 1200
        assert performance['overall_metrics']['roas'] == 3.0  # 2700 / 900
        assert len(performance['channel_performance']) == 2
    
    def test_campaign_optimization(self, mock_campaign_manager):
        """Test campaign optimization"""
        # Create test campaign with performance data
        campaign_id = 'CAMP_TEST_001'
        campaign = Campaign(
            campaign_id=campaign_id,
            name='Test Campaign',
            campaign_type='acquisition',
            channel='multi_channel',
            target_segment='Test Segment',
            budget=1000.0,
            daily_budget=33.33,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            status='active',
            objectives=[],
            targeting_parameters={'channels': {'google_ads': {'platform_campaign_id': 'GADS_001'}}},
            creative_assets=[],
            performance_metrics={'roas': 4.5, 'ctr': 0.025, 'conversion_rate': 0.035},
            optimization_history=[]
        )
        
        mock_campaign_manager.active_campaigns[campaign_id] = campaign
        
        # Test budget increase optimization
        optimization_action = {
            'type': 'increase_budget',
            'factor': 1.2,
            'reasoning': 'High ROAS indicates scaling opportunity'
        }
        
        mock_campaign_manager.google_ads.update_campaign_budget.return_value = {
            'success': True
        }
        
        result = mock_campaign_manager.optimize_campaign(campaign_id, optimization_action)
        
        assert result['success'] is True
        assert result['action'] == 'budget_increased'
        assert result['new_budget'] == 1200.0  # 1000 * 1.2
        assert len(campaign.optimization_history) == 1
    
    def test_performance_grade_calculation(self, mock_campaign_manager):
        """Test performance grade calculation"""
        # Excellent performance
        excellent_metrics = {'roas': 4.5, 'ctr': 0.06, 'conversion_rate': 0.09}
        grade = mock_campaign_manager._calculate_performance_grade(excellent_metrics)
        assert grade == 'A'
        
        # Poor performance  
        poor_metrics = {'roas': 0.5, 'ctr': 0.005, 'conversion_rate': 0.01}
        grade = mock_campaign_manager._calculate_performance_grade(poor_metrics)
        assert grade == 'F'
        
        # Medium performance
        medium_metrics = {'roas': 2.5, 'ctr': 0.025, 'conversion_rate': 0.04}
        grade = mock_campaign_manager._calculate_performance_grade(medium_metrics)
        assert grade in ['B', 'C']
    
    def test_optimization_opportunities_identification(self, mock_campaign_manager):
        """Test identification of optimization opportunities"""
        # Low ROAS scenario
        low_roas_metrics = {'roas': 1.5, 'ctr': 0.03, 'conversion_rate': 0.04}
        channel_performance = {'google_ads': {'roas': 1.5}}
        
        opportunities = mock_campaign_manager._identify_optimization_opportunities(
            low_roas_metrics, channel_performance
        )
        
        assert len(opportunities) > 0
        assert any(opp['type'] == 'improve_roas' for opp in opportunities)
        
        # High performing channel scenario
        high_perf_metrics = {'roas': 3.0, 'ctr': 0.03, 'conversion_rate': 0.04}
        channel_performance = {'google_ads': {'roas': 4.5}, 'meta_ads': {'roas': 2.0}}
        
        opportunities = mock_campaign_manager._identify_optimization_opportunities(
            high_perf_metrics, channel_performance
        )
        
        scaling_opps = [opp for opp in opportunities if opp['type'] == 'scale_top_channel']
        assert len(scaling_opps) > 0
        assert scaling_opps[0]['channel'] == 'google_ads'
    
    def test_creative_refresh_optimization(self, mock_campaign_manager):
        """Test creative refresh optimization"""
        # Create campaign with underperforming creatives
        campaign_id = 'CAMP_TEST_001'
        low_performing_asset = CreativeAsset(
            asset_id="ASSET_LOW",
            asset_type="search_ad",
            content='{"headline": "Old Ad"}',
            metadata={'channel': 'google_ads'},
            performance_score=0.3,  # Low performance
            a_b_test_variant="A"
        )
        
        campaign = Campaign(
            campaign_id=campaign_id,
            name='Test Campaign',
            campaign_type='acquisition',
            channel='multi_channel',
            target_segment='Test Segment',
            budget=1000.0,
            daily_budget=33.33,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            status='active',
            objectives=[],
            targeting_parameters={},
            creative_assets=[low_performing_asset],
            performance_metrics={},
            optimization_history=[]
        )
        
        mock_campaign_manager.active_campaigns[campaign_id] = campaign
        
        # Mock LLM response for creative refresh
        with patch.object(mock_campaign_manager, '_call_llm') as mock_llm:
            mock_llm.return_value = '{"headline": "Fresh New Ad", "description": "Better copy!"}'
            
            optimization_action = {
                'type': 'refresh_creative',
                'reasoning': 'Low CTR indicates creative fatigue'
            }
            
            result = mock_campaign_manager.optimize_campaign(campaign_id, optimization_action)
            
            assert result['success'] is True
            assert result['action'] == 'creative_refreshed'
            assert result['assets_refreshed'] > 0
            assert len(campaign.creative_assets) > 1  # New assets added
    
    def test_budget_optimization_constraints(self, mock_campaign_manager):
        """Test budget optimization respects constraints"""
        campaign_id = 'CAMP_TEST_001'
        
        # Create campaign at maximum budget
        from config.settings import settings
        max_budget = settings.MAX_DAILY_BUDGET * 30
        
        campaign = Campaign(
            campaign_id=campaign_id,
            name='Test Campaign',
            campaign_type='acquisition',
            channel='multi_channel',
            target_segment='Test Segment',
            budget=max_budget * 0.9,  # Close to max
            daily_budget=max_budget * 0.9 / 30,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            status='active',
            objectives=[],
            targeting_parameters={'channels': {'google_ads': {'platform_campaign_id': 'GADS_001'}}},
            creative_assets=[],
            performance_metrics={},
            optimization_history=[]
        )
        
        mock_campaign_manager.active_campaigns[campaign_id] = campaign
        
        # Try to increase budget beyond maximum
        optimization_action = {
            'type': 'increase_budget',
            'factor': 1.5,  # Would exceed maximum
            'reasoning': 'High performance'
        }
        
        mock_campaign_manager.google_ads.update_campaign_budget.return_value = {'success': True}
        
        result = mock_campaign_manager.optimize_budget_increase(campaign, optimization_action)
        
        # Should be capped at maximum
        assert result['new_budget'] <= max_budget
    
    def test_get_all_campaigns(self, mock_campaign_manager):
        """Test getting all campaigns overview"""
        # Add test campaigns
        campaign1 = Campaign(
            campaign_id='CAMP_001',
            name='Campaign 1',
            campaign_type='acquisition',
            channel='google_ads',
            target_segment='Segment 1',
            budget=1000.0,
            daily_budget=33.33,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            status='active',
            objectives=[],
            targeting_parameters={},
            creative_assets=[],
            performance_metrics={'roas': 3.5, 'spend': 500.0},
            optimization_history=[]
        )
        
        campaign2 = Campaign(
            campaign_id='CAMP_002',
            name='Campaign 2',
            campaign_type='retention',
            channel='email',
            target_segment='Segment 2',
            budget=500.0,
            daily_budget=16.67,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            status='paused',
            objectives=[],
            targeting_parameters={},
            creative_assets=[],
            performance_metrics={'roas': 2.8, 'spend': 250.0},
            optimization_history=[]
        )
        
        mock_campaign_manager.active_campaigns['CAMP_001'] = campaign1
        mock_campaign_manager.active_campaigns['CAMP_002'] = campaign2
        
        overview = mock_campaign_manager.get_all_campaigns()
        
        assert overview['active_campaigns'] == 2
        assert len(overview['campaigns']) == 2
        assert 'CAMP_001' in overview['campaigns']
        assert 'CAMP_002' in overview['campaigns']
        assert overview['campaigns']['CAMP_001']['type'] == 'acquisition'
        assert overview['campaigns']['CAMP_002']['status'] == 'paused'
    
    def test_fallback_strategies(self, mock_campaign_manager):
        """Test fallback strategies when AI generation fails"""
        # Test fallback campaign strategy
        fallback_strategy = mock_campaign_manager._create_fallback_strategy(
            'acquisition', 'Test Segment', ['google_ads']
        )
        
        assert 'positioning' in fallback_strategy
        assert 'value_propositions' in fallback_strategy
        assert 'audience_insights' in fallback_strategy
        
        # Test fallback creative
        spec = {'type': 'search_ad', 'format': 'text'}
        fallback_creative = mock_campaign_manager._create_fallback_creative(
            spec, 'google_ads', 'acquisition'
        )
        
        assert fallback_creative.asset_type == 'search_ad'
        assert fallback_creative.metadata['channel'] == 'google_ads'
        assert fallback_creative.metadata.get('fallback') is True
    
    def test_creative_asset_aggregation(self, mock_campaign_manager, sample_creative_assets):
        """Test creative asset aggregation from multiple channels"""
        channel_campaigns = {
            'google_ads': {
                'creative_assets': sample_creative_assets[:1]
            },
            'meta_ads': {
                'creative_assets': sample_creative_assets[1:]
            }
        }
        
        aggregated = mock_campaign_manager._aggregate_creative_assets(channel_campaigns)
        
        assert len(aggregated) == 2
        assert aggregated[0].metadata['channel'] == 'google_ads'
        assert aggregated[1].metadata['channel'] == 'meta_ads'
    
    def test_campaign_manager_error_handling(self, mock_campaign_manager):
        """Test error handling in campaign manager"""
        # Test campaign creation with invalid data
        invalid_data = {
            'campaign_type': 'invalid_type',
            'target_segment': '',
            'budget': -1000,  # Invalid budget
            'channels': []  # No channels
        }
        
        result = mock_campaign_manager.create_campaign(**invalid_data)
        
        # Should handle gracefully
        assert result['success'] is False
        assert 'error' in result
    
    def test_performance_tracking_edge_cases(self, mock_campaign_manager):
        """Test performance tracking with edge cases"""
        campaign_id = 'CAMP_EDGE_001'
        
        # Campaign with no platform campaigns
        campaign = Campaign(
            campaign_id=campaign_id,
            name='Edge Case Campaign',
            campaign_type='test',
            channel='multi_channel',
            target_segment='Test',
            budget=1000.0,
            daily_budget=33.33,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            status='active',
            objectives=[],
            targeting_parameters={'channels': {}},  # No channels
            creative_assets=[],
            performance_metrics={},
            optimization_history=[]
        )
        
        mock_campaign_manager.active_campaigns[campaign_id] = campaign
        
        performance = mock_campaign_manager.get_campaign_performance(campaign_id)
        
        # Should handle gracefully with zero metrics
        assert 'overall_metrics' in performance
        assert performance['overall_metrics']['spend'] == 0
        assert performance['overall_metrics']['roas'] == 0


class TestCreativeAsset:
    """Test suite for CreativeAsset dataclass"""
    
    def test_creative_asset_creation(self):
        """Test CreativeAsset creation"""
        asset = CreativeAsset(
            asset_id="ASSET_TEST_001",
            asset_type="search_ad",
            content='{"headline": "Test Ad", "description": "Test description"}',
            metadata={"channel": "google_ads", "campaign_type": "acquisition"},
            performance_score=0.75,
            a_b_test_variant="B"
        )
        
        assert asset.asset_id == "ASSET_TEST_001"
        assert asset.asset_type == "search_ad"
        assert asset.performance_score == 0.75
        assert asset.a_b_test_variant == "B"
        assert asset.metadata["channel"] == "google_ads"
    
    def test_creative_asset_serialization(self):
        """Test CreativeAsset serialization"""
        asset = CreativeAsset(
            asset_id="ASSET_TEST_002",
            asset_type="email_campaign",
            content='{"subject": "Newsletter", "body": "Content"}',
            metadata={"channel": "email"},
            performance_score=0.0,
            a_b_test_variant="A"
        )
        
        # Test conversion to dict (for API responses)
        asset_dict = {
            'asset_id': asset.asset_id,
            'asset_type': asset.asset_type,
            'content': asset.content,
            'metadata': asset.metadata,
            'performance_score': asset.performance_score,
            'a_b_test_variant': asset.a_b_test_variant
        }
        
        assert asset_dict['asset_id'] == "ASSET_TEST_002"
        assert asset_dict['asset_type'] == "email_campaign"


class TestCampaign:
    """Test suite for Campaign dataclass"""
    
    def test_campaign_creation(self):
        """Test Campaign creation"""
        campaign = Campaign(
            campaign_id="CAMP_TEST_001",
            name="Test Campaign",
            campaign_type="acquisition",
            channel="multi_channel",
            target_segment="Young Adults",
            budget=2000.0,
            daily_budget=66.67,
            start_date=datetime(2024, 1, 15),
            end_date=datetime(2024, 2, 14),
            status="active",
            objectives=["increase_conversions", "improve_brand_awareness"],
            targeting_parameters={"demographics": {"age_range": "18-35"}},
            creative_assets=[],
            performance_metrics={"roas": 3.2, "ctr": 0.025},
            optimization_history=[]
        )
        
        assert campaign.campaign_id == "CAMP_TEST_001"
        assert campaign.campaign_type == "acquisition"
        assert campaign.budget == 2000.0
        assert campaign.status == "active"
        assert len(campaign.objectives) == 2
        assert campaign.performance_metrics["roas"] == 3.2
    
    def test_campaign_with_assets(self):
        """Test Campaign with creative assets"""
        asset = CreativeAsset(
            asset_id="ASSET_001",
            asset_type="search_ad",
            content='{"headline": "Test"}',
            metadata={},
            performance_score=0.8,
            a_b_test_variant="A"
        )
        
        campaign = Campaign(
            campaign_id="CAMP_WITH_ASSETS",
            name="Campaign with Assets",
            campaign_type="retention",
            channel="google_ads",
            target_segment="Existing Customers",
            budget=1000.0,
            daily_budget=33.33,
            start_date=datetime.now(),
            end_date=datetime.now() + timedelta(days=30),
            status="active",
            objectives=["retain_customers"],
            targeting_parameters={},
            creative_assets=[asset],
            performance_metrics={},
            optimization_history=[]
        )
        
        assert len(campaign.creative_assets) == 1
        assert campaign.creative_assets[0].asset_id == "ASSET_001"
        assert campaign.creative_assets[0].performance_score == 0.8


if __name__ == "__main__":
    pytest.main([__file__])