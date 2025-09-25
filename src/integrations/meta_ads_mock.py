"""
Mock Meta Ads (Facebook/Instagram) API integration for testing and demonstration
"""
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

from config.logging_config import get_agent_logger


@dataclass
class MetaAdsAccount:
    """Meta Ads account structure"""
    account_id: str
    account_name: str
    currency: str
    business_id: str
    account_status: str


@dataclass
class MetaCampaign:
    """Meta Ads campaign structure"""
    campaign_id: str
    name: str
    objective: str
    status: str
    budget_type: str
    daily_budget: Optional[float]
    lifetime_budget: Optional[float]
    bid_strategy: str


class MetaAdsMock:
    """
    Mock Meta Ads API client for testing campaign management functionality
    """
    
    def __init__(self):
        self.logger = get_agent_logger("MetaAdsMock")
        
        # Mock account data
        self.account = MetaAdsAccount(
            account_id="act_1234567890",
            account_name="Demo E-commerce Business",
            currency="USD",
            business_id="1234567890",
            account_status="ACTIVE"
        )
        
        # Mock campaigns storage
        self.campaigns: Dict[str, MetaCampaign] = {}
        self.campaign_metrics: Dict[str, Dict[str, float]] = {}
        self.ad_sets: Dict[str, List[Dict]] = {}
        self.ads: Dict[str, List[Dict]] = {}
        
        # Available objectives
        self.available_objectives = [
            'CONVERSIONS', 'TRAFFIC', 'REACH', 'BRAND_AWARENESS',
            'LEAD_GENERATION', 'APP_INSTALLS', 'VIDEO_VIEWS'
        ]
        
        # Available placements
        self.available_placements = [
            'facebook_feed', 'instagram_feed', 'facebook_stories',
            'instagram_stories', 'facebook_right_column', 'audience_network'
        ]
        
        self.logger.log_action("meta_ads_mock_initialized", {"account_id": self.account.account_id})
    
    def create_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Meta Ads campaign"""
        campaign_id = f"META_{uuid.uuid4().hex[:12].upper()}"
        
        try:
            # Extract campaign parameters
            name = campaign_data.get('name', f'Campaign_{campaign_id}')
            budget = campaign_data.get('budget', 1000.0)
            targeting = campaign_data.get('targeting', {})
            creative_assets = campaign_data.get('creative_assets', [])
            optimization_goal = campaign_data.get('optimization_goal', 'conversions')
            
            # Map optimization goal to Meta objective
            objective_mapping = {
                'conversions': 'CONVERSIONS',
                'traffic': 'TRAFFIC',
                'awareness': 'BRAND_AWARENESS',
                'engagement': 'REACH',
                'leads': 'LEAD_GENERATION'
            }
            objective = objective_mapping.get(optimization_goal, 'CONVERSIONS')
            
            # Create campaign object
            campaign = MetaCampaign(
                campaign_id=campaign_id,
                name=name,
                objective=objective,
                status="ACTIVE",
                budget_type="DAILY",
                daily_budget=budget / 30,  # Convert to daily budget
                lifetime_budget=None,
                bid_strategy="LOWEST_COST_WITHOUT_CAP"
            )
            
            # Store campaign
            self.campaigns[campaign_id] = campaign
            
            # Create ad sets and ads
            self._create_ad_sets(campaign_id, targeting, budget / 30)
            self._create_ads(campaign_id, creative_assets, targeting)
            
            # Initialize performance metrics
            self._initialize_campaign_metrics(campaign_id, budget)
            
            self.logger.log_action("meta_ads_campaign_created", {
                "campaign_id": campaign_id,
                "name": name,
                "objective": objective,
                "daily_budget": campaign.daily_budget
            })
            
            return {
                "success": True,
                "platform_campaign_id": campaign_id,
                "campaign_name": name,
                "objective": objective,
                "status": "ACTIVE",
                "message": "Meta campaign created successfully"
            }
            
        except Exception as e:
            self.logger.log_error("meta_ads_campaign_creation_failed", str(e))
            return {
                "success": False,
                "error": str(e),
                "platform_campaign_id": None
            }
    
    def _create_ad_sets(self, campaign_id: str, targeting: Dict, daily_budget: float):
        """Create ad sets for the campaign"""
        ad_sets = []
        
        # Get targeting parameters
        demographics = targeting.get('demographics', {})
        interests = targeting.get('interests', ['shopping', 'ecommerce'])
        behaviors = targeting.get('behaviors', ['online_shoppers'])
        geographic = targeting.get('geographic', 'United States')
        
        # Create 1-3 ad sets based on targeting diversity
        num_ad_sets = min(3, max(1, len(interests)))
        budget_per_ad_set = daily_budget / num_ad_sets
        
        for i in range(num_ad_sets):
            ad_set_id = f"AS_{uuid.uuid4().hex[:8].upper()}"
            
            # Create audience for this ad set
            audience = self._create_audience_definition(
                demographics, 
                interests[i:i+1] if i < len(interests) else interests[:1],
                behaviors,
                geographic
            )
            
            ad_set = {
                'ad_set_id': ad_set_id,
                'name': f'AdSet_{i+1}_{campaign_id[:8]}',
                'status': 'ACTIVE',
                'daily_budget': budget_per_ad_set,
                'bid_amount': random.uniform(1.0, 5.0),
                'optimization_goal': 'CONVERSIONS',
                'billing_event': 'IMPRESSIONS',
                'targeting': audience,
                'placements': self._select_placements(),
                'start_time': datetime.now().isoformat(),
                'attribution_spec': [{'event_type': 'PAGE_VIEW', 'window_days': 7}]
            }
            
            ad_sets.append(ad_set)
        
        self.ad_sets[campaign_id] = ad_sets
    
    def _create_audience_definition(self, demographics: Dict, interests: List[str], 
                                  behaviors: List[str], geographic: str) -> Dict[str, Any]:
        """Create Meta Ads audience targeting definition"""
        audience = {
            'geo_locations': {
                'countries': [geographic] if geographic != 'United States' else ['US'],
            },
            'age_min': demographics.get('age_min', 18),
            'age_max': demographics.get('age_max', 65),
            'genders': demographics.get('genders', [1, 2]),  # 1=male, 2=female
            'interests': [{'id': f'6003{i:06d}', 'name': interest} for i, interest in enumerate(interests)],
            'behaviors': [{'id': f'6004{i:06d}', 'name': behavior} for i, behavior in enumerate(behaviors)],
            'custom_audiences': [],
            'excluded_custom_audiences': [],
            'lookalike_audiences': []
        }
        
        # Add lookalike audiences if specified
        if demographics.get('use_lookalikes', False):
            audience['lookalike_audiences'] = [
                {'id': f'LAL_{uuid.uuid4().hex[:8]}', 'ratio': 0.01, 'country': 'US'}
            ]
        
        return audience
    
    def _select_placements(self) -> List[str]:
        """Select ad placements for the campaign"""
        # Randomly select 2-4 placements
        num_placements = random.randint(2, 4)
        return random.sample(self.available_placements, num_placements)
    
    def _create_ads(self, campaign_id: str, creative_assets: List[Dict], targeting: Dict):
        """Create ads from creative assets"""
        if campaign_id not in self.ad_sets:
            return
        
        campaign_ads = []
        
        for ad_set in self.ad_sets[campaign_id]:
            ad_set_id = ad_set['ad_set_id']
            
            # Create 1-2 ads per ad set
            for i, asset in enumerate(creative_assets[:2]):
                if asset.get('asset_type') in ['news_feed_ad', 'story_ad', 'video_ad']:
                    ad_id = f"AD_{uuid.uuid4().hex[:8].upper()}"
                    
                    try:
                        # Parse creative content
                        import json
                        content = json.loads(asset.get('content', '{}'))
                        
                        ad = {
                            'ad_id': ad_id,
                            'ad_set_id': ad_set_id,
                            'name': f'Ad_{i+1}_{ad_set_id[:6]}',
                            'status': 'ACTIVE',
                            'creative': self._format_creative(content, asset.get('asset_type')),
                            'tracking_specs': [
                                {'action.type': 'page_view'},
                                {'action.type': 'purchase'},
                                {'action.type': 'add_to_cart'}
                            ]
                        }
                        
                        campaign_ads.append(ad)
                        
                    except:
                        # Fallback ad
                        campaign_ads.append(self._create_fallback_ad(ad_id, ad_set_id, i))
        
        self.ads[campaign_id] = campaign_ads
    
    def _format_creative(self, content: Dict, asset_type: str) -> Dict[str, Any]:
        """Format creative content for Meta Ads"""
        creative = {
            'object_story_spec': {
                'page_id': '123456789',
                'link_data': {
                    'link': 'https://demo-ecommerce.com',
                    'message': content.get('primary_text', 'Shop our amazing products!'),
                    'name': content.get('headline', 'Quality Products'),
                    'description': content.get('description', 'Great deals on quality items'),
                    'call_to_action': {
                        'type': 'SHOP_NOW',
                        'value': {'link': 'https://demo-ecommerce.com'}
                    }
                }
            }
        }
        
        if asset_type == 'video_ad':
            creative['object_story_spec']['video_data'] = {
                'video_id': f'VID_{uuid.uuid4().hex[:8]}',
                'message': content.get('primary_text', 'Watch our video!'),
                'call_to_action': {'type': 'LEARN_MORE'}
            }
        elif asset_type == 'story_ad':
            creative['object_story_spec']['link_data']['child_attachments'] = [
                {
                    'link': 'https://demo-ecommerce.com',
                    'name': content.get('headline', 'Shop Now'),
                    'description': content.get('description', 'Limited time offer')
                }
            ]
        
        return creative
    
    def _create_fallback_ad(self, ad_id: str, ad_set_id: str, index: int) -> Dict[str, Any]:
        """Create fallback ad when creative parsing fails"""
        return {
            'ad_id': ad_id,
            'ad_set_id': ad_set_id,
            'name': f'Fallback_Ad_{index}_{ad_set_id[:6]}',
            'status': 'ACTIVE',
            'creative': {
                'object_story_spec': {
                    'page_id': '123456789',
                    'link_data': {
                        'link': 'https://demo-ecommerce.com',
                        'message': 'Discover amazing products at great prices!',
                        'name': 'Quality Products',
                        'description': 'Shop now for exclusive deals',
                        'call_to_action': {'type': 'SHOP_NOW'}
                    }
                }
            }
        }
    
    def _initialize_campaign_metrics(self, campaign_id: str, budget: float):
        """Initialize realistic performance metrics for a campaign"""
        # Simulate initial performance
        impressions = int(budget * random.uniform(30, 120))
        clicks = int(impressions * random.uniform(0.01, 0.04))
        conversions = int(clicks * random.uniform(0.02, 0.07))
        spend = budget * random.uniform(0.85, 1.0)
        
        # Meta-specific metrics
        post_engagement = int(impressions * random.uniform(0.02, 0.08))
        video_views = int(impressions * random.uniform(0.15, 0.35))
        
        metrics = {
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'spend': spend,
            'revenue': conversions * random.uniform(40, 120),
            'post_engagement': post_engagement,
            'video_views': video_views,
            'page_views': int(clicks * random.uniform(0.8, 1.2)),
            'add_to_cart': int(conversions * random.uniform(2.0, 4.0)),
            'ctr': clicks / impressions if impressions > 0 else 0,
            'cpc': spend / clicks if clicks > 0 else 0,
            'conversion_rate': conversions / clicks if clicks > 0 else 0,
            'cpm': (spend / impressions) * 1000 if impressions > 0 else 0,
            'roas': 0,  # Will be calculated
            'frequency': random.uniform(1.1, 2.5),
            'reach': int(impressions / random.uniform(1.1, 2.5))
        }
        
        # Calculate ROAS
        metrics['roas'] = metrics['revenue'] / metrics['spend'] if metrics['spend'] > 0 else 0
        
        self.campaign_metrics[campaign_id] = metrics
    
    def get_campaign_metrics(self, campaign_id: str) -> Dict[str, float]:
        """Get performance metrics for a campaign"""
        if campaign_id not in self.campaign_metrics:
            return {"error": "Campaign not found"}
        
        # Simulate metric evolution over time
        metrics = self.campaign_metrics[campaign_id].copy()
        
        # Add realistic variance
        variance_factor = random.uniform(0.92, 1.08)
        volume_metrics = ['impressions', 'clicks', 'conversions', 'spend', 'revenue', 
                         'post_engagement', 'video_views', 'page_views', 'add_to_cart']
        
        for key in volume_metrics:
            if key in metrics:
                metrics[key] = metrics[key] * variance_factor
        
        # Recalculate derived metrics
        if metrics['impressions'] > 0:
            metrics['ctr'] = metrics['clicks'] / metrics['impressions']
            metrics['cpm'] = (metrics['spend'] / metrics['impressions']) * 1000
            metrics['reach'] = int(metrics['impressions'] / metrics.get('frequency', 1.5))
        
        if metrics['clicks'] > 0:
            metrics['cpc'] = metrics['spend'] / metrics['clicks']
            metrics['conversion_rate'] = metrics['conversions'] / metrics['clicks']
        
        if metrics['spend'] > 0:
            metrics['roas'] = metrics['revenue'] / metrics['spend']
        
        # Update stored metrics
        self.campaign_metrics[campaign_id] = metrics
        
        return metrics
    
    def update_campaign_budget(self, campaign_id: str, new_daily_budget: float) -> Dict[str, Any]:
        """Update campaign daily budget"""
        if campaign_id not in self.campaigns:
            return {"success": False, "error": "Campaign not found"}
        
        old_budget = self.campaigns[campaign_id].daily_budget
        self.campaigns[campaign_id].daily_budget = new_daily_budget
        
        # Update ad set budgets proportionally
        if campaign_id in self.ad_sets:
            budget_per_ad_set = new_daily_budget / len(self.ad_sets[campaign_id])
            for ad_set in self.ad_sets[campaign_id]:
                ad_set['daily_budget'] = budget_per_ad_set
        
        # Update projected metrics
        if campaign_id in self.campaign_metrics:
            budget_ratio = new_daily_budget / old_budget if old_budget and old_budget > 0 else 1.0
            metrics = self.campaign_metrics[campaign_id]
            
            # Scale volume metrics
            volume_metrics = ['impressions', 'clicks', 'conversions', 'spend']
            for key in volume_metrics:
                if key in metrics:
                    metrics[key] = metrics[key] * budget_ratio
        
        self.logger.log_action("meta_ads_budget_updated", {
            "campaign_id": campaign_id,
            "old_budget": old_budget,
            "new_budget": new_daily_budget
        })
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "old_daily_budget": old_budget,
            "new_daily_budget": new_daily_budget,
            "message": "Daily budget updated successfully"
        }
    
    def pause_campaign(self, campaign_id: str) -> Dict[str, Any]:
        """Pause a campaign"""
        if campaign_id not in self.campaigns:
            return {"success": False, "error": "Campaign not found"}
        
        self.campaigns[campaign_id].status = "PAUSED"
        
        # Pause all ad sets
        if campaign_id in self.ad_sets:
            for ad_set in self.ad_sets[campaign_id]:
                ad_set['status'] = 'PAUSED'
        
        # Pause all ads
        if campaign_id in self.ads:
            for ad in self.ads[campaign_id]:
                ad['status'] = 'PAUSED'
        
        self.logger.log_action("meta_ads_campaign_paused", {"campaign_id": campaign_id})
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "status": "PAUSED",
            "message": "Campaign paused successfully"
        }
    
    def enable_campaign(self, campaign_id: str) -> Dict[str, Any]:
        """Enable a paused campaign"""
        if campaign_id not in self.campaigns:
            return {"success": False, "error": "Campaign not found"}
        
        self.campaigns[campaign_id].status = "ACTIVE"
        
        # Enable all ad sets
        if campaign_id in self.ad_sets:
            for ad_set in self.ad_sets[campaign_id]:
                ad_set['status'] = 'ACTIVE'
        
        # Enable all ads
        if campaign_id in self.ads:
            for ad in self.ads[campaign_id]:
                ad['status'] = 'ACTIVE'
        
        self.logger.log_action("meta_ads_campaign_enabled", {"campaign_id": campaign_id})
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "status": "ACTIVE",
            "message": "Campaign enabled successfully"
        }
    
    def create_custom_audience(self, audience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a custom audience"""
        audience_id = f"CA_{uuid.uuid4().hex[:8].upper()}"
        
        audience = {
            'audience_id': audience_id,
            'name': audience_data.get('name', f'Custom_Audience_{audience_id}'),
            'subtype': audience_data.get('subtype', 'CUSTOM'),
            'customer_file_source': audience_data.get('source', 'USER_PROVIDED_ONLY'),
            'retention_days': audience_data.get('retention_days', 180),
            'approximate_count': random.randint(1000, 50000)
        }
        
        self.logger.log_action("meta_ads_custom_audience_created", {
            "audience_id": audience_id,
            "name": audience['name']
        })
        
        return {
            "success": True,
            "audience_id": audience_id,
            "name": audience['name'],
            "approximate_count": audience['approximate_count'],
            "message": "Custom audience created successfully"
        }
    
    def create_lookalike_audience(self, source_audience_id: str, 
                                 country: str = 'US', ratio: float = 0.01) -> Dict[str, Any]:
        """Create a lookalike audience"""
        lookalike_id = f"LAL_{uuid.uuid4().hex[:8].upper()}"
        
        lookalike = {
            'audience_id': lookalike_id,
            'name': f'Lookalike_{source_audience_id}_{int(ratio*100)}%',
            'subtype': 'LOOKALIKE',
            'origin_audience_id': source_audience_id,
            'country': country,
            'ratio': ratio,
            'approximate_count': random.randint(500000, 5000000)
        }
        
        self.logger.log_action("meta_ads_lookalike_created", {
            "lookalike_id": lookalike_id,
            "source_id": source_audience_id,
            "ratio": ratio
        })
        
        return {
            "success": True,
            "audience_id": lookalike_id,
            "name": lookalike['name'],
            "approximate_count": lookalike['approximate_count'],
            "message": "Lookalike audience created successfully"
        }
    
    def get_audience_insights(self, audience_id: str) -> Dict[str, Any]:
        """Get insights for an audience"""
        # Simulate audience insights
        insights = {
            'audience_size': random.randint(10000, 1000000),
            'age_distribution': {
                '18-24': random.uniform(0.15, 0.25),
                '25-34': random.uniform(0.25, 0.35),
                '35-44': random.uniform(0.20, 0.30),
                '45-54': random.uniform(0.15, 0.25),
                '55+': random.uniform(0.05, 0.15)
            },
            'gender_distribution': {
                'male': random.uniform(0.45, 0.55),
                'female': random.uniform(0.45, 0.55)
            },
            'top_interests': [
                'Shopping', 'Fashion', 'Technology', 'Travel', 'Food'
            ],
            'estimated_reach': random.randint(5000, 500000),
            'frequency_distribution': {
                '1': 0.6,
                '2': 0.25,
                '3+': 0.15
            }
        }
        
        return {
            "audience_id": audience_id,
            "insights": insights
        }
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        return {
            "account_id": self.account.account_id,
            "account_name": self.account.account_name,
            "currency": self.account.currency,
            "account_status": self.account.account_status,
            "business_id": self.account.business_id,
            "total_campaigns": len(self.campaigns),
            "active_campaigns": len([c for c in self.campaigns.values() if c.status == "ACTIVE"]),
            "paused_campaigns": len([c for c in self.campaigns.values() if c.status == "PAUSED"])
        }
    
    def get_all_campaigns(self) -> List[Dict[str, Any]]:
        """Get list of all campaigns"""
        campaigns_list = []
        
        for campaign in self.campaigns.values():
            metrics = self.campaign_metrics.get(campaign.campaign_id, {})
            
            campaigns_list.append({
                "campaign_id": campaign.campaign_id,
                "name": campaign.name,
                "objective": campaign.objective,
                "status": campaign.status,
                "daily_budget": campaign.daily_budget,
                "metrics": metrics
            })
        
        return campaigns_list
    
    def simulate_performance_change(self, campaign_id: str, change_factor: float):
        """Simulate performance change for testing optimization algorithms"""
        if campaign_id in self.campaign_metrics:
            metrics = self.campaign_metrics[campaign_id]
            
            # Apply change factor to key metrics
            for key in ['clicks', 'conversions', 'revenue', 'post_engagement']:
                if key in metrics:
                    metrics[key] = metrics[key] * change_factor
            
            # Recalculate derived metrics
            if metrics.get('impressions', 0) > 0:
                metrics['ctr'] = metrics['clicks'] / metrics['impressions']
            if metrics.get('clicks', 0) > 0:
                metrics['conversion_rate'] = metrics['conversions'] / metrics['clicks']
            if metrics.get('spend', 0) > 0:
                metrics['roas'] = metrics['revenue'] / metrics['spend']