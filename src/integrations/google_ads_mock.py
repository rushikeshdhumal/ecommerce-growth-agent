"""
Mock Google Ads API integration for testing and demonstration
"""
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

from config.logging_config import get_agent_logger


@dataclass
class GoogleAdsAccount:
    """Google Ads account structure"""
    account_id: str
    account_name: str
    currency: str
    time_zone: str
    customer_id: str


@dataclass
class GoogleAdsCampaign:
    """Google Ads campaign structure"""
    campaign_id: str
    name: str
    status: str
    campaign_type: str
    budget: float
    bidding_strategy: str
    target_cpa: Optional[float]
    target_roas: Optional[float]


class GoogleAdsMock:
    """
    Mock Google Ads API client for testing campaign management functionality
    """
    
    def __init__(self):
        self.logger = get_agent_logger("GoogleAdsMock")
        
        # Mock account data
        self.account = GoogleAdsAccount(
            account_id="123-456-7890",
            account_name="Demo E-commerce Account",
            currency="USD",
            time_zone="America/New_York",
            customer_id="1234567890"
        )
        
        # Mock campaigns storage
        self.campaigns: Dict[str, GoogleAdsCampaign] = {}
        self.campaign_metrics: Dict[str, Dict[str, float]] = {}
        self.ad_groups: Dict[str, List[Dict]] = {}
        self.keywords: Dict[str, List[Dict]] = {}
        
        # Performance simulation parameters
        self.base_performance = {
            'ctr': 0.025,
            'conversion_rate': 0.035,
            'cpc': 2.50,
            'quality_score': 7.5
        }
        
        self.logger.log_action("google_ads_mock_initialized", {"account_id": self.account.account_id})
    
    def create_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new Google Ads campaign"""
        campaign_id = f"GADS_{uuid.uuid4().hex[:12].upper()}"
        
        try:
            # Extract campaign parameters
            name = campaign_data.get('name', f'Campaign_{campaign_id}')
            budget = campaign_data.get('budget', 1000.0)
            targeting = campaign_data.get('targeting', {})
            creative_assets = campaign_data.get('creative_assets', [])
            
            # Create campaign object
            campaign = GoogleAdsCampaign(
                campaign_id=campaign_id,
                name=name,
                status="ENABLED",
                campaign_type="SEARCH",
                budget=budget,
                bidding_strategy="TARGET_CPA",
                target_cpa=targeting.get('target_cpa', 25.0),
                target_roas=targeting.get('target_roas', 4.0)
            )
            
            # Store campaign
            self.campaigns[campaign_id] = campaign
            
            # Create ad groups and keywords
            self._create_ad_groups(campaign_id, targeting, creative_assets)
            
            # Initialize performance metrics
            self._initialize_campaign_metrics(campaign_id, budget)
            
            self.logger.log_action("google_ads_campaign_created", {
                "campaign_id": campaign_id,
                "name": name,
                "budget": budget
            })
            
            return {
                "success": True,
                "platform_campaign_id": campaign_id,
                "campaign_name": name,
                "status": "ENABLED",
                "message": "Campaign created successfully"
            }
            
        except Exception as e:
            self.logger.log_error("google_ads_campaign_creation_failed", str(e))
            return {
                "success": False,
                "error": str(e),
                "platform_campaign_id": None
            }
    
    def _create_ad_groups(self, campaign_id: str, targeting: Dict, creative_assets: List[Dict]):
        """Create ad groups and ads for the campaign"""
        ad_groups = []
        
        # Create ad groups based on targeting
        keywords = targeting.get('keywords', ['ecommerce', 'online shopping'])
        
        for i, keyword_theme in enumerate(keywords[:3]):  # Max 3 ad groups
            ad_group_id = f"AG_{uuid.uuid4().hex[:8].upper()}"
            
            ad_group = {
                'ad_group_id': ad_group_id,
                'name': f'AdGroup_{keyword_theme}',
                'status': 'ENABLED',
                'cpc_bid': random.uniform(1.0, 5.0),
                'keywords': self._generate_keywords(keyword_theme),
                'ads': self._create_ads_from_assets(creative_assets, keyword_theme)
            }
            
            ad_groups.append(ad_group)
        
        self.ad_groups[campaign_id] = ad_groups
    
    def _generate_keywords(self, theme: str) -> List[Dict]:
        """Generate keywords for an ad group theme"""
        base_keywords = {
            'ecommerce': ['buy online', 'online store', 'ecommerce platform', 'online shopping'],
            'shopping': ['online shopping', 'shop online', 'buy products', 'shopping deals'],
            'discount': ['discount shopping', 'sale items', 'cheap products', 'best deals'],
            'premium': ['premium products', 'luxury items', 'high quality', 'exclusive deals']
        }
        
        theme_keywords = base_keywords.get(theme, ['online', 'shop', 'buy', 'deals'])
        
        keywords = []
        for keyword in theme_keywords:
            keywords.append({
                'keyword_id': f"KW_{uuid.uuid4().hex[:8].upper()}",
                'text': keyword,
                'match_type': random.choice(['EXACT', 'PHRASE', 'BROAD']),
                'status': 'ENABLED',
                'cpc_bid': random.uniform(0.5, 3.0),
                'quality_score': random.uniform(5.0, 10.0)
            })
        
        return keywords
    
    def _create_ads_from_assets(self, creative_assets: List[Dict], theme: str) -> List[Dict]:
        """Create ads from creative assets"""
        ads = []
        
        for asset in creative_assets[:2]:  # Max 2 ads per ad group
            if asset.get('asset_type') in ['search_ad', 'display_ad']:
                try:
                    # Parse creative content
                    import json
                    content = json.loads(asset.get('content', '{}'))
                    
                    ad = {
                        'ad_id': f"AD_{uuid.uuid4().hex[:8].upper()}",
                        'type': 'RESPONSIVE_SEARCH_AD',
                        'status': 'ENABLED',
                        'headlines': content.get('headlines', [f'Quality Products - {theme}']),
                        'descriptions': content.get('descriptions', ['Shop now for great deals!']),
                        'final_url': 'https://demo-ecommerce.com',
                        'path1': 'shop',
                        'path2': theme.lower().replace(' ', '')
                    }
                    ads.append(ad)
                except:
                    # Fallback ad
                    ads.append({
                        'ad_id': f"AD_{uuid.uuid4().hex[:8].upper()}",
                        'type': 'RESPONSIVE_SEARCH_AD',
                        'status': 'ENABLED',
                        'headlines': [f'Quality Products - {theme}', 'Shop Now'],
                        'descriptions': ['Great deals on quality products', 'Free shipping available'],
                        'final_url': 'https://demo-ecommerce.com'
                    })
        
        return ads
    
    def _initialize_campaign_metrics(self, campaign_id: str, budget: float):
        """Initialize realistic performance metrics for a campaign"""
        # Simulate initial performance based on budget and random factors
        impressions = int(budget * random.uniform(20, 80))
        clicks = int(impressions * random.uniform(0.015, 0.045))
        conversions = int(clicks * random.uniform(0.02, 0.06))
        cost = budget * random.uniform(0.8, 1.0)
        
        metrics = {
            'impressions': impressions,
            'clicks': clicks,
            'conversions': conversions,
            'cost': cost,
            'revenue': conversions * random.uniform(50, 150),
            'ctr': clicks / impressions if impressions > 0 else 0,
            'cpc': cost / clicks if clicks > 0 else 0,
            'conversion_rate': conversions / clicks if clicks > 0 else 0,
            'roas': 0,  # Will be calculated
            'quality_score': random.uniform(6.0, 9.0),
            'search_impression_share': random.uniform(0.4, 0.8)
        }
        
        # Calculate ROAS
        metrics['roas'] = metrics['revenue'] / metrics['cost'] if metrics['cost'] > 0 else 0
        
        self.campaign_metrics[campaign_id] = metrics
    
    def get_campaign_metrics(self, campaign_id: str) -> Dict[str, float]:
        """Get performance metrics for a campaign"""
        if campaign_id not in self.campaign_metrics:
            return {"error": "Campaign not found"}
        
        # Simulate metric evolution over time
        metrics = self.campaign_metrics[campaign_id].copy()
        
        # Add some realistic variance
        variance_factor = random.uniform(0.95, 1.05)
        for key in ['impressions', 'clicks', 'conversions', 'cost', 'revenue']:
            if key in metrics:
                metrics[key] = metrics[key] * variance_factor
        
        # Recalculate derived metrics
        if metrics['impressions'] > 0:
            metrics['ctr'] = metrics['clicks'] / metrics['impressions']
        if metrics['clicks'] > 0:
            metrics['cpc'] = metrics['cost'] / metrics['clicks']
            metrics['conversion_rate'] = metrics['conversions'] / metrics['clicks']
        if metrics['cost'] > 0:
            metrics['roas'] = metrics['revenue'] / metrics['cost']
        
        # Update stored metrics
        self.campaign_metrics[campaign_id] = metrics
        
        return metrics
    
    def update_campaign_budget(self, campaign_id: str, new_budget: float) -> Dict[str, Any]:
        """Update campaign budget"""
        if campaign_id not in self.campaigns:
            return {"success": False, "error": "Campaign not found"}
        
        old_budget = self.campaigns[campaign_id].budget
        self.campaigns[campaign_id].budget = new_budget
        
        # Update projected metrics based on budget change
        if campaign_id in self.campaign_metrics:
            budget_ratio = new_budget / old_budget if old_budget > 0 else 1.0
            metrics = self.campaign_metrics[campaign_id]
            
            # Scale volume metrics
            for key in ['impressions', 'clicks', 'conversions', 'cost']:
                if key in metrics:
                    metrics[key] = metrics[key] * budget_ratio
        
        self.logger.log_action("google_ads_budget_updated", {
            "campaign_id": campaign_id,
            "old_budget": old_budget,
            "new_budget": new_budget
        })
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "old_budget": old_budget,
            "new_budget": new_budget,
            "message": "Budget updated successfully"
        }
    
    def pause_campaign(self, campaign_id: str) -> Dict[str, Any]:
        """Pause a campaign"""
        if campaign_id not in self.campaigns:
            return {"success": False, "error": "Campaign not found"}
        
        self.campaigns[campaign_id].status = "PAUSED"
        
        self.logger.log_action("google_ads_campaign_paused", {"campaign_id": campaign_id})
        
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
        
        self.campaigns[campaign_id].status = "ENABLED"
        
        self.logger.log_action("google_ads_campaign_enabled", {"campaign_id": campaign_id})
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "status": "ENABLED",
            "message": "Campaign enabled successfully"
        }
    
    def update_keywords(self, campaign_id: str, keyword_updates: List[Dict]) -> Dict[str, Any]:
        """Update keywords for a campaign"""
        if campaign_id not in self.ad_groups:
            return {"success": False, "error": "Campaign not found"}
        
        updated_count = 0
        
        for ad_group in self.ad_groups[campaign_id]:
            for keyword in ad_group['keywords']:
                for update in keyword_updates:
                    if update.get('keyword_text') == keyword['text']:
                        # Update keyword properties
                        if 'cpc_bid' in update:
                            keyword['cpc_bid'] = update['cpc_bid']
                        if 'status' in update:
                            keyword['status'] = update['status']
                        updated_count += 1
        
        self.logger.log_action("google_ads_keywords_updated", {
            "campaign_id": campaign_id,
            "updated_count": updated_count
        })
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "updated_keywords": updated_count,
            "message": f"Updated {updated_count} keywords"
        }
    
    def get_keyword_performance(self, campaign_id: str) -> Dict[str, Any]:
        """Get keyword performance data"""
        if campaign_id not in self.ad_groups:
            return {"error": "Campaign not found"}
        
        keyword_performance = []
        
        for ad_group in self.ad_groups[campaign_id]:
            for keyword in ad_group['keywords']:
                # Simulate keyword performance
                impressions = random.randint(100, 2000)
                clicks = int(impressions * random.uniform(0.01, 0.05))
                conversions = int(clicks * random.uniform(0.02, 0.08))
                cost = clicks * keyword['cpc_bid']
                
                perf = {
                    'keyword_id': keyword['keyword_id'],
                    'keyword_text': keyword['text'],
                    'match_type': keyword['match_type'],
                    'impressions': impressions,
                    'clicks': clicks,
                    'conversions': conversions,
                    'cost': round(cost, 2),
                    'ctr': round(clicks / impressions, 4) if impressions > 0 else 0,
                    'conversion_rate': round(conversions / clicks, 4) if clicks > 0 else 0,
                    'quality_score': keyword['quality_score']
                }
                
                keyword_performance.append(perf)
        
        return {
            "campaign_id": campaign_id,
            "keyword_performance": keyword_performance
        }
    
    def create_responsive_search_ad(self, ad_group_id: str, ad_data: Dict) -> Dict[str, Any]:
        """Create a new responsive search ad"""
        ad_id = f"AD_{uuid.uuid4().hex[:8].upper()}"
        
        ad = {
            'ad_id': ad_id,
            'ad_group_id': ad_group_id,
            'type': 'RESPONSIVE_SEARCH_AD',
            'status': 'ENABLED',
            'headlines': ad_data.get('headlines', []),
            'descriptions': ad_data.get('descriptions', []),
            'final_url': ad_data.get('final_url', 'https://demo-ecommerce.com'),
            'path1': ad_data.get('path1', ''),
            'path2': ad_data.get('path2', '')
        }
        
        # Add to appropriate campaign's ad groups
        for campaign_id, ad_groups in self.ad_groups.items():
            for ad_group in ad_groups:
                if ad_group['ad_group_id'] == ad_group_id:
                    ad_group['ads'].append(ad)
                    break
        
        self.logger.log_action("google_ads_ad_created", {
            "ad_id": ad_id,
            "ad_group_id": ad_group_id
        })
        
        return {
            "success": True,
            "ad_id": ad_id,
            "status": "ENABLED",
            "message": "Responsive search ad created successfully"
        }
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        return {
            "account_id": self.account.account_id,
            "account_name": self.account.account_name,
            "currency": self.account.currency,
            "time_zone": self.account.time_zone,
            "total_campaigns": len(self.campaigns),
            "active_campaigns": len([c for c in self.campaigns.values() if c.status == "ENABLED"]),
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
                "status": campaign.status,
                "budget": campaign.budget,
                "campaign_type": campaign.campaign_type,
                "metrics": metrics
            })
        
        return campaigns_list
    
    def simulate_performance_change(self, campaign_id: str, change_factor: float):
        """Simulate performance change for testing optimization algorithms"""
        if campaign_id in self.campaign_metrics:
            metrics = self.campaign_metrics[campaign_id]
            
            # Apply change factor to key metrics
            for key in ['clicks', 'conversions', 'revenue']:
                if key in metrics:
                    metrics[key] = metrics[key] * change_factor
            
            # Recalculate derived metrics
            if metrics.get('impressions', 0) > 0:
                metrics['ctr'] = metrics['clicks'] / metrics['impressions']
            if metrics.get('clicks', 0) > 0:
                metrics['conversion_rate'] = metrics['conversions'] / metrics['clicks']
            if metrics.get('cost', 0) > 0:
                metrics['roas'] = metrics['revenue'] / metrics['cost']