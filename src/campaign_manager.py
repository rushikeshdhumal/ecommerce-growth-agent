"""
Campaign Manager for E-commerce Growth Agent
Handles multi-channel campaign creation, optimization, and creative generation
"""
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

import openai
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings, CAMPAIGN_TEMPLATES
from config.logging_config import get_agent_logger
from src.integrations.google_ads_mock import GoogleAdsMock
from src.integrations.meta_ads_mock import MetaAdsMock
from src.integrations.klaviyo_mock import KlaviyoMock


@dataclass
class Campaign:
    """Campaign data structure"""
    campaign_id: str
    name: str
    campaign_type: str
    channel: str
    target_segment: str
    budget: float
    daily_budget: float
    start_date: datetime
    end_date: datetime
    status: str
    objectives: List[str]
    targeting_parameters: Dict[str, Any]
    creative_assets: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    optimization_history: List[Dict[str, Any]]


@dataclass
class CreativeAsset:
    """Creative asset structure"""
    asset_id: str
    asset_type: str  # text, image, video, email
    content: str
    metadata: Dict[str, Any]
    performance_score: float
    a_b_test_variant: str


class CampaignManager:
    """
    Manages multi-channel marketing campaigns with AI-powered creative generation
    and automated optimization
    """
    
    def __init__(self):
        self.logger = get_agent_logger("CampaignManager")
        
        # Initialize API clients
        self._setup_ai_clients()
        self._setup_platform_clients()
        
        # Campaign storage
        self.active_campaigns: Dict[str, Campaign] = {}
        self.campaign_templates = CAMPAIGN_TEMPLATES
        
        # Performance tracking
        self.performance_history = []
        self.optimization_results = []
        
        self.logger.log_action("campaign_manager_initialized", {})
    
    def _setup_ai_clients(self):
        """Initialize AI clients for creative generation"""
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
            self.openai_client = openai
        
        if settings.ANTHROPIC_API_KEY:
            self.anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    
    def _setup_platform_clients(self):
        """Initialize mock platform clients"""
        self.google_ads = GoogleAdsMock()
        self.meta_ads = MetaAdsMock()
        self.klaviyo = KlaviyoMock()
        
        self.platform_clients = {
            'google_ads': self.google_ads,
            'meta_ads': self.meta_ads,
            'email': self.klaviyo,
            'sms': self.klaviyo,
            'display': self.google_ads  # Using Google for display
        }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_llm(self, prompt: str, system_message: str = None, max_tokens: int = 2048) -> str:
        """Call LLM for creative generation with retry logic"""
        try:
            if settings.AGENT_MODEL.startswith("gpt"):
                messages = []
                if system_message:
                    messages.append({"role": "system", "content": system_message})
                messages.append({"role": "user", "content": prompt})
                
                response = self.openai_client.ChatCompletion.create(
                    model=settings.AGENT_MODEL,
                    messages=messages,
                    temperature=settings.TEMPERATURE,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content
            
            elif settings.AGENT_MODEL.startswith("claude"):
                full_prompt = f"{system_message}\n\nHuman: {prompt}\n\nAssistant:" if system_message else prompt
                response = self.anthropic_client.messages.create(
                    model=settings.AGENT_MODEL,
                    max_tokens=max_tokens,
                    temperature=settings.TEMPERATURE,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                return response.content[0].text
            
        except Exception as e:
            self.logger.log_error("creative_generation_failed", str(e))
            raise
    
    def create_campaign(self, campaign_type: str, target_segment: str, 
                       budget: float, channels: List[str], 
                       custom_objectives: List[str] = None) -> Dict[str, Any]:
        """Create a new multi-channel marketing campaign"""
        
        campaign_id = f"CAMP_{uuid.uuid4().hex[:8].upper()}"
        
        self.logger.log_action("campaign_creation_started", {
            "campaign_id": campaign_id,
            "campaign_type": campaign_type,
            "target_segment": target_segment,
            "budget": budget,
            "channels": channels
        })
        
        try:
            # Get campaign template
            template = self.campaign_templates.get(campaign_type, self.campaign_templates['acquisition'])
            
            # Calculate budget allocation across channels
            budget_allocation = self._calculate_budget_allocation(budget, channels, template)
            
            # Generate campaign strategy and creative brief
            campaign_strategy = self._generate_campaign_strategy(
                campaign_type, target_segment, budget, channels, custom_objectives
            )
            
            # Create campaigns for each channel
            channel_campaigns = {}
            total_success = True
            
            for channel in channels:
                channel_budget = budget_allocation.get(channel, budget / len(channels))
                
                try:
                    # Generate channel-specific creative assets
                    creative_assets = self._generate_creative_assets(
                        channel, campaign_type, target_segment, campaign_strategy
                    )
                    
                    # Create targeting parameters
                    targeting_params = self._create_targeting_parameters(
                        channel, target_segment, campaign_strategy
                    )
                    
                    # Create campaign on platform
                    platform_result = self._create_platform_campaign(
                        channel, campaign_id, channel_budget, targeting_params, creative_assets
                    )
                    
                    if platform_result.get('success'):
                        channel_campaigns[channel] = {
                            "platform_campaign_id": platform_result.get('platform_campaign_id'),
                            "budget": channel_budget,
                            "creative_assets": creative_assets,
                            "targeting_parameters": targeting_params,
                            "status": "active"
                        }
                    else:
                        total_success = False
                        self.logger.log_error("channel_campaign_failed", f"Failed to create {channel} campaign", {
                            "campaign_id": campaign_id,
                            "channel": channel
                        })
                        
                except Exception as e:
                    total_success = False
                    self.logger.log_error("channel_campaign_error", str(e), {
                        "campaign_id": campaign_id,
                        "channel": channel
                    })
            
            if channel_campaigns:  # At least one channel succeeded
                # Create master campaign object
                campaign = Campaign(
                    campaign_id=campaign_id,
                    name=f"{campaign_type.title()} Campaign - {target_segment}",
                    campaign_type=campaign_type,
                    channel="multi_channel",
                    target_segment=target_segment,
                    budget=budget,
                    daily_budget=budget / 30,  # Assume 30-day campaign
                    start_date=datetime.now(),
                    end_date=datetime.now() + timedelta(days=30),
                    status="active",
                    objectives=custom_objectives or template.get('objectives', []),
                    targeting_parameters={"channels": channel_campaigns},
                    creative_assets=self._aggregate_creative_assets(channel_campaigns),
                    performance_metrics={},
                    optimization_history=[]
                )
                
                # Store campaign
                self.active_campaigns[campaign_id] = campaign
                
                self.logger.log_action("campaign_created_successfully", {
                    "campaign_id": campaign_id,
                    "channels_created": len(channel_campaigns),
                    "total_budget": budget
                })
                
                return {
                    "success": True,
                    "campaign_id": campaign_id,
                    "campaign": asdict(campaign),
                    "channels_created": list(channel_campaigns.keys()),
                    "budget_allocation": budget_allocation,
                    "creative_assets_count": len(campaign.creative_assets)
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to create campaigns on any channel",
                    "campaign_id": campaign_id
                }
                
        except Exception as e:
            self.logger.log_error("campaign_creation_failed", str(e), {"campaign_id": campaign_id})
            return {
                "success": False,
                "error": str(e),
                "campaign_id": campaign_id
            }
    
    def _calculate_budget_allocation(self, total_budget: float, channels: List[str], template: Dict) -> Dict[str, float]:
        """Calculate optimal budget allocation across channels"""
        template_allocation = template.get('budget_allocation', {})
        
        # If template has allocation for requested channels, use it
        if all(channel in template_allocation for channel in channels):
            total_template_weight = sum(template_allocation[ch] for ch in channels)
            return {
                channel: total_budget * (template_allocation[channel] / total_template_weight)
                for channel in channels
            }
        
        # Otherwise, use performance-based allocation
        channel_weights = self._get_channel_performance_weights(channels)
        total_weight = sum(channel_weights.values())
        
        return {
            channel: total_budget * (weight / total_weight)
            for channel, weight in channel_weights.items()
        }
    
    def _get_channel_performance_weights(self, channels: List[str]) -> Dict[str, float]:
        """Get performance-based weights for budget allocation"""
        # Default weights based on typical performance
        default_weights = {
            'google_ads': 0.35,
            'meta_ads': 0.30,
            'email': 0.20,
            'sms': 0.10,
            'display': 0.15
        }
        
        # TODO: In a real implementation, this would use historical ROAS data
        weights = {}
        for channel in channels:
            weights[channel] = default_weights.get(channel, 0.20)
        
        return weights
    
    def _generate_campaign_strategy(self, campaign_type: str, target_segment: str, 
                                  budget: float, channels: List[str], 
                                  custom_objectives: List[str] = None) -> Dict[str, Any]:
        """Generate comprehensive campaign strategy using AI"""
        
        strategy_prompt = f"""
        Create a comprehensive marketing campaign strategy with the following parameters:
        
        Campaign Type: {campaign_type}
        Target Segment: {target_segment}
        Budget: ${budget:,.2f}
        Channels: {', '.join(channels)}
        Custom Objectives: {custom_objectives or 'Use campaign type defaults'}
        
        Please provide:
        1. Campaign positioning and key messaging
        2. Target audience insights and pain points
        3. Value propositions for each channel
        4. Success metrics and KPIs
        5. Creative direction and themes
        6. Timing and sequencing strategy
        
        Format as structured JSON with clear sections.
        """
        
        system_message = """You are an expert marketing strategist. Create detailed, actionable campaign strategies that are data-driven and customer-focused. Consider channel-specific best practices and audience behavior patterns."""
        
        try:
            strategy_response = self._call_llm(strategy_prompt, system_message)
            strategy = json.loads(strategy_response)
        except (json.JSONDecodeError, Exception):
            # Fallback strategy
            strategy = self._create_fallback_strategy(campaign_type, target_segment, channels)
        
        return strategy
    
    def _generate_creative_assets(self, channel: str, campaign_type: str, 
                                target_segment: str, strategy: Dict[str, Any]) -> List[CreativeAsset]:
        """Generate channel-specific creative assets using AI"""
        
        assets = []
        
        # Get channel-specific requirements
        channel_specs = self._get_channel_creative_specs(channel)
        
        for spec in channel_specs:
            creative_prompt = self._create_creative_prompt(
                spec, channel, campaign_type, target_segment, strategy
            )
            
            try:
                creative_content = self._call_llm(creative_prompt, 
                    "You are a creative copywriter and marketing expert. Generate compelling, conversion-focused creative content.")
                
                asset = CreativeAsset(
                    asset_id=f"ASSET_{uuid.uuid4().hex[:8].upper()}",
                    asset_type=spec['type'],
                    content=creative_content,
                    metadata={
                        "channel": channel,
                        "campaign_type": campaign_type,
                        "target_segment": target_segment,
                        "spec": spec
                    },
                    performance_score=0.0,  # Will be updated based on performance
                    a_b_test_variant="A"
                )
                
                assets.append(asset)
                
                # Generate B variant for A/B testing
                if spec.get('create_variants', True):
                    variant_prompt = f"Create a different variation of this creative:\n{creative_content}\n\nMake it significantly different while maintaining the same core message and call-to-action."
                    
                    variant_content = self._call_llm(variant_prompt,
                        "Create compelling variations for A/B testing. Focus on different angles, emotions, or benefits.")
                    
                    variant_asset = CreativeAsset(
                        asset_id=f"ASSET_{uuid.uuid4().hex[:8].upper()}",
                        asset_type=spec['type'],
                        content=variant_content,
                        metadata={
                            "channel": channel,
                            "campaign_type": campaign_type,
                            "target_segment": target_segment,
                            "spec": spec
                        },
                        performance_score=0.0,
                        a_b_test_variant="B"
                    )
                    
                    assets.append(variant_asset)
                
            except Exception as e:
                self.logger.log_error("creative_generation_failed", str(e), {
                    "channel": channel,
                    "spec": spec
                })
                
                # Create fallback creative
                fallback_asset = self._create_fallback_creative(spec, channel, campaign_type)
                assets.append(fallback_asset)
        
        return assets
    
    def _get_channel_creative_specs(self, channel: str) -> List[Dict[str, Any]]:
        """Get creative specifications for each channel"""
        specs = {
            'google_ads': [
                {
                    'type': 'search_ad',
                    'format': 'text',
                    'headline_limit': 30,
                    'description_limit': 90,
                    'create_variants': True
                },
                {
                    'type': 'display_ad',
                    'format': 'text_image',
                    'headline_limit': 25,
                    'description_limit': 80,
                    'create_variants': True
                }
            ],
            'meta_ads': [
                {
                    'type': 'news_feed_ad',
                    'format': 'text_image',
                    'headline_limit': 40,
                    'text_limit': 125,
                    'create_variants': True
                },
                {
                    'type': 'story_ad',
                    'format': 'image_video',
                    'text_limit': 50,
                    'create_variants': False
                }
            ],
            'email': [
                {
                    'type': 'email_campaign',
                    'format': 'html_text',
                    'subject_limit': 50,
                    'preview_limit': 90,
                    'create_variants': True
                }
            ],
            'sms': [
                {
                    'type': 'sms_message',
                    'format': 'text',
                    'message_limit': 160,
                    'create_variants': True
                }
            ],
            'display': [
                {
                    'type': 'banner_ad',
                    'format': 'text_image',
                    'headline_limit': 25,
                    'description_limit': 75,
                    'create_variants': True
                }
            ]
        }
        
        return specs.get(channel, [{'type': 'generic', 'format': 'text', 'create_variants': False}])
    
    def _create_creative_prompt(self, spec: Dict, channel: str, campaign_type: str, 
                              target_segment: str, strategy: Dict) -> str:
        """Create prompt for creative generation"""
        
        key_message = strategy.get('positioning', {}).get('key_message', 'Drive engagement and conversions')
        value_props = strategy.get('value_propositions', {}).get(channel, ['Quality products', 'Great service'])
        
        prompt = f"""
        Create compelling {spec['type']} creative for {channel} with these requirements:
        
        CAMPAIGN CONTEXT:
        - Type: {campaign_type}
        - Target: {target_segment}
        - Channel: {channel}
        - Key Message: {key_message}
        - Value Props: {', '.join(value_props) if isinstance(value_props, list) else value_props}
        
        CREATIVE SPECIFICATIONS:
        - Format: {spec['format']}
        - Type: {spec['type']}
        """
        
        # Add specific limits based on creative type
        if 'headline_limit' in spec:
            prompt += f"\n- Headline: Max {spec['headline_limit']} characters"
        if 'description_limit' in spec:
            prompt += f"\n- Description: Max {spec['description_limit']} characters"
        if 'text_limit' in spec:
            prompt += f"\n- Text: Max {spec['text_limit']} characters"
        if 'subject_limit' in spec:
            prompt += f"\n- Subject: Max {spec['subject_limit']} characters"
        if 'message_limit' in spec:
            prompt += f"\n- Message: Max {spec['message_limit']} characters"
        
        prompt += f"""
        
        REQUIREMENTS:
        - Include compelling call-to-action
        - Focus on benefits over features
        - Create urgency where appropriate
        - Match channel best practices
        - Target {target_segment} specifically
        
        Provide the creative content in a structured JSON format with all required components.
        """
        
        return prompt
    
    def _create_targeting_parameters(self, channel: str, target_segment: str, 
                                   strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Create targeting parameters for each channel"""
        
        # Base targeting from strategy
        audience_insights = strategy.get('audience_insights', {})
        
        # Channel-specific targeting
        targeting = {
            'target_segment': target_segment,
            'demographics': audience_insights.get('demographics', {}),
            'interests': audience_insights.get('interests', []),
            'behaviors': audience_insights.get('behaviors', []),
            'geographic': audience_insights.get('geographic', 'United States')
        }
        
        # Add channel-specific parameters
        if channel == 'google_ads':
            targeting.update({
                'keywords': self._generate_keywords(target_segment, strategy),
                'match_types': ['exact', 'phrase', 'broad'],
                'device_targeting': 'all',
                'location_targeting': 'geo_target'
            })
        
        elif channel == 'meta_ads':
            targeting.update({
                'lookalike_audiences': True,
                'custom_audiences': [target_segment],
                'placement': ['news_feed', 'stories', 'right_column'],
                'age_range': audience_insights.get('age_range', '25-55')
            })
        
        elif channel in ['email', 'sms']:
            targeting.update({
                'segment_filters': self._create_email_filters(target_segment),
                'frequency_cap': '1_per_day',
                'send_time_optimization': True
            })
        
        return targeting
    
    def _generate_keywords(self, target_segment: str, strategy: Dict) -> List[str]:
        """Generate relevant keywords for search campaigns"""
        # This would typically use keyword research tools
        base_keywords = [
            'ecommerce',
            'online shopping',
            'buy online',
            'discount',
            'sale'
        ]
        
        # Add segment-specific keywords
        if 'high_value' in target_segment.lower():
            base_keywords.extend(['premium', 'luxury', 'exclusive'])
        elif 'new' in target_segment.lower():
            base_keywords.extend(['first time', 'new customer', 'welcome'])
        
        return base_keywords
    
    def _create_email_filters(self, target_segment: str) -> Dict[str, Any]:
        """Create email segmentation filters"""
        filters = {
            'engagement_level': 'all',
            'purchase_history': 'any',
            'lifecycle_stage': 'all'
        }
        
        if 'high_value' in target_segment.lower():
            filters['purchase_history'] = 'high_value'
        elif 'churned' in target_segment.lower():
            filters['engagement_level'] = 'low'
            filters['last_purchase'] = '90_days_ago'
        
        return filters
    
    def _create_platform_campaign(self, channel: str, campaign_id: str, 
                                 budget: float, targeting: Dict, 
                                 creative_assets: List[CreativeAsset]) -> Dict[str, Any]:
        """Create campaign on the specific platform"""
        
        platform_client = self.platform_clients.get(channel)
        if not platform_client:
            return {"success": False, "error": f"No client available for channel: {channel}"}
        
        try:
            # Prepare platform-specific campaign data
            campaign_data = {
                'name': f"{campaign_id}_{channel}",
                'budget': budget,
                'targeting': targeting,
                'creative_assets': [asdict(asset) for asset in creative_assets],
                'optimization_goal': 'conversions'
            }
            
            # Create campaign on platform
            result = platform_client.create_campaign(campaign_data)
            
            return result
            
        except Exception as e:
            self.logger.log_error("platform_campaign_creation_failed", str(e), {
                "channel": channel,
                "campaign_id": campaign_id
            })
            return {"success": False, "error": str(e)}
    
    def get_campaign_performance(self, campaign_id: str) -> Dict[str, Any]:
        """Get performance metrics for a campaign"""
        if campaign_id not in self.active_campaigns:
            return {"error": "Campaign not found"}
        
        campaign = self.active_campaigns[campaign_id]
        
        # Collect performance from all channels
        channel_performance = {}
        overall_metrics = {
            'spend': 0,
            'revenue': 0,
            'clicks': 0,
            'impressions': 0,
            'conversions': 0
        }
        
        for channel, channel_data in campaign.targeting_parameters.get('channels', {}).items():
            platform_client = self.platform_clients.get(channel)
            if platform_client:
                platform_campaign_id = channel_data.get('platform_campaign_id')
                if platform_campaign_id:
                    metrics = platform_client.get_campaign_metrics(platform_campaign_id)
                    channel_performance[channel] = metrics
                    
                    # Aggregate overall metrics
                    for key in overall_metrics:
                        overall_metrics[key] += metrics.get(key, 0)
        
        # Calculate derived metrics
        overall_metrics['ctr'] = (overall_metrics['clicks'] / overall_metrics['impressions'] 
                                 if overall_metrics['impressions'] > 0 else 0)
        overall_metrics['conversion_rate'] = (overall_metrics['conversions'] / overall_metrics['clicks'] 
                                            if overall_metrics['clicks'] > 0 else 0)
        overall_metrics['roas'] = (overall_metrics['revenue'] / overall_metrics['spend'] 
                                 if overall_metrics['spend'] > 0 else 0)
        overall_metrics['cpc'] = (overall_metrics['spend'] / overall_metrics['clicks'] 
                                if overall_metrics['clicks'] > 0 else 0)
        
        # Update campaign performance
        campaign.performance_metrics = overall_metrics
        
        return {
            'campaign_id': campaign_id,
            'overall_metrics': overall_metrics,
            'channel_performance': channel_performance,
            'performance_grade': self._calculate_performance_grade(overall_metrics),
            'optimization_opportunities': self._identify_optimization_opportunities(overall_metrics, channel_performance)
        }
    
    def optimize_campaign(self, campaign_id: str, optimization_action: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization to a campaign"""
        if campaign_id not in self.active_campaigns:
            return {"success": False, "error": "Campaign not found"}
        
        campaign = self.active_campaigns[campaign_id]
        action_type = optimization_action.get('type')
        
        self.logger.log_action("campaign_optimization_started", {
            "campaign_id": campaign_id,
            "optimization_type": action_type
        })
        
        try:
            if action_type == 'increase_budget':
                result = self._optimize_budget_increase(campaign, optimization_action)
            elif action_type == 'decrease_budget':
                result = self._optimize_budget_decrease(campaign, optimization_action)
            elif action_type == 'refresh_creative':
                result = self._optimize_creative_refresh(campaign, optimization_action)
            elif action_type == 'refine_targeting':
                result = self._optimize_targeting(campaign, optimization_action)
            elif action_type == 'pause_underperforming':
                result = self._pause_underperforming_elements(campaign, optimization_action)
            else:
                return {"success": False, "error": f"Unknown optimization type: {action_type}"}
            
            # Log optimization result
            optimization_record = {
                'timestamp': datetime.now().isoformat(),
                'action_type': action_type,
                'action_details': optimization_action,
                'result': result,
                'performance_before': campaign.performance_metrics.copy()
            }
            
            campaign.optimization_history.append(optimization_record)
            
            self.logger.log_optimization(
                action_type,
                optimization_record['performance_before'],
                result.get('new_performance', {}),
                result.get('improvement_estimate', 0)
            )
            
            return result
            
        except Exception as e:
            self.logger.log_error("campaign_optimization_failed", str(e), {
                "campaign_id": campaign_id,
                "optimization_type": action_type
            })
            return {"success": False, "error": str(e)}
    
    def _optimize_budget_increase(self, campaign: Campaign, action: Dict) -> Dict[str, Any]:
        """Optimize campaign by increasing budget"""
        factor = action.get('factor', 1.2)
        new_budget = min(campaign.budget * factor, settings.MAX_DAILY_BUDGET * 30)
        
        budget_increase = new_budget - campaign.budget
        
        # Update budget across channels proportionally
        for channel, channel_data in campaign.targeting_parameters.get('channels', {}).items():
            platform_client = self.platform_clients.get(channel)
            if platform_client:
                current_budget = channel_data.get('budget', 0)
                new_channel_budget = current_budget * factor
                
                platform_client.update_campaign_budget(
                    channel_data.get('platform_campaign_id'),
                    new_channel_budget
                )
                
                channel_data['budget'] = new_channel_budget
        
        campaign.budget = new_budget
        campaign.daily_budget = new_budget / 30
        
        return {
            "success": True,
            "action": "budget_increased",
            "old_budget": campaign.budget / factor,
            "new_budget": new_budget,
            "increase_amount": budget_increase,
            "improvement_estimate": 15  # Estimated % improvement
        }
    
    def _optimize_budget_decrease(self, campaign: Campaign, action: Dict) -> Dict[str, Any]:
        """Optimize campaign by decreasing budget"""
        factor = action.get('factor', 0.8)
        new_budget = max(campaign.budget * factor, settings.MIN_CAMPAIGN_BUDGET)
        
        budget_decrease = campaign.budget - new_budget
        
        # Update budget across channels proportionally
        for channel, channel_data in campaign.targeting_parameters.get('channels', {}).items():
            platform_client = self.platform_clients.get(channel)
            if platform_client:
                current_budget = channel_data.get('budget', 0)
                new_channel_budget = current_budget * factor
                
                platform_client.update_campaign_budget(
                    channel_data.get('platform_campaign_id'),
                    new_channel_budget
                )
                
                channel_data['budget'] = new_channel_budget
        
        campaign.budget = new_budget
        campaign.daily_budget = new_budget / 30
        
        return {
            "success": True,
            "action": "budget_decreased",
            "old_budget": campaign.budget / factor,
            "new_budget": new_budget,
            "decrease_amount": budget_decrease,
            "improvement_estimate": 5  # Estimated % improvement in efficiency
        }
    
    def _optimize_creative_refresh(self, campaign: Campaign, action: Dict) -> Dict[str, Any]:
        """Optimize campaign by refreshing creative assets"""
        refreshed_assets = []
        
        # Generate new creative assets for underperforming ones
        for asset in campaign.creative_assets:
            if asset.performance_score < 0.6:  # Refresh low-performing assets
                try:
                    # Generate new creative
                    refresh_prompt = f"""
                    Refresh this creative asset that's underperforming:
                    Current: {asset.content}
                    
                    Create a completely new version with:
                    - Different angle or benefit focus
                    - Fresh copy and messaging
                    - Maintain the same call-to-action goal
                    
                    Type: {asset.asset_type}
                    Channel: {asset.metadata.get('channel')}
                    """
                    
                    new_content = self._call_llm(refresh_prompt, 
                        "You are a creative expert focused on improving ad performance through fresh, compelling copy.")
                    
                    new_asset = CreativeAsset(
                        asset_id=f"ASSET_{uuid.uuid4().hex[:8].upper()}",
                        asset_type=asset.asset_type,
                        content=new_content,
                        metadata=asset.metadata.copy(),
                        performance_score=0.0,
                        a_b_test_variant="C"  # New variant
                    )
                    
                    refreshed_assets.append(new_asset)
                    
                except Exception as e:
                    self.logger.log_error("creative_refresh_failed", str(e), {
                        "asset_id": asset.asset_id
                    })
        
        # Add refreshed assets to campaign
        campaign.creative_assets.extend(refreshed_assets)
        
        return {
            "success": True,
            "action": "creative_refreshed",
            "assets_refreshed": len(refreshed_assets),
            "new_assets": [asdict(asset) for asset in refreshed_assets],
            "improvement_estimate": 12  # Estimated % improvement
        }
    
    def _calculate_performance_grade(self, metrics: Dict[str, float]) -> str:
        """Calculate overall performance grade for campaign"""
        roas = metrics.get('roas', 0)
        ctr = metrics.get('ctr', 0)
        conversion_rate = metrics.get('conversion_rate', 0)
        
        score = 0
        
        # ROAS scoring (40% weight)
        if roas >= 4.0:
            score += 40
        elif roas >= 3.0:
            score += 32
        elif roas >= 2.0:
            score += 24
        elif roas >= 1.0:
            score += 16
        
        # CTR scoring (30% weight)
        if ctr >= 0.05:
            score += 30
        elif ctr >= 0.03:
            score += 24
        elif ctr >= 0.02:
            score += 18
        elif ctr >= 0.01:
            score += 12
        
        # Conversion rate scoring (30% weight)
        if conversion_rate >= 0.08:
            score += 30
        elif conversion_rate >= 0.05:
            score += 24
        elif conversion_rate >= 0.03:
            score += 18
        elif conversion_rate >= 0.02:
            score += 12
        
        if score >= 85:
            return "A"
        elif score >= 70:
            return "B"
        elif score >= 55:
            return "C"
        elif score >= 40:
            return "D"
        else:
            return "F"
    
    def _identify_optimization_opportunities(self, overall_metrics: Dict, 
                                           channel_performance: Dict) -> List[Dict[str, Any]]:
        """Identify specific optimization opportunities"""
        opportunities = []
        
        # Low ROAS opportunity
        if overall_metrics.get('roas', 0) < settings.MIN_ROAS_THRESHOLD:
            opportunities.append({
                "type": "improve_roas",
                "priority": "high",
                "description": "ROAS below threshold - optimize targeting and creative",
                "actions": ["refine_targeting", "refresh_creative"]
            })
        
        # Low CTR opportunity
        if overall_metrics.get('ctr', 0) < 0.02:
            opportunities.append({
                "type": "improve_ctr",
                "priority": "medium",
                "description": "CTR below benchmark - refresh creative assets",
                "actions": ["refresh_creative", "a_b_test_headlines"]
            })
        
        # High-performing channel scaling
        best_channel = None
        best_roas = 0
        for channel, perf in channel_performance.items():
            if perf.get('roas', 0) > best_roas:
                best_roas = perf.get('roas', 0)
                best_channel = channel
        
        if best_channel and best_roas > settings.MIN_ROAS_THRESHOLD * 1.5:
            opportunities.append({
                "type": "scale_top_channel",
                "priority": "high",
                "description": f"Scale {best_channel} - highest ROAS at {best_roas:.2f}",
                "actions": ["increase_budget"],
                "channel": best_channel
            })
        
        return opportunities
    
    def _create_fallback_strategy(self, campaign_type: str, target_segment: str, 
                                channels: List[str]) -> Dict[str, Any]:
        """Create fallback strategy when AI generation fails"""
        return {
            "positioning": {
                "key_message": f"Engage {target_segment} with {campaign_type} campaign"
            },
            "value_propositions": {
                channel: ["Quality products", "Great service", "Competitive prices"]
                for channel in channels
            },
            "audience_insights": {
                "demographics": {"age_range": "25-55"},
                "interests": ["shopping", "deals"],
                "behaviors": ["online_purchaser"]
            }
        }
    
    def _create_fallback_creative(self, spec: Dict, channel: str, campaign_type: str) -> CreativeAsset:
        """Create fallback creative when AI generation fails"""
        fallback_content = {
            "headline": f"Special {campaign_type.title()} Offer",
            "description": "Don't miss out on great deals. Shop now!",
            "call_to_action": "Shop Now"
        }
        
        return CreativeAsset(
            asset_id=f"ASSET_{uuid.uuid4().hex[:8].upper()}",
            asset_type=spec['type'],
            content=json.dumps(fallback_content),
            metadata={"channel": channel, "fallback": True},
            performance_score=0.0,
            a_b_test_variant="A"
        )
    
    def _aggregate_creative_assets(self, channel_campaigns: Dict) -> List[CreativeAsset]:
        """Aggregate creative assets from all channels"""
        all_assets = []
        for channel_data in channel_campaigns.values():
            all_assets.extend(channel_data.get('creative_assets', []))
        return all_assets
    
    def get_all_campaigns(self) -> Dict[str, Any]:
        """Get overview of all campaigns"""
        return {
            "active_campaigns": len(self.active_campaigns),
            "campaigns": {
                campaign_id: {
                    "name": campaign.name,
                    "type": campaign.campaign_type,
                    "budget": campaign.budget,
                    "status": campaign.status,
                    "performance": campaign.performance_metrics
                }
                for campaign_id, campaign in self.active_campaigns.items()
            }
        }