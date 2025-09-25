"""
Mock Klaviyo API integration for email and SMS marketing automation
"""
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

from config.logging_config import get_agent_logger


@dataclass
class KlaviyoAccount:
    """Klaviyo account structure"""
    account_id: str
    account_name: str
    company_name: str
    time_zone: str
    contact_information: Dict[str, str]


@dataclass
class EmailCampaign:
    """Email campaign structure"""
    campaign_id: str
    name: str
    status: str
    campaign_type: str
    subject_line: str
    from_email: str
    from_name: str
    template_id: Optional[str]
    send_time: Optional[datetime]


@dataclass
class SMSCampaign:
    """SMS campaign structure"""
    campaign_id: str
    name: str
    status: str
    message_body: str
    send_time: Optional[datetime]
    from_number: str


class KlaviyoMock:
    """
    Mock Klaviyo API client for email and SMS marketing automation
    """
    
    def __init__(self):
        self.logger = get_agent_logger("KlaviyoMock")
        
        # Mock account data
        self.account = KlaviyoAccount(
            account_id="PK_12345abcdef",
            account_name="Demo E-commerce Store",
            company_name="Demo Company",
            time_zone="America/New_York",
            contact_information={
                "organization_name": "Demo Company",
                "email": "marketing@demo-company.com",
                "phone": "+1-555-0123"
            }
        )
        
        # Campaign storage
        self.email_campaigns: Dict[str, EmailCampaign] = {}
        self.sms_campaigns: Dict[str, SMSCampaign] = {}
        self.campaign_metrics: Dict[str, Dict[str, float]] = {}
        
        # Lists and segments
        self.lists: Dict[str, Dict] = {}
        self.segments: Dict[str, Dict] = {}
        
        # Templates
        self.email_templates: Dict[str, Dict] = {}
        
        # Mock subscriber data
        self.subscribers = self._generate_mock_subscribers()
        
        self.logger.log_action("klaviyo_mock_initialized", {
            "account_id": self.account.account_id,
            "subscriber_count": len(self.subscribers)
        })
    
    def _generate_mock_subscribers(self) -> List[Dict]:
        """Generate mock subscriber data"""
        subscribers = []
        
        for i in range(5000):  # 5k mock subscribers
            subscriber = {
                'id': f"SUB_{uuid.uuid4().hex[:8].upper()}",
                'email': f"user{i}@example.com",
                'phone_number': f"+1555{random.randint(1000000, 9999999)}",
                'first_name': f"User{i}",
                'last_name': f"Last{i}",
                'created': datetime.now() - timedelta(days=random.randint(1, 365)),
                'updated': datetime.now() - timedelta(days=random.randint(0, 30)),
                'opted_in_to_email': random.choice([True, False]),
                'opted_in_to_sms': random.choice([True, False]),
                'properties': {
                    'total_orders': random.randint(0, 20),
                    'total_spent': random.uniform(0, 2000),
                    'last_order_date': datetime.now() - timedelta(days=random.randint(0, 180)),
                    'preferred_category': random.choice(['Electronics', 'Clothing', 'Home', 'Books']),
                    'acquisition_source': random.choice(['organic', 'paid_search', 'social', 'email']),
                    'location': random.choice(['US', 'CA', 'UK', 'AU'])
                }
            }
            subscribers.append(subscriber)
        
        return subscribers
    
    def create_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create email or SMS campaign based on campaign data"""
        channel = campaign_data.get('targeting', {}).get('channel', 'email')
        
        if channel == 'sms':
            return self.create_sms_campaign(campaign_data)
        else:
            return self.create_email_campaign(campaign_data)
    
    def create_email_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new email campaign"""
        campaign_id = f"EMAIL_{uuid.uuid4().hex[:10].upper()}"
        
        try:
            # Extract campaign parameters
            name = campaign_data.get('name', f'Email_Campaign_{campaign_id}')
            targeting = campaign_data.get('targeting', {})
            creative_assets = campaign_data.get('creative_assets', [])
            
            # Extract email creative content
            email_creative = None
            for asset in creative_assets:
                if asset.get('asset_type') == 'email_campaign':
                    email_creative = asset
                    break
            
            if email_creative:
                try:
                    import json
                    content = json.loads(email_creative.get('content', '{}'))
                    subject_line = content.get('subject_line', 'Special Offer Just for You!')
                    from_name = content.get('from_name', 'Demo Store')
                except:
                    subject_line = 'Special Offer Just for You!'
                    from_name = 'Demo Store'
            else:
                subject_line = 'Special Offer Just for You!'
                from_name = 'Demo Store'
            
            # Create campaign object
            campaign = EmailCampaign(
                campaign_id=campaign_id,
                name=name,
                status="DRAFT",
                campaign_type="REGULAR",
                subject_line=subject_line,
                from_email="no-reply@demo-company.com",
                from_name=from_name,
                template_id=None,
                send_time=None
            )
            
            # Store campaign
            self.email_campaigns[campaign_id] = campaign
            
            # Create recipient list based on targeting
            recipient_list = self._create_recipient_list(targeting, 'email')
            
            # Initialize performance metrics
            self._initialize_email_metrics(campaign_id, len(recipient_list))
            
            self.logger.log_action("klaviyo_email_campaign_created", {
                "campaign_id": campaign_id,
                "name": name,
                "subject_line": subject_line,
                "recipient_count": len(recipient_list)
            })
            
            return {
                "success": True,
                "platform_campaign_id": campaign_id,
                "campaign_name": name,
                "campaign_type": "email",
                "status": "DRAFT",
                "recipient_count": len(recipient_list),
                "message": "Email campaign created successfully"
            }
            
        except Exception as e:
            self.logger.log_error("klaviyo_email_campaign_creation_failed", str(e))
            return {
                "success": False,
                "error": str(e),
                "platform_campaign_id": None
            }
    
    def create_sms_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new SMS campaign"""
        campaign_id = f"SMS_{uuid.uuid4().hex[:10].upper()}"
        
        try:
            # Extract campaign parameters
            name = campaign_data.get('name', f'SMS_Campaign_{campaign_id}')
            targeting = campaign_data.get('targeting', {})
            creative_assets = campaign_data.get('creative_assets', [])
            
            # Extract SMS creative content
            sms_creative = None
            for asset in creative_assets:
                if asset.get('asset_type') == 'sms_message':
                    sms_creative = asset
                    break
            
            if sms_creative:
                try:
                    import json
                    content = json.loads(sms_creative.get('content', '{}'))
                    message_body = content.get('message', 'Exclusive offer! Shop now and save 20%. Reply STOP to opt out.')
                except:
                    message_body = 'Exclusive offer! Shop now and save 20%. Reply STOP to opt out.'
            else:
                message_body = 'Exclusive offer! Shop now and save 20%. Reply STOP to opt out.'
            
            # Create campaign object
            campaign = SMSCampaign(
                campaign_id=campaign_id,
                name=name,
                status="DRAFT",
                message_body=message_body,
                send_time=None,
                from_number="+15551234567"
            )
            
            # Store campaign
            self.sms_campaigns[campaign_id] = campaign
            
            # Create recipient list based on targeting
            recipient_list = self._create_recipient_list(targeting, 'sms')
            
            # Initialize performance metrics
            self._initialize_sms_metrics(campaign_id, len(recipient_list))
            
            self.logger.log_action("klaviyo_sms_campaign_created", {
                "campaign_id": campaign_id,
                "name": name,
                "message_length": len(message_body),
                "recipient_count": len(recipient_list)
            })
            
            return {
                "success": True,
                "platform_campaign_id": campaign_id,
                "campaign_name": name,
                "campaign_type": "sms",
                "status": "DRAFT",
                "recipient_count": len(recipient_list),
                "message": "SMS campaign created successfully"
            }
            
        except Exception as e:
            self.logger.log_error("klaviyo_sms_campaign_creation_failed", str(e))
            return {
                "success": False,
                "error": str(e),
                "platform_campaign_id": None
            }
    
    def _create_recipient_list(self, targeting: Dict, channel: str) -> List[Dict]:
        """Create recipient list based on targeting criteria"""
        # Filter subscribers based on opt-in status
        if channel == 'email':
            eligible_subscribers = [s for s in self.subscribers if s['opted_in_to_email']]
        else:
            eligible_subscribers = [s for s in self.subscribers if s['opted_in_to_sms']]
        
        # Apply targeting filters
        segment_filters = targeting.get('segment_filters', {})
        
        if segment_filters.get('purchase_history') == 'high_value':
            eligible_subscribers = [s for s in eligible_subscribers 
                                  if s['properties']['total_spent'] > 500]
        elif segment_filters.get('purchase_history') == 'low_value':
            eligible_subscribers = [s for s in eligible_subscribers 
                                  if s['properties']['total_spent'] < 100]
        
        if segment_filters.get('engagement_level') == 'low':
            # Simulate low engagement based on last order date
            cutoff_date = datetime.now() - timedelta(days=90)
            eligible_subscribers = [s for s in eligible_subscribers 
                                  if s['properties']['last_order_date'] < cutoff_date]
        
        # Apply random sampling if too many recipients
        if len(eligible_subscribers) > 10000:
            eligible_subscribers = random.sample(eligible_subscribers, 10000)
        
        return eligible_subscribers
    
    def _initialize_email_metrics(self, campaign_id: str, recipient_count: int):
        """Initialize email campaign performance metrics"""
        # Simulate realistic email performance
        delivered = int(recipient_count * random.uniform(0.95, 0.99))
        opened = int(delivered * random.uniform(0.18, 0.28))
        clicked = int(opened * random.uniform(0.15, 0.25))
        conversions = int(clicked * random.uniform(0.08, 0.15))
        unsubscribed = int(delivered * random.uniform(0.001, 0.005))
        
        metrics = {
            'recipients': recipient_count,
            'delivered': delivered,
            'bounced': recipient_count - delivered,
            'opened': opened,
            'clicked': clicked,
            'conversions': conversions,
            'unsubscribed': unsubscribed,
            'marked_as_spam': int(delivered * random.uniform(0.0001, 0.001)),
            'revenue': conversions * random.uniform(30, 100),
            
            # Calculated metrics
            'delivery_rate': delivered / recipient_count if recipient_count > 0 else 0,
            'open_rate': opened / delivered if delivered > 0 else 0,
            'click_rate': clicked / delivered if delivered > 0 else 0,
            'click_to_open_rate': clicked / opened if opened > 0 else 0,
            'conversion_rate': conversions / clicked if clicked > 0 else 0,
            'unsubscribe_rate': unsubscribed / delivered if delivered > 0 else 0,
            'revenue_per_recipient': 0  # Will be calculated
        }
        
        # Calculate revenue per recipient
        metrics['revenue_per_recipient'] = metrics['revenue'] / recipient_count if recipient_count > 0 else 0
        
        self.campaign_metrics[campaign_id] = metrics
    
    def _initialize_sms_metrics(self, campaign_id: str, recipient_count: int):
        """Initialize SMS campaign performance metrics"""
        # Simulate realistic SMS performance
        delivered = int(recipient_count * random.uniform(0.97, 0.995))
        clicked = int(delivered * random.uniform(0.05, 0.12))
        conversions = int(clicked * random.uniform(0.12, 0.25))
        opt_outs = int(delivered * random.uniform(0.002, 0.008))
        
        metrics = {
            'recipients': recipient_count,
            'delivered': delivered,
            'failed': recipient_count - delivered,
            'clicked': clicked,
            'conversions': conversions,
            'opt_outs': opt_outs,
            'revenue': conversions * random.uniform(25, 80),
            
            # Calculated metrics
            'delivery_rate': delivered / recipient_count if recipient_count > 0 else 0,
            'click_rate': clicked / delivered if delivered > 0 else 0,
            'conversion_rate': conversions / clicked if clicked > 0 else 0,
            'opt_out_rate': opt_outs / delivered if delivered > 0 else 0,
            'revenue_per_recipient': 0  # Will be calculated
        }
        
        # Calculate revenue per recipient
        metrics['revenue_per_recipient'] = metrics['revenue'] / recipient_count if recipient_count > 0 else 0
        
        self.campaign_metrics[campaign_id] = metrics
    
    def send_campaign(self, campaign_id: str, send_time: Optional[str] = None) -> Dict[str, Any]:
        """Send an email or SMS campaign"""
        # Check if it's email or SMS campaign
        if campaign_id in self.email_campaigns:
            campaign = self.email_campaigns[campaign_id]
            campaign.status = "SENT"
            campaign.send_time = datetime.now() if not send_time else datetime.fromisoformat(send_time)
            campaign_type = "email"
        elif campaign_id in self.sms_campaigns:
            campaign = self.sms_campaigns[campaign_id]
            campaign.status = "SENT"
            campaign.send_time = datetime.now() if not send_time else datetime.fromisoformat(send_time)
            campaign_type = "sms"
        else:
            return {"success": False, "error": "Campaign not found"}
        
        self.logger.log_action("klaviyo_campaign_sent", {
            "campaign_id": campaign_id,
            "campaign_type": campaign_type,
            "send_time": campaign.send_time.isoformat()
        })
        
        return {
            "success": True,
            "campaign_id": campaign_id,
            "status": "SENT",
            "send_time": campaign.send_time.isoformat(),
            "message": f"{campaign_type.title()} campaign sent successfully"
        }
    
    def get_campaign_metrics(self, campaign_id: str) -> Dict[str, float]:
        """Get performance metrics for a campaign"""
        if campaign_id not in self.campaign_metrics:
            return {"error": "Campaign not found"}
        
        # Simulate metric evolution over time
        metrics = self.campaign_metrics[campaign_id].copy()
        
        # Add some realistic variance (smaller for email/SMS)
        variance_factor = random.uniform(0.98, 1.02)
        
        # Apply variance to volume metrics
        volume_metrics = ['opened', 'clicked', 'conversions', 'revenue']
        for key in volume_metrics:
            if key in metrics:
                metrics[key] = metrics[key] * variance_factor
        
        # Recalculate rates
        if 'delivered' in metrics and metrics['delivered'] > 0:
            if 'opened' in metrics:
                metrics['open_rate'] = metrics['opened'] / metrics['delivered']
            metrics['click_rate'] = metrics['clicked'] / metrics['delivered']
        
        if 'opened' in metrics and metrics['opened'] > 0:
            metrics['click_to_open_rate'] = metrics['clicked'] / metrics['opened']
        
        if metrics['clicked'] > 0:
            metrics['conversion_rate'] = metrics['conversions'] / metrics['clicked']
        
        if metrics['recipients'] > 0:
            metrics['revenue_per_recipient'] = metrics['revenue'] / metrics['recipients']
        
        # Update stored metrics
        self.campaign_metrics[campaign_id] = metrics
        
        return metrics
    
    def create_list(self, list_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new subscriber list"""
        list_id = f"LIST_{uuid.uuid4().hex[:8].upper()}"
        
        list_obj = {
            'list_id': list_id,
            'name': list_data.get('name', f'List_{list_id}'),
            'created': datetime.now().isoformat(),
            'subscriber_count': 0,
            'opt_in_process': list_data.get('opt_in_process', 'SINGLE_OPT_IN'),
            'list_type': list_data.get('list_type', 'REGULAR')
        }
        
        self.lists[list_id] = list_obj
        
        self.logger.log_action("klaviyo_list_created", {
            "list_id": list_id,
            "name": list_obj['name']
        })
        
        return {
            "success": True,
            "list_id": list_id,
            "name": list_obj['name'],
            "message": "List created successfully"
        }
    
    def create_segment(self, segment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new dynamic segment"""
        segment_id = f"SEG_{uuid.uuid4().hex[:8].upper()}"
        
        segment = {
            'segment_id': segment_id,
            'name': segment_data.get('name', f'Segment_{segment_id}'),
            'created': datetime.now().isoformat(),
            'definition': segment_data.get('definition', {}),
            'estimated_size': random.randint(100, 5000),
            'segment_type': 'DYNAMIC'
        }
        
        self.segments[segment_id] = segment
        
        self.logger.log_action("klaviyo_segment_created", {
            "segment_id": segment_id,
            "name": segment['name'],
            "estimated_size": segment['estimated_size']
        })
        
        return {
            "success": True,
            "segment_id": segment_id,
            "name": segment['name'],
            "estimated_size": segment['estimated_size'],
            "message": "Segment created successfully"
        }
    
    def create_flow(self, flow_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an automated email flow"""
        flow_id = f"FLOW_{uuid.uuid4().hex[:8].upper()}"
        
        flow = {
            'flow_id': flow_id,
            'name': flow_data.get('name', f'Flow_{flow_id}'),
            'status': 'DRAFT',
            'trigger_type': flow_data.get('trigger_type', 'LIST_TRIGGER'),
            'created': datetime.now().isoformat(),
            'emails': flow_data.get('emails', []),
            'total_emails': len(flow_data.get('emails', [])),
            'estimated_recipients': random.randint(50, 1000)
        }
        
        self.logger.log_action("klaviyo_flow_created", {
            "flow_id": flow_id,
            "name": flow['name'],
            "trigger_type": flow['trigger_type']
        })
        
        return {
            "success": True,
            "flow_id": flow_id,
            "name": flow['name'],
            "status": "DRAFT",
            "message": "Flow created successfully"
        }
    
    def update_campaign_budget(self, campaign_id: str, new_budget: float) -> Dict[str, Any]:
        """Update campaign budget (not applicable for email/SMS, but included for interface consistency)"""
        # Email/SMS campaigns don't have traditional budgets, but we can simulate cost updates
        if campaign_id in self.campaign_metrics:
            # Simulate cost based on recipient count and pricing
            metrics = self.campaign_metrics[campaign_id]
            recipients = metrics.get('recipients', 0)
            
            if campaign_id.startswith('EMAIL_'):
                cost_per_send = 0.001  # $0.001 per email
            else:  # SMS
                cost_per_send = 0.05   # $0.05 per SMS
            
            estimated_cost = recipients * cost_per_send
            
            return {
                "success": True,
                "campaign_id": campaign_id,
                "estimated_cost": estimated_cost,
                "recipients": recipients,
                "message": "Budget information updated"
            }
        
        return {"success": False, "error": "Campaign not found"}
    
    def get_subscriber_data(self, filters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get subscriber data with optional filters"""
        filtered_subscribers = self.subscribers.copy()
        
        if filters:
            if filters.get('email_consent'):
                filtered_subscribers = [s for s in filtered_subscribers if s['opted_in_to_email']]
            
            if filters.get('sms_consent'):
                filtered_subscribers = [s for s in filtered_subscribers if s['opted_in_to_sms']]
            
            if filters.get('min_orders'):
                min_orders = filters['min_orders']
                filtered_subscribers = [s for s in filtered_subscribers 
                                      if s['properties']['total_orders'] >= min_orders]
        
        # Calculate aggregated metrics
        total_subscribers = len(filtered_subscribers)
        email_subscribers = len([s for s in filtered_subscribers if s['opted_in_to_email']])
        sms_subscribers = len([s for s in filtered_subscribers if s['opted_in_to_sms']])
        
        return {
            "total_subscribers": total_subscribers,
            "email_subscribers": email_subscribers,
            "sms_subscribers": sms_subscribers,
            "email_opt_in_rate": email_subscribers / total_subscribers if total_subscribers > 0 else 0,
            "sms_opt_in_rate": sms_subscribers / total_subscribers if total_subscribers > 0 else 0,
            "subscribers": filtered_subscribers[:100]  # Return first 100 for demo
        }
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        return {
            "account_id": self.account.account_id,
            "account_name": self.account.account_name,
            "company_name": self.account.company_name,
            "time_zone": self.account.time_zone,
            "total_email_campaigns": len(self.email_campaigns),
            "total_sms_campaigns": len(self.sms_campaigns),
            "total_subscribers": len(self.subscribers),
            "email_subscribers": len([s for s in self.subscribers if s['opted_in_to_email']]),
            "sms_subscribers": len([s for s in self.subscribers if s['opted_in_to_sms']])
        }
    
    def get_all_campaigns(self) -> List[Dict[str, Any]]:
        """Get list of all campaigns (email and SMS)"""
        campaigns_list = []
        
        # Add email campaigns
        for campaign in self.email_campaigns.values():
            metrics = self.campaign_metrics.get(campaign.campaign_id, {})
            
            campaigns_list.append({
                "campaign_id": campaign.campaign_id,
                "name": campaign.name,
                "type": "email",
                "status": campaign.status,
                "subject_line": campaign.subject_line,
                "send_time": campaign.send_time.isoformat() if campaign.send_time else None,
                "metrics": metrics
            })
        
        # Add SMS campaigns
        for campaign in self.sms_campaigns.values():
            metrics = self.campaign_metrics.get(campaign.campaign_id, {})
            
            campaigns_list.append({
                "campaign_id": campaign.campaign_id,
                "name": campaign.name,
                "type": "sms",
                "status": campaign.status,
                "message_body": campaign.message_body[:50] + "..." if len(campaign.message_body) > 50 else campaign.message_body,
                "send_time": campaign.send_time.isoformat() if campaign.send_time else None,
                "metrics": metrics
            })
        
        return campaigns_list
    
    def simulate_performance_change(self, campaign_id: str, change_factor: float):
        """Simulate performance change for testing optimization algorithms"""
        if campaign_id in self.campaign_metrics:
            metrics = self.campaign_metrics[campaign_id]
            
            # Apply change factor to key metrics
            for key in ['clicked', 'conversions', 'revenue']:
                if key in metrics:
                    metrics[key] = metrics[key] * change_factor
            
            # Recalculate derived metrics
            if metrics.get('delivered', 0) > 0:
                metrics['click_rate'] = metrics['clicked'] / metrics['delivered']
            if metrics.get('clicked', 0) > 0:
                metrics['conversion_rate'] = metrics['conversions'] / metrics['clicked']
            if metrics.get('recipients', 0) > 0:
                metrics['revenue_per_recipient'] = metrics['revenue'] / metrics['recipients']