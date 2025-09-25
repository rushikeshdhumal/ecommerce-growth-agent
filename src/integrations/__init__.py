"""
Platform integrations package for E-commerce Growth Agent
Mock and real API integrations for marketing platforms
"""

import logging
from typing import Dict, List, Any, Optional, Protocol, runtime_checkable
from abc import ABC, abstractmethod

# Package metadata
__version__ = "1.0.0"
__author__ = "E-commerce Growth Agent Team"

logger = logging.getLogger(__name__)

# Platform interface protocol
@runtime_checkable
class PlatformInterface(Protocol):
    """Protocol defining the interface for marketing platform integrations"""
    
    def create_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new campaign"""
        ...
    
    def get_campaign_metrics(self, campaign_id: str) -> Dict[str, float]:
        """Get performance metrics for a campaign"""
        ...
    
    def update_campaign_budget(self, campaign_id: str, new_budget: float) -> Dict[str, Any]:
        """Update campaign budget"""
        ...
    
    def pause_campaign(self, campaign_id: str) -> Dict[str, Any]:
        """Pause a campaign"""
        ...
    
    def enable_campaign(self, campaign_id: str) -> Dict[str, Any]:
        """Enable a paused campaign"""
        ...
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information"""
        ...


# Abstract base class for platform integrations
class BasePlatformIntegration(ABC):
    """Base class for all platform integrations"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.platform_name = "unknown"
        self.is_mock = True
        self.api_version = "1.0"
    
    @abstractmethod
    def create_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_campaign_metrics(self, campaign_id: str) -> Dict[str, float]:
        pass
    
    @abstractmethod
    def update_campaign_budget(self, campaign_id: str, new_budget: float) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def pause_campaign(self, campaign_id: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def enable_campaign(self, campaign_id: str) -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict[str, Any]:
        pass
    
    def get_platform_info(self) -> Dict[str, Any]:
        """Get platform integration information"""
        return {
            'platform_name': self.platform_name,
            'is_mock': self.is_mock,
            'api_version': self.api_version,
            'class_name': self.__class__.__name__
        }


# Import platform integrations with error handling
try:
    from .google_ads_mock import GoogleAdsMock
    _GOOGLE_ADS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Google Ads mock not available: {e}")
    _GOOGLE_ADS_AVAILABLE = False
    class GoogleAdsMock:
        def __init__(self):
            raise ImportError("Google Ads mock integration not available")

try:
    from .meta_ads_mock import MetaAdsMock
    _META_ADS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Meta Ads mock not available: {e}")
    _META_ADS_AVAILABLE = False
    class MetaAdsMock:
        def __init__(self):
            raise ImportError("Meta Ads mock integration not available")

try:
    from .klaviyo_mock import KlaviyoMock
    _KLAVIYO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Klaviyo mock not available: {e}")
    _KLAVIYO_AVAILABLE = False
    class KlaviyoMock:
        def __init__(self):
            raise ImportError("Klaviyo mock integration not available")

try:
    from .shopify_mock import ShopifyMock
    _SHOPIFY_AVAILABLE = True
except ImportError:
    _SHOPIFY_AVAILABLE = False
    class ShopifyMock(BasePlatformIntegration):
        def __init__(self):
            super().__init__()
            self.platform_name = "shopify"
            self.products = []
            self.orders = []
            self.customers = []
        
        def create_campaign(self, campaign_data):
            return {"success": False, "error": "Shopify doesn't support campaigns"}
        
        def get_campaign_metrics(self, campaign_id):
            return {"error": "Shopify doesn't support campaigns"}
        
        def update_campaign_budget(self, campaign_id, budget):
            return {"success": False, "error": "Shopify doesn't support campaigns"}
        
        def pause_campaign(self, campaign_id):
            return {"success": False, "error": "Shopify doesn't support campaigns"}
        
        def enable_campaign(self, campaign_id):
            return {"success": False, "error": "Shopify doesn't support campaigns"}
        
        def get_account_info(self):
            return {
                "shop_name": "Demo E-commerce Store",
                "plan": "Basic Shopify",
                "currency": "USD",
                "total_products": len(self.products),
                "total_orders": len(self.orders),
                "total_customers": len(self.customers)
            }

# Platform registry
PLATFORM_REGISTRY = {
    'google_ads': {
        'class': GoogleAdsMock,
        'available': _GOOGLE_ADS_AVAILABLE,
        'type': 'advertising',
        'channels': ['search', 'display', 'video'],
        'capabilities': ['campaigns', 'keywords', 'ads', 'audiences']
    },
    'meta_ads': {
        'class': MetaAdsMock,
        'available': _META_ADS_AVAILABLE,
        'type': 'advertising',
        'channels': ['facebook', 'instagram'],
        'capabilities': ['campaigns', 'ad_sets', 'ads', 'audiences']
    },
    'klaviyo': {
        'class': KlaviyoMock,
        'available': _KLAVIYO_AVAILABLE,
        'type': 'email_marketing',
        'channels': ['email', 'sms'],
        'capabilities': ['campaigns', 'flows', 'lists', 'segments']
    },
    'shopify': {
        'class': ShopifyMock,
        'available': _SHOPIFY_AVAILABLE,
        'type': 'ecommerce_platform',
        'channels': ['ecommerce'],
        'capabilities': ['products', 'orders', 'customers']
    }
}

# Package exports
__all__ = [
    # Interfaces and base classes
    'PlatformInterface',
    'BasePlatformIntegration',
    
    # Platform implementations
    'GoogleAdsMock',
    'MetaAdsMock', 
    'KlaviyoMock',
    'ShopifyMock',
    
    # Utility functions
    'get_platform_client',
    'get_available_platforms',
    'validate_platform_integration',
    'create_platform_manager',
    
    # Constants
    'PLATFORM_REGISTRY',
    'SUPPORTED_CHANNELS'
]

# Supported marketing channels
SUPPORTED_CHANNELS = {
    'google_ads': ['search', 'display', 'video'],
    'meta_ads': ['facebook', 'instagram', 'audience_network'],
    'email': ['email_campaigns', 'automated_flows'],
    'sms': ['sms_campaigns', 'automated_sms'],
    'display': ['banner_ads', 'native_ads']
}


def get_platform_client(platform_name: str, config: Optional[Dict[str, Any]] = None) -> Optional[BasePlatformIntegration]:
    """
    Factory function to create platform client instances
    
    Args:
        platform_name: Name of the platform (google_ads, meta_ads, klaviyo, shopify)
        config: Optional configuration dictionary
        
    Returns:
        Platform client instance or None if not available
    """
    if platform_name not in PLATFORM_REGISTRY:
        logger.error(f"Unknown platform: {platform_name}")
        return None
    
    platform_info = PLATFORM_REGISTRY[platform_name]
    
    if not platform_info['available']:
        logger.error(f"Platform not available: {platform_name}")
        return None
    
    try:
        client_class = platform_info['class']
        client = client_class()
        
        logger.info(f"Created {platform_name} client successfully")
        return client
        
    except Exception as e:
        logger.error(f"Failed to create {platform_name} client: {e}")
        return None


def get_available_platforms() -> Dict[str, Dict[str, Any]]:
    """
    Get list of available platform integrations
    
    Returns:
        Dictionary of available platforms with their info
    """
    available = {}
    
    for platform_name, platform_info in PLATFORM_REGISTRY.items():
        if platform_info['available']:
            available[platform_name] = {
                'type': platform_info['type'],
                'channels': platform_info['channels'],
                'capabilities': platform_info['capabilities']
            }
    
    return available


def validate_platform_integration(platform_name: str) -> Dict[str, Any]:
    """
    Validate a platform integration
    
    Args:
        platform_name: Name of the platform to validate
        
    Returns:
        Validation results dictionary
    """
    validation_results = {
        'valid': False,
        'errors': [],
        'warnings': [],
        'capabilities': []
    }
    
    if platform_name not in PLATFORM_REGISTRY:
        validation_results['errors'].append(f"Unknown platform: {platform_name}")
        return validation_results
    
    platform_info = PLATFORM_REGISTRY[platform_name]
    
    if not platform_info['available']:
        validation_results['errors'].append(f"Platform not available: {platform_name}")
        return validation_results
    
    try:
        # Test client creation
        client = get_platform_client(platform_name)
        
        if client is None:
            validation_results['errors'].append(f"Failed to create {platform_name} client")
            return validation_results
        
        # Test interface compliance
        if not isinstance(client, BasePlatformIntegration):
            validation_results['warnings'].append(f"{platform_name} doesn't inherit from BasePlatformIntegration")
        
        # Test required methods
        required_methods = [
            'create_campaign', 'get_campaign_metrics', 'update_campaign_budget',
            'pause_campaign', 'enable_campaign', 'get_account_info'
        ]
        
        for method_name in required_methods:
            if not hasattr(client, method_name) or not callable(getattr(client, method_name)):
                validation_results['errors'].append(f"{platform_name} missing required method: {method_name}")
        
        # Test basic functionality
        try:
            account_info = client.get_account_info()
            if isinstance(account_info, dict):
                validation_results['capabilities'].append('account_info')
        except Exception as e:
            validation_results['warnings'].append(f"get_account_info failed: {e}")
        
        if not validation_results['errors']:
            validation_results['valid'] = True
            validation_results['capabilities'] = platform_info['capabilities']
        
    except Exception as e:
        validation_results['errors'].append(f"Validation failed: {e}")
    
    return validation_results


class PlatformManager:
    """Manager class for handling multiple platform integrations"""
    
    def __init__(self, platforms_config: Optional[Dict[str, Dict[str, Any]]] = None):
        self.platforms = {}
        self.config = platforms_config or {}
        self.logger = logging.getLogger(f"{__name__}.PlatformManager")
        
        self._initialize_platforms()
    
    def _initialize_platforms(self):
        """Initialize all available platform clients"""
        for platform_name in PLATFORM_REGISTRY:
            if PLATFORM_REGISTRY[platform_name]['available']:
                try:
                    platform_config = self.config.get(platform_name, {})
                    client = get_platform_client(platform_name, platform_config)
                    
                    if client:
                        self.platforms[platform_name] = client
                        self.logger.info(f"Initialized {platform_name} integration")
                
                except Exception as e:
                    self.logger.error(f"Failed to initialize {platform_name}: {e}")
    
    def get_platform(self, platform_name: str) -> Optional[BasePlatformIntegration]:
        """Get a specific platform client"""
        return self.platforms.get(platform_name)
    
    def get_available_platforms(self) -> List[str]:
        """Get list of available platform names"""
        return list(self.platforms.keys())
    
    def create_campaign_multi_platform(self, platforms: List[str], campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create campaign across multiple platforms"""
        results = {}
        
        for platform_name in platforms:
            if platform_name in self.platforms:
                try:
                    client = self.platforms[platform_name]
                    result = client.create_campaign(campaign_data)
                    results[platform_name] = result
                
                except Exception as e:
                    results[platform_name] = {
                        'success': False,
                        'error': str(e)
                    }
            else:
                results[platform_name] = {
                    'success': False,
                    'error': 'Platform not available'
                }
        
        return results
    
    def get_metrics_multi_platform(self, campaign_mapping: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """Get metrics from multiple platforms"""
        results = {}
        
        for platform_name, campaign_id in campaign_mapping.items():
            if platform_name in self.platforms:
                try:
                    client = self.platforms[platform_name]
                    metrics = client.get_campaign_metrics(campaign_id)
                    results[platform_name] = metrics
                
                except Exception as e:
                    results[platform_name] = {'error': str(e)}
        
        return results
    
    def validate_all_platforms(self) -> Dict[str, Dict[str, Any]]:
        """Validate all platform integrations"""
        validation_results = {}
        
        for platform_name in self.platforms:
            validation_results[platform_name] = validate_platform_integration(platform_name)
        
        return validation_results


def create_platform_manager(config: Optional[Dict[str, Dict[str, Any]]] = None) -> PlatformManager:
    """
    Factory function to create a platform manager instance
    
    Args:
        config: Optional configuration for platforms
        
    Returns:
        PlatformManager instance
    """
    return PlatformManager(config)


# Integration status
INTEGRATION_STATUS = {
    'total_platforms': len(PLATFORM_REGISTRY),
    'available_platforms': sum(1 for p in PLATFORM_REGISTRY.values() if p['available']),
    'platforms': {name: info['available'] for name, info in PLATFORM_REGISTRY.items()},
    'advertising_platforms': sum(1 for p in PLATFORM_REGISTRY.values() if p['available'] and p['type'] == 'advertising'),
    'email_platforms': sum(1 for p in PLATFORM_REGISTRY.values() if p['available'] and p['type'] == 'email_marketing'),
    'ecommerce_platforms': sum(1 for p in PLATFORM_REGISTRY.values() if p['available'] and p['type'] == 'ecommerce_platform')
}

# Show initialization status
def _show_integration_status():
    """Show integration initialization status"""
    available_count = INTEGRATION_STATUS['available_platforms']
    total_count = INTEGRATION_STATUS['total_platforms']
    
    logger.info(f"Platform integrations: {available_count}/{total_count} available")
    
    for platform_name, available in INTEGRATION_STATUS['platforms'].items():
        status_icon = "✅" if available else "❌"
        logger.debug(f"{status_icon} {platform_name}")

# Auto-show status
try:
    _show_integration_status()
except Exception:
    pass

# Version for easy access
version = __version__