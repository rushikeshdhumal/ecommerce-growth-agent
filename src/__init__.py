"""
Core source package for E-commerce Growth Agent
Main entry point for all agent functionality and components
"""

import logging
import warnings
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

# Package metadata
__version__ = "1.0.0"
__author__ = "E-commerce Growth Agent Team"
__description__ = "Autonomous AI-powered e-commerce marketing optimization system"
__license__ = "MIT"

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress common warnings during imports
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

# Core component imports with error handling
try:
    from .agent import (
        EcommerceGrowthAgent,
        AgentPhase,
        AgentState, 
        AgentDecision
    )
    _AGENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Agent core not available: {e}")
    _AGENT_AVAILABLE = False
    # Create dummy classes for graceful degradation
    class EcommerceGrowthAgent:
        def __init__(self):
            raise ImportError("Agent core not available")
    class AgentPhase:
        pass
    class AgentState:
        pass
    class AgentDecision:
        pass

try:
    from .data_pipeline import (
        DataPipeline,
        CustomerSegment,
        MarketOpportunity
    )
    _DATA_PIPELINE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Data pipeline not available: {e}")
    _DATA_PIPELINE_AVAILABLE = False
    class DataPipeline:
        def __init__(self):
            raise ImportError("Data pipeline not available")
    class CustomerSegment:
        pass
    class MarketOpportunity:
        pass

try:
    from .campaign_manager import (
        CampaignManager,
        Campaign,
        CreativeAsset
    )
    _CAMPAIGN_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Campaign manager not available: {e}")
    _CAMPAIGN_MANAGER_AVAILABLE = False
    class CampaignManager:
        def __init__(self):
            raise ImportError("Campaign manager not available")
    class Campaign:
        pass
    class CreativeAsset:
        pass

try:
    from .evaluation import (
        EvaluationSystem,
        PerformanceMetric,
        ABTestResult,
        AnomalyDetection
    )
    _EVALUATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Evaluation system not available: {e}")
    _EVALUATION_AVAILABLE = False
    class EvaluationSystem:
        def __init__(self):
            raise ImportError("Evaluation system not available")
    class PerformanceMetric:
        pass
    class ABTestResult:
        pass
    class AnomalyDetection:
        pass

try:
    from .utils import (
        format_currency,
        format_percentage,
        format_large_number,
        calculate_percentage_change,
        generate_color_palette,
        MetricsCalculator,
        ColorGenerator
    )
    _UTILS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Utilities not available: {e}")
    _UTILS_AVAILABLE = False
    # Provide basic fallback implementations
    def format_currency(amount, currency="USD"):
        return f"${amount:,.2f}" if currency == "USD" else f"{amount:,.2f} {currency}"
    def format_percentage(value, decimal_places=2):
        return f"{value:.{decimal_places}f}%"
    def format_large_number(number):
        if number >= 1_000_000:
            return f"{number/1_000_000:.1f}M"
        elif number >= 1_000:
            return f"{number/1_000:.1f}K"
        else:
            return f"{number:,.0f}"
    def calculate_percentage_change(current, previous):
        return ((current - previous) / previous) * 100 if previous != 0 else 0.0
    def generate_color_palette(n_colors):
        return ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:n_colors]
    class MetricsCalculator:
        pass
    class ColorGenerator:
        pass

# Integration imports with error handling
try:
    from .integrations.google_ads_mock import GoogleAdsMock
    _GOOGLE_ADS_AVAILABLE = True
except ImportError:
    _GOOGLE_ADS_AVAILABLE = False
    class GoogleAdsMock:
        def __init__(self):
            raise ImportError("Google Ads integration not available")

try:
    from .integrations.meta_ads_mock import MetaAdsMock
    _META_ADS_AVAILABLE = True
except ImportError:
    _META_ADS_AVAILABLE = False
    class MetaAdsMock:
        def __init__(self):
            raise ImportError("Meta Ads integration not available")

try:
    from .integrations.klaviyo_mock import KlaviyoMock
    _KLAVIYO_AVAILABLE = True
except ImportError:
    _KLAVIYO_AVAILABLE = False
    class KlaviyoMock:
        def __init__(self):
            raise ImportError("Klaviyo integration not available")

# Package exports
__all__ = [
    # Core classes
    'EcommerceGrowthAgent',
    'DataPipeline', 
    'CampaignManager',
    'EvaluationSystem',
    
    # Data structures
    'AgentPhase',
    'AgentState',
    'AgentDecision',
    'CustomerSegment',
    'MarketOpportunity',
    'Campaign',
    'CreativeAsset',
    'PerformanceMetric',
    'ABTestResult',
    'AnomalyDetection',
    
    # Integrations
    'GoogleAdsMock',
    'MetaAdsMock', 
    'KlaviyoMock',
    
    # Utilities
    'format_currency',
    'format_percentage',
    'format_large_number',
    'calculate_percentage_change',
    'generate_color_palette',
    'MetricsCalculator',
    'ColorGenerator',
    
    # System functions
    'initialize_system',
    'create_agent_instance',
    'get_system_status',
    'validate_system',
    'get_version_info'
]

# System availability status
SYSTEM_STATUS = {
    'agent_core': _AGENT_AVAILABLE,
    'data_pipeline': _DATA_PIPELINE_AVAILABLE,
    'campaign_manager': _CAMPAIGN_MANAGER_AVAILABLE,
    'evaluation_system': _EVALUATION_AVAILABLE,
    'utils': _UTILS_AVAILABLE,
    'google_ads': _GOOGLE_ADS_AVAILABLE,
    'meta_ads': _META_ADS_AVAILABLE,
    'klaviyo': _KLAVIYO_AVAILABLE
}


def initialize_system(config_override: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Initialize the complete E-commerce Growth Agent system
    
    Args:
        config_override: Optional configuration overrides
        
    Returns:
        Initialization results dictionary
    """
    initialization_results = {
        'success': True,
        'errors': [],
        'warnings': [],
        'components_initialized': [],
        'components_failed': []
    }
    
    try:
        # Initialize configuration
        try:
            from config import initialize_config, settings
            if initialize_config():
                initialization_results['components_initialized'].append('configuration')
                logger.info("Configuration initialized successfully")
            else:
                initialization_results['components_failed'].append('configuration')
                initialization_results['warnings'].append("Configuration initialization failed")
        except ImportError as e:
            initialization_results['warnings'].append(f"Configuration not available: {e}")
        
        # Initialize data layer
        try:
            from data import initialize_database, validate_database
            if initialize_database():
                validation = validate_database()
                if validation['valid']:
                    initialization_results['components_initialized'].append('database')
                    logger.info("Database initialized and validated successfully")
                else:
                    initialization_results['warnings'].append(f"Database validation issues: {validation['errors']}")
            else:
                initialization_results['components_failed'].append('database')
        except ImportError as e:
            initialization_results['warnings'].append(f"Data layer not available: {e}")
        
        # Test core components
        components_to_test = [
            ('agent_core', _AGENT_AVAILABLE),
            ('data_pipeline', _DATA_PIPELINE_AVAILABLE),
            ('campaign_manager', _CAMPAIGN_MANAGER_AVAILABLE),
            ('evaluation_system', _EVALUATION_AVAILABLE)
        ]
        
        for component_name, available in components_to_test:
            if available:
                initialization_results['components_initialized'].append(component_name)
                logger.debug(f"{component_name} is available")
            else:
                initialization_results['components_failed'].append(component_name)
                initialization_results['errors'].append(f"{component_name} is not available")
        
        # Check if minimum required components are available
        required_components = ['agent_core', 'data_pipeline', 'campaign_manager']
        missing_required = [comp for comp in required_components 
                          if comp in initialization_results['components_failed']]
        
        if missing_required:
            initialization_results['success'] = False
            initialization_results['errors'].append(f"Missing required components: {missing_required}")
        
        # Apply configuration overrides
        if config_override and 'settings' in locals():
            for key, value in config_override.items():
                if hasattr(settings, key):
                    setattr(settings, key, value)
                    logger.info(f"Applied config override: {key} = {value}")
        
        logger.info(f"System initialization completed. Success: {initialization_results['success']}")
        
    except Exception as e:
        initialization_results['success'] = False
        initialization_results['errors'].append(f"System initialization failed: {e}")
        logger.error(f"System initialization failed: {e}")
    
    return initialization_results


def create_agent_instance(config: Optional[Dict[str, Any]] = None) -> Union[EcommerceGrowthAgent, None]:
    """
    Create a properly initialized agent instance
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        EcommerceGrowthAgent instance or None if creation fails
    """
    try:
        if not _AGENT_AVAILABLE:
            logger.error("Agent core not available - cannot create agent instance")
            return None
        
        # Initialize system first
        init_result = initialize_system(config)
        if not init_result['success']:
            logger.error(f"System initialization failed: {init_result['errors']}")
            return None
        
        # Create agent instance
        agent = EcommerceGrowthAgent()
        
        logger.info("Agent instance created successfully")
        return agent
        
    except Exception as e:
        logger.error(f"Failed to create agent instance: {e}")
        return None


def get_system_status() -> Dict[str, Any]:
    """
    Get comprehensive system status information
    
    Returns:
        System status dictionary
    """
    status = {
        'version': __version__,
        'components_available': SYSTEM_STATUS,
        'core_components_ready': all([
            _AGENT_AVAILABLE,
            _DATA_PIPELINE_AVAILABLE, 
            _CAMPAIGN_MANAGER_AVAILABLE,
            _EVALUATION_AVAILABLE
        ]),
        'integrations_available': {
            'google_ads': _GOOGLE_ADS_AVAILABLE,
            'meta_ads': _META_ADS_AVAILABLE,
            'klaviyo': _KLAVIYO_AVAILABLE
        },
        'utilities_available': _UTILS_AVAILABLE,
        'system_ready': True
    }
    
    # Check for critical missing components
    critical_components = ['agent_core', 'data_pipeline', 'campaign_manager']
    missing_critical = [comp for comp in critical_components 
                       if not SYSTEM_STATUS.get(comp, False)]
    
    if missing_critical:
        status['system_ready'] = False
        status['missing_critical_components'] = missing_critical
    
    # Get configuration status if available
    try:
        from config import get_config_summary, validate_config
        status['configuration'] = {
            'summary': get_config_summary(),
            'validation': validate_config()
        }
    except ImportError:
        status['configuration'] = {'error': 'Configuration module not available'}
    
    # Get database status if available
    try:
        from data import get_database_stats, validate_database
        status['database'] = {
            'stats': get_database_stats(),
            'validation': validate_database()
        }
    except ImportError:
        status['database'] = {'error': 'Data module not available'}
    
    return status


def validate_system() -> Dict[str, Any]:
    """
    Perform comprehensive system validation
    
    Returns:
        Validation results dictionary
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'component_validations': {}
    }
    
    # Validate core components
    for component_name, available in SYSTEM_STATUS.items():
        if component_name in ['agent_core', 'data_pipeline', 'campaign_manager', 'evaluation_system']:
            if not available:
                validation_results['valid'] = False
                validation_results['errors'].append(f"Critical component missing: {component_name}")
        elif not available:
            validation_results['warnings'].append(f"Optional component missing: {component_name}")
    
    # Validate configuration
    try:
        from config import validate_config
        config_validation = validate_config()
        validation_results['component_validations']['configuration'] = config_validation
        
        if not config_validation['valid']:
            validation_results['valid'] = False
            validation_results['errors'].extend(config_validation['errors'])
        
        validation_results['warnings'].extend(config_validation['warnings'])
        
    except ImportError:
        validation_results['warnings'].append("Configuration validation not available")
    
    # Validate database
    try:
        from data import validate_database
        db_validation = validate_database()
        validation_results['component_validations']['database'] = db_validation
        
        if not db_validation['valid']:
            validation_results['warnings'].append("Database validation issues found")
        
    except ImportError:
        validation_results['warnings'].append("Database validation not available")
    
    # Test agent creation
    if validation_results['valid']:
        try:
            test_agent = create_agent_instance()
            if test_agent is None:
                validation_results['valid'] = False
                validation_results['errors'].append("Failed to create agent instance")
            else:
                validation_results['component_validations']['agent_creation'] = True
                # Clean up test agent
                del test_agent
        except Exception as e:
            validation_results['valid'] = False
            validation_results['errors'].append(f"Agent creation test failed: {e}")
    
    return validation_results


def get_version_info() -> Dict[str, str]:
    """
    Get detailed version information for all components
    
    Returns:
        Version information dictionary
    """
    version_info = {
        'package_version': __version__,
        'package_author': __author__,
        'package_license': __license__,
        'description': __description__
    }
    
    # Get Python version
    import sys
    version_info['python_version'] = sys.version.split()[0]
    
    # Get key dependency versions
    try:
        import pandas
        version_info['pandas_version'] = pandas.__version__
    except ImportError:
        version_info['pandas_version'] = 'not available'
    
    try:
        import numpy
        version_info['numpy_version'] = numpy.__version__
    except ImportError:
        version_info['numpy_version'] = 'not available'
    
    try:
        import sklearn
        version_info['sklearn_version'] = sklearn.__version__
    except ImportError:
        version_info['sklearn_version'] = 'not available'
    
    try:
        import streamlit
        version_info['streamlit_version'] = streamlit.__version__
    except ImportError:
        version_info['streamlit_version'] = 'not available'
    
    try:
        import plotly
        version_info['plotly_version'] = plotly.__version__
    except ImportError:
        version_info['plotly_version'] = 'not available'
    
    # Get AI library versions
    try:
        import openai
        version_info['openai_version'] = openai.__version__
    except ImportError:
        version_info['openai_version'] = 'not available'
    
    try:
        import anthropic
        version_info['anthropic_version'] = anthropic.__version__
    except ImportError:
        version_info['anthropic_version'] = 'not available'
    
    return version_info


# Quick access functions for common operations
def quick_demo() -> Dict[str, Any]:
    """
    Quick demonstration of system capabilities
    
    Returns:
        Demo results dictionary
    """
    demo_results = {
        'success': False,
        'steps_completed': [],
        'errors': []
    }
    
    try:
        # Step 1: System validation
        validation = validate_system()
        if validation['valid']:
            demo_results['steps_completed'].append('system_validation')
        else:
            demo_results['errors'].extend(validation['errors'])
            return demo_results
        
        # Step 2: Create agent
        agent = create_agent_instance()
        if agent:
            demo_results['steps_completed'].append('agent_creation')
        else:
            demo_results['errors'].append('Failed to create agent')
            return demo_results
        
        # Step 3: Get system status
        status = get_system_status()
        demo_results['system_status'] = status
        demo_results['steps_completed'].append('status_check')
        
        # Step 4: Run quick test
        if hasattr(agent, 'get_agent_status'):
            agent_status = agent.get_agent_status()
            demo_results['agent_status'] = agent_status
            demo_results['steps_completed'].append('agent_status')
        
        demo_results['success'] = True
        logger.info("Quick demo completed successfully")
        
    except Exception as e:
        demo_results['errors'].append(f"Demo failed: {e}")
        logger.error(f"Quick demo failed: {e}")
    
    return demo_results


# Package initialization message
def _show_initialization_status():
    """Show package initialization status"""
    available_components = sum(SYSTEM_STATUS.values())
    total_components = len(SYSTEM_STATUS)
    
    logger.info(f"E-commerce Growth Agent v{__version__}")
    logger.info(f"Components available: {available_components}/{total_components}")
    
    if not SYSTEM_STATUS['agent_core']:
        logger.warning("⚠️  Agent core not available - system functionality limited")
    
    if all(SYSTEM_STATUS[comp] for comp in ['agent_core', 'data_pipeline', 'campaign_manager']):
        logger.info("✅ Core system ready")
    else:
        logger.warning("⚠️  Some core components missing - check installation")


# Show status on import
try:
    _show_initialization_status()
except Exception:
    pass  # Suppress errors during import

# Version for easy access
version = __version__