"""
Configuration package for E-commerce Growth Agent
Centralizes all configuration management and settings
"""

# Version information
__version__ = "1.0.0"
__author__ = "E-commerce Growth Agent Team"

# Import main configuration components
from .settings import (
    Settings,
    settings,
    CAMPAIGN_TEMPLATES,
    MODEL_CONFIGS
)

from .logging_config import (
    setup_logging,
    get_agent_logger,
    AgentLogger
)

# Package-level exports
__all__ = [
    # Settings
    'Settings',
    'settings',
    'CAMPAIGN_TEMPLATES', 
    'MODEL_CONFIGS',
    
    # Logging
    'setup_logging',
    'get_agent_logger',
    'AgentLogger',
    
    # Utility functions
    'initialize_config',
    'validate_config',
    'get_config_summary'
]


def initialize_config():
    """
    Initialize all configuration components
    Should be called at application startup
    """
    try:
        # Setup logging first
        setup_logging()
        
        # Validate settings
        validation_result = validate_config()
        
        if validation_result['valid']:
            logger = get_agent_logger("ConfigInit")
            logger.log_action("config_initialized", {
                "version": __version__,
                "model": settings.AGENT_MODEL,
                "environment": getattr(settings, 'ENVIRONMENT', 'development')
            })
            return True
        else:
            print(f"Configuration validation failed: {validation_result['errors']}")
            return False
            
    except Exception as e:
        print(f"Failed to initialize configuration: {e}")
        return False


def validate_config():
    """
    Validate configuration settings
    Returns dict with validation results
    """
    errors = []
    warnings = []
    
    # Validate required settings
    if not settings.AGENT_MODEL:
        errors.append("AGENT_MODEL is required")
    
    # Validate API keys (warn if missing)
    if not settings.OPENAI_API_KEY and not settings.ANTHROPIC_API_KEY:
        warnings.append("No AI API keys configured - system will use fallback responses")
    
    # Validate numeric settings
    if settings.MAX_DAILY_BUDGET <= 0:
        errors.append("MAX_DAILY_BUDGET must be positive")
    
    if settings.MIN_ROAS_THRESHOLD <= 0:
        errors.append("MIN_ROAS_THRESHOLD must be positive")
    
    if not (0 <= settings.TEMPERATURE <= 1):
        errors.append("TEMPERATURE must be between 0 and 1")
    
    # Validate performance thresholds
    thresholds = settings.PERFORMANCE_THRESHOLDS
    if thresholds.get('min_ctr', 0) <= 0:
        errors.append("min_ctr threshold must be positive")
    
    if thresholds.get('min_conversion_rate', 0) <= 0:
        errors.append("min_conversion_rate threshold must be positive")
    
    # Validate available channels
    if not settings.AVAILABLE_CHANNELS:
        errors.append("At least one marketing channel must be available")
    
    # Validate sample data size
    if settings.SAMPLE_DATA_SIZE < 100:
        warnings.append("SAMPLE_DATA_SIZE is very low - may affect segmentation quality")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }


def get_config_summary():
    """
    Get a summary of current configuration
    Useful for debugging and system status
    """
    return {
        'version': __version__,
        'agent_model': settings.AGENT_MODEL,
        'temperature': settings.TEMPERATURE,
        'max_iterations': settings.MAX_ITERATIONS,
        'max_daily_budget': settings.MAX_DAILY_BUDGET,
        'min_roas_threshold': settings.MIN_ROAS_THRESHOLD,
        'available_channels': settings.AVAILABLE_CHANNELS,
        'customer_segments': settings.CUSTOMER_SEGMENT_COUNT,
        'sample_data_size': settings.SAMPLE_DATA_SIZE,
        'log_level': settings.LOG_LEVEL,
        'database_url': settings.DATABASE_URL,
        'has_openai_key': bool(settings.OPENAI_API_KEY),
        'has_anthropic_key': bool(settings.ANTHROPIC_API_KEY),
        'performance_thresholds': settings.PERFORMANCE_THRESHOLDS,
        'campaign_templates': list(CAMPAIGN_TEMPLATES.keys()),
        'supported_models': list(MODEL_CONFIGS.keys())
    }


# Configuration constants
DEFAULT_CONFIG = {
    'AGENT_MODEL': 'gpt-4',
    'TEMPERATURE': 0.7,
    'MAX_ITERATIONS': 10,
    'MAX_DAILY_BUDGET': 1000.0,
    'MIN_ROAS_THRESHOLD': 2.0,
    'SAMPLE_DATA_SIZE': 10000,
    'CUSTOMER_SEGMENT_COUNT': 5,
    'LOG_LEVEL': 'INFO'
}

# Environment-specific configurations
ENVIRONMENT_CONFIGS = {
    'development': {
        'USE_MOCK_APIS': True,
        'DEBUG': True,
        'SAMPLE_DATA_SIZE': 1000,
        'LOG_LEVEL': 'DEBUG'
    },
    'staging': {
        'USE_MOCK_APIS': True,
        'DEBUG': False,
        'SAMPLE_DATA_SIZE': 5000,
        'LOG_LEVEL': 'INFO'
    },
    'production': {
        'USE_MOCK_APIS': False,
        'DEBUG': False,
        'SAMPLE_DATA_SIZE': 10000,
        'LOG_LEVEL': 'INFO'
    }
}

# Feature flags
FEATURE_FLAGS = {
    'ENABLE_AB_TESTING': True,
    'ENABLE_ADVANCED_SEGMENTATION': True,
    'ENABLE_PREDICTIVE_ANALYTICS': False,
    'ENABLE_REAL_TIME_OPTIMIZATION': True,
    'ENABLE_CUSTOM_MODELS': False,
    'ENABLE_MULTI_TENANT': False
}


def get_feature_flag(flag_name: str, default: bool = False) -> bool:
    """
    Get feature flag value with fallback to default
    """
    return FEATURE_FLAGS.get(flag_name, default)


def set_feature_flag(flag_name: str, value: bool):
    """
    Set feature flag value (for development/testing)
    """
    FEATURE_FLAGS[flag_name] = value


# Validation schemas for configuration
CONFIG_SCHEMA = {
    'required_fields': [
        'AGENT_MODEL',
        'MAX_DAILY_BUDGET',
        'MIN_ROAS_THRESHOLD',
        'AVAILABLE_CHANNELS'
    ],
    'numeric_fields': {
        'TEMPERATURE': (0, 1),
        'MAX_ITERATIONS': (1, 100),
        'MAX_DAILY_BUDGET': (0, float('inf')),
        'MIN_ROAS_THRESHOLD': (0, float('inf')),
        'SAMPLE_DATA_SIZE': (100, 100000),
        'CUSTOMER_SEGMENT_COUNT': (2, 20)
    },
    'choice_fields': {
        'AGENT_MODEL': ['gpt-4', 'gpt-3.5-turbo', 'claude-3-opus', 'claude-3-sonnet'],
        'LOG_LEVEL': ['DEBUG', 'INFO', 'WARNING', 'ERROR']
    }
}


# Auto-initialize if imported
try:
    # Only initialize if not already done
    if not hasattr(settings, '_initialized'):
        initialize_config()
        settings._initialized = True
except Exception as e:
    print(f"Warning: Failed to auto-initialize config: {e}")
    print("Please run initialize_config() manually")


# Export version for easy access
version = __version__