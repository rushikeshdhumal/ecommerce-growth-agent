"""
Configuration settings for the E-commerce Growth Agent
"""
import os
from typing import Dict, Any
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # API Configuration
    OPENAI_API_KEY: str = Field(default="", env="OPENAI_API_KEY")
    ANTHROPIC_API_KEY: str = Field(default="", env="ANTHROPIC_API_KEY")
    
    # Agent Configuration
    AGENT_MODEL: str = Field(default="gpt-4", env="AGENT_MODEL")
    MAX_ITERATIONS: int = Field(default=10, env="MAX_ITERATIONS")
    TEMPERATURE: float = Field(default=0.7, env="TEMPERATURE")
    
    # Campaign Configuration
    MIN_CAMPAIGN_BUDGET: float = Field(default=100.0, env="MIN_CAMPAIGN_BUDGET")
    MAX_CAMPAIGN_BUDGET: float = Field(default=10000.0, env="MAX_CAMPAIGN_BUDGET")
    MIN_ROAS_THRESHOLD: float = Field(default=2.0, env="MIN_ROAS_THRESHOLD")
    
    # Data Configuration
    CUSTOMER_SEGMENT_COUNT: int = Field(default=5, env="CUSTOMER_SEGMENT_COUNT")
    SAMPLE_DATA_SIZE: int = Field(default=10000, env="SAMPLE_DATA_SIZE")
    
    # Database Configuration
    DATABASE_URL: str = Field(default="sqlite:///ecommerce_agent.db", env="DATABASE_URL")
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FILE: str = Field(default="logs/agent.log", env="LOG_FILE")
    
    # Performance Thresholds
    PERFORMANCE_THRESHOLDS: Dict[str, float] = {
        "min_ctr": 0.01,
        "min_conversion_rate": 0.02,
        "max_cac": 500.0,
        "min_customer_ltv": 100.0,
        "max_churn_risk": 0.8
    }
    
    # Campaign Channels
    AVAILABLE_CHANNELS: list = [
        "google_ads",
        "meta_ads", 
        "email",
        "sms",
        "display"
    ]
    
    # Safety Guardrails
    MAX_DAILY_BUDGET: float = Field(default=1000.0, env="MAX_DAILY_BUDGET")
    MIN_AUDIENCE_SIZE: int = Field(default=1000, env="MIN_AUDIENCE_SIZE")
    MAX_FREQUENCY_CAP: int = Field(default=3, env="MAX_FREQUENCY_CAP")
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


# Campaign Templates
CAMPAIGN_TEMPLATES = {
    "acquisition": {
        "objective": "customer_acquisition",
        "target_audience": "cold_prospects",
        "budget_allocation": {"google_ads": 0.4, "meta_ads": 0.4, "email": 0.2},
        "optimization_goal": "conversions"
    },
    "retention": {
        "objective": "customer_retention", 
        "target_audience": "existing_customers",
        "budget_allocation": {"email": 0.6, "sms": 0.2, "display": 0.2},
        "optimization_goal": "engagement"
    },
    "winback": {
        "objective": "winback_churned",
        "target_audience": "churned_customers", 
        "budget_allocation": {"email": 0.5, "meta_ads": 0.3, "google_ads": 0.2},
        "optimization_goal": "conversions"
    },
    "upsell": {
        "objective": "increase_order_value",
        "target_audience": "high_value_customers",
        "budget_allocation": {"email": 0.7, "display": 0.3},
        "optimization_goal": "revenue"
    }
}


# Model Configurations
MODEL_CONFIGS = {
    "gpt-4": {
        "max_tokens": 4096,
        "temperature": 0.7,
        "provider": "openai"
    },
    "gpt-3.5-turbo": {
        "max_tokens": 4096, 
        "temperature": 0.7,
        "provider": "openai"
    },
    "claude-3-opus": {
        "max_tokens": 4096,
        "temperature": 0.7,
        "provider": "anthropic"
    }
}