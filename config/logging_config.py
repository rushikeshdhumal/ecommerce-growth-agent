"""
Logging configuration for the E-commerce Growth Agent
"""
import logging
import structlog
import sys
from pathlib import Path
from config.settings import settings


def setup_logging():
    """Configure structured logging for the application"""
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(settings.LOG_FILE),
            logging.StreamHandler(sys.stdout)
        ]
    )


class AgentLogger:
    """Specialized logger for agent operations"""
    
    def __init__(self, name: str):
        self.logger = structlog.get_logger(name)
    
    def log_decision(self, phase: str, decision: str, reasoning: str, data: dict = None):
        """Log agent decision with structured data"""
        self.logger.info(
            "agent_decision",
            phase=phase,
            decision=decision, 
            reasoning=reasoning,
            data=data or {}
        )
    
    def log_action(self, action_type: str, parameters: dict, result: dict = None):
        """Log agent action execution"""
        self.logger.info(
            "agent_action",
            action_type=action_type,
            parameters=parameters,
            result=result or {}
        )
    
    def log_observation(self, metric_type: str, values: dict, insights: str = None):
        """Log agent observations and metrics"""
        self.logger.info(
            "agent_observation",
            metric_type=metric_type,
            values=values,
            insights=insights
        )
    
    def log_error(self, error_type: str, error_message: str, context: dict = None):
        """Log errors with context"""
        self.logger.error(
            "agent_error",
            error_type=error_type,
            error_message=error_message,
            context=context or {}
        )
    
    def log_performance(self, campaign_id: str, metrics: dict, benchmark: dict = None):
        """Log campaign performance metrics"""
        self.logger.info(
            "campaign_performance",
            campaign_id=campaign_id,
            metrics=metrics,
            benchmark=benchmark or {}
        )
    
    def log_optimization(self, optimization_type: str, before: dict, after: dict, improvement: float):
        """Log optimization actions and results"""
        self.logger.info(
            "optimization_action",
            optimization_type=optimization_type,
            before_state=before,
            after_state=after,
            improvement_percent=improvement
        )


def get_agent_logger(name: str) -> AgentLogger:
    """Get a configured agent logger instance"""
    return AgentLogger(name)