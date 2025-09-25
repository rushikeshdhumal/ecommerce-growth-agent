"""
Core Agent implementation for E-commerce Growth Agent
Implements the plan→act→observe loop for autonomous marketing campaign management
"""
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

import openai
import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from config.settings import settings, CAMPAIGN_TEMPLATES
from config.logging_config import get_agent_logger
from src.data_pipeline import DataPipeline
from src.campaign_manager import CampaignManager
from src.evaluation import EvaluationSystem


class AgentPhase(Enum):
    """Agent execution phases"""
    PLANNING = "planning"
    ACTING = "acting" 
    OBSERVING = "observing"
    IDLE = "idle"


@dataclass
class AgentState:
    """Current state of the agent"""
    phase: AgentPhase
    iteration: int
    last_action_time: datetime
    active_campaigns: List[str]
    performance_metrics: Dict[str, float]
    reasoning_chain: List[str]
    current_objectives: List[str]


@dataclass
class AgentDecision:
    """Structure for agent decisions"""
    decision_type: str
    confidence: float
    reasoning: str
    parameters: Dict[str, Any]
    expected_outcome: str
    risk_assessment: str


class EcommerceGrowthAgent:
    """
    Autonomous E-commerce Growth Agent that follows plan→act→observe loop
    to optimize marketing campaigns across multiple channels
    """
    
    def __init__(self):
        self.logger = get_agent_logger("EcommerceGrowthAgent")
        self.data_pipeline = DataPipeline()
        self.campaign_manager = CampaignManager()
        self.evaluation_system = EvaluationSystem()
        
        # Initialize AI clients
        self._setup_ai_clients()
        
        # Agent state
        self.state = AgentState(
            phase=AgentPhase.IDLE,
            iteration=0,
            last_action_time=datetime.now(),
            active_campaigns=[],
            performance_metrics={},
            reasoning_chain=[],
            current_objectives=[]
        )
        
        # Performance tracking
        self.performance_history = []
        self.decision_history = []
        
        self.logger.log_action("agent_initialized", {"model": settings.AGENT_MODEL})
    
    def _setup_ai_clients(self):
        """Initialize AI API clients"""
        if settings.OPENAI_API_KEY:
            openai.api_key = settings.OPENAI_API_KEY
            self.openai_client = openai
        
        if settings.ANTHROPIC_API_KEY:
            self.anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def _call_llm(self, prompt: str, system_message: str = None) -> str:
        """Call LLM with retry logic"""
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
                    max_tokens=4096
                )
                return response.choices[0].message.content
            
            elif settings.AGENT_MODEL.startswith("claude"):
                full_prompt = f"{system_message}\n\nHuman: {prompt}\n\nAssistant:" if system_message else prompt
                response = self.anthropic_client.messages.create(
                    model=settings.AGENT_MODEL,
                    max_tokens=4096,
                    temperature=settings.TEMPERATURE,
                    messages=[{"role": "user", "content": full_prompt}]
                )
                return response.content[0].text
            
        except Exception as e:
            self.logger.log_error("llm_call_failed", str(e), {"model": settings.AGENT_MODEL})
            raise
    
    def run_iteration(self) -> Dict[str, Any]:
        """Execute one complete plan→act→observe iteration"""
        self.state.iteration += 1
        iteration_start = time.time()
        
        self.logger.log_action("iteration_started", {"iteration": self.state.iteration})
        
        try:
            # Plan Phase
            self.state.phase = AgentPhase.PLANNING
            planning_results = self._planning_phase()
            
            # Act Phase
            self.state.phase = AgentPhase.ACTING
            action_results = self._acting_phase(planning_results)
            
            # Observe Phase  
            self.state.phase = AgentPhase.OBSERVING
            observation_results = self._observation_phase(action_results)
            
            # Update state
            self.state.phase = AgentPhase.IDLE
            self.state.last_action_time = datetime.now()
            
            iteration_time = time.time() - iteration_start
            
            iteration_summary = {
                "iteration": self.state.iteration,
                "duration": iteration_time,
                "planning": planning_results,
                "actions": action_results,
                "observations": observation_results,
                "state": asdict(self.state)
            }
            
            self.logger.log_action("iteration_completed", iteration_summary)
            return iteration_summary
            
        except Exception as e:
            self.logger.log_error("iteration_failed", str(e), {"iteration": self.state.iteration})
            self.state.phase = AgentPhase.IDLE
            raise
    
    def _planning_phase(self) -> Dict[str, Any]:
        """Plan phase: Analyze data and identify opportunities"""
        self.logger.log_decision("planning", "start_analysis", "Beginning data analysis and opportunity identification")
        
        # Get current data insights
        customer_segments = self.data_pipeline.get_customer_segments()
        performance_data = self.data_pipeline.get_performance_metrics()
        market_opportunities = self.data_pipeline.identify_opportunities()
        
        # Generate reasoning prompt
        planning_prompt = self._create_planning_prompt(customer_segments, performance_data, market_opportunities)
        
        # Get AI analysis
        system_message = """You are an expert e-commerce growth strategist. Analyze the provided data and create a strategic plan for marketing campaign optimization. Focus on:
1. Identifying the highest-impact opportunities
2. Prioritizing actions based on potential ROI
3. Assessing risks and constraints
4. Recommending specific campaigns and channels

Respond with a structured JSON analysis including opportunities, priorities, and recommended actions."""
        
        ai_analysis = self._call_llm(planning_prompt, system_message)
        
        try:
            planning_results = json.loads(ai_analysis)
        except json.JSONDecodeError:
            # Fallback to structured planning
            planning_results = self._fallback_planning(customer_segments, performance_data, market_opportunities)
        
        # Validate and enhance planning results
        planning_results = self._validate_planning_results(planning_results)
        
        # Update agent state
        self.state.current_objectives = planning_results.get("objectives", [])
        self.state.reasoning_chain.append(f"Planning: {planning_results.get('summary', 'Completed analysis')}")
        
        self.logger.log_decision(
            "planning", 
            "plan_created", 
            planning_results.get("reasoning", "Plan created successfully"),
            planning_results
        )
        
        return planning_results
    
    def _acting_phase(self, planning_results: Dict[str, Any]) -> Dict[str, Any]:
        """Act phase: Execute planned campaigns and optimizations"""
        self.logger.log_decision("acting", "execute_plan", "Executing planned marketing actions")
        
        actions_taken = []
        
        # Execute recommended actions from planning phase
        for action in planning_results.get("recommended_actions", []):
            try:
                action_result = self._execute_action(action)
                actions_taken.append(action_result)
                
                # Add delay between actions to respect rate limits
                time.sleep(1)
                
            except Exception as e:
                self.logger.log_error("action_execution_failed", str(e), action)
                actions_taken.append({
                    "action": action,
                    "status": "failed",
                    "error": str(e)
                })
        
        # Optimize existing campaigns
        optimization_results = self._optimize_active_campaigns()
        
        acting_results = {
            "actions_taken": actions_taken,
            "optimizations": optimization_results,
            "total_actions": len(actions_taken)
        }
        
        self.state.reasoning_chain.append(f"Acting: Executed {len(actions_taken)} actions")
        
        return acting_results
    
    def _observation_phase(self, action_results: Dict[str, Any]) -> Dict[str, Any]:
        """Observe phase: Monitor performance and learn from results"""
        self.logger.log_decision("observing", "monitor_performance", "Analyzing campaign performance and outcomes")
        
        # Collect current performance metrics
        current_metrics = self.evaluation_system.calculate_current_metrics()
        
        # Compare with historical performance
        performance_trends = self.evaluation_system.analyze_trends()
        
        # Identify insights and learnings
        insights = self._generate_insights(current_metrics, performance_trends, action_results)
        
        # Update performance tracking
        self.performance_history.append({
            "timestamp": datetime.now().isoformat(),
            "metrics": current_metrics,
            "actions": action_results,
            "insights": insights
        })
        
        # Safety checks and anomaly detection
        anomalies = self.evaluation_system.detect_anomalies(current_metrics)
        
        observation_results = {
            "current_metrics": current_metrics,
            "performance_trends": performance_trends,
            "insights": insights,
            "anomalies": anomalies,
            "learning_summary": self._create_learning_summary(insights)
        }
        
        # Update agent state
        self.state.performance_metrics = current_metrics
        self.state.reasoning_chain.append(f"Observing: {len(insights)} insights generated")
        
        self.logger.log_observation("performance_analysis", current_metrics, str(insights))
        
        return observation_results
    
    def _create_planning_prompt(self, segments: Dict, performance: Dict, opportunities: List) -> str:
        """Create detailed prompt for planning phase"""
        return f"""
        Analyze the current e-commerce performance data and create an optimal marketing strategy:

        CUSTOMER SEGMENTS:
        {json.dumps(segments, indent=2)}

        CURRENT PERFORMANCE METRICS:
        {json.dumps(performance, indent=2)}

        IDENTIFIED OPPORTUNITIES:
        {json.dumps(opportunities, indent=2)}

        CURRENT CAMPAIGN PORTFOLIO:
        Active Campaigns: {len(self.state.active_campaigns)}
        Recent Performance: {self.state.performance_metrics}

        CONSTRAINTS:
        - Maximum daily budget: ${settings.MAX_DAILY_BUDGET}
        - Minimum ROAS threshold: {settings.MIN_ROAS_THRESHOLD}
        - Available channels: {settings.AVAILABLE_CHANNELS}

        Please provide a strategic analysis including:
        1. Top 3 highest-impact opportunities with ROI estimates
        2. Recommended campaign actions with specific parameters
        3. Budget allocation recommendations
        4. Risk assessment for each recommendation
        5. Success metrics to track

        Format as JSON with clear structure.
        """
    
    def _execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific marketing action"""
        action_type = action.get("type")
        
        if action_type == "create_campaign":
            return self._create_campaign(action)
        elif action_type == "optimize_budget":
            return self._optimize_budget(action)
        elif action_type == "update_targeting":
            return self._update_targeting(action)
        elif action_type == "pause_campaign":
            return self._pause_campaign(action)
        elif action_type == "create_creative":
            return self._create_creative(action)
        else:
            raise ValueError(f"Unknown action type: {action_type}")
    
    def _create_campaign(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new marketing campaign"""
        campaign_params = action.get("parameters", {})
        
        # Generate campaign using campaign manager
        campaign_result = self.campaign_manager.create_campaign(
            campaign_type=campaign_params.get("campaign_type", "acquisition"),
            target_segment=campaign_params.get("target_segment"),
            budget=campaign_params.get("budget", 1000),
            channels=campaign_params.get("channels", ["google_ads"])
        )
        
        if campaign_result.get("success"):
            campaign_id = campaign_result.get("campaign_id")
            self.state.active_campaigns.append(campaign_id)
            
            self.logger.log_action("campaign_created", {
                "campaign_id": campaign_id,
                "parameters": campaign_params
            })
        
        return {
            "action": action,
            "result": campaign_result,
            "status": "success" if campaign_result.get("success") else "failed"
        }
    
    def _optimize_active_campaigns(self) -> List[Dict[str, Any]]:
        """Optimize all currently active campaigns"""
        optimizations = []
        
        for campaign_id in self.state.active_campaigns:
            try:
                # Get campaign performance
                performance = self.campaign_manager.get_campaign_performance(campaign_id)
                
                # Determine optimization actions
                optimization_actions = self._determine_optimizations(campaign_id, performance)
                
                # Execute optimizations
                for opt_action in optimization_actions:
                    result = self.campaign_manager.optimize_campaign(campaign_id, opt_action)
                    optimizations.append({
                        "campaign_id": campaign_id,
                        "action": opt_action,
                        "result": result
                    })
                    
            except Exception as e:
                self.logger.log_error("optimization_failed", str(e), {"campaign_id": campaign_id})
        
        return optimizations
    
    def _determine_optimizations(self, campaign_id: str, performance: Dict) -> List[Dict]:
        """Determine what optimizations to apply to a campaign"""
        optimizations = []
        
        # Budget optimization
        if performance.get("roas", 0) > settings.MIN_ROAS_THRESHOLD * 1.5:
            optimizations.append({
                "type": "increase_budget",
                "factor": 1.2,
                "reasoning": "High ROAS indicates opportunity to scale"
            })
        elif performance.get("roas", 0) < settings.MIN_ROAS_THRESHOLD:
            optimizations.append({
                "type": "decrease_budget", 
                "factor": 0.8,
                "reasoning": "Low ROAS requires budget reduction"
            })
        
        # Creative optimization
        if performance.get("ctr", 0) < settings.PERFORMANCE_THRESHOLDS["min_ctr"]:
            optimizations.append({
                "type": "refresh_creative",
                "reasoning": "Low CTR indicates creative fatigue"
            })
        
        # Audience optimization
        if performance.get("conversion_rate", 0) < settings.PERFORMANCE_THRESHOLDS["min_conversion_rate"]:
            optimizations.append({
                "type": "refine_targeting",
                "reasoning": "Low conversion rate suggests targeting issues"
            })
        
        return optimizations
    
    def _generate_insights(self, metrics: Dict, trends: Dict, actions: Dict) -> List[str]:
        """Generate insights from performance data and recent actions"""
        insights = []
        
        # Performance insights
        if metrics.get("overall_roas", 0) > 3.0:
            insights.append("Strong overall ROAS indicates effective campaign portfolio")
        
        if trends.get("cac_trend", 0) < -0.1:
            insights.append("Customer acquisition cost is improving over time")
        
        # Action effectiveness insights
        successful_actions = [a for a in actions.get("actions_taken", []) if a.get("status") == "success"]
        if len(successful_actions) > 0:
            insights.append(f"Successfully executed {len(successful_actions)} optimization actions")
        
        # Channel performance insights
        best_channel = max(metrics.get("channel_performance", {}), key=lambda x: metrics["channel_performance"][x].get("roas", 0), default=None)
        if best_channel:
            insights.append(f"{best_channel} is the top performing channel with highest ROAS")
        
        return insights
    
    def _create_learning_summary(self, insights: List[str]) -> str:
        """Create a summary of learnings from the current iteration"""
        if not insights:
            return "No significant insights generated this iteration"
        
        return f"Key learnings: {'; '.join(insights[:3])}"
    
    def _fallback_planning(self, segments: Dict, performance: Dict, opportunities: List) -> Dict:
        """Fallback planning logic when AI analysis fails"""
        return {
            "objectives": ["optimize_existing_campaigns", "identify_new_segments"],
            "recommended_actions": [
                {
                    "type": "optimize_budget",
                    "priority": "high",
                    "parameters": {"optimization_type": "performance_based"}
                }
            ],
            "summary": "Fallback plan: Focus on optimization of existing campaigns",
            "reasoning": "AI analysis unavailable, using rule-based planning"
        }
    
    def _validate_planning_results(self, results: Dict) -> Dict:
        """Validate and sanitize planning results"""
        # Ensure required fields exist
        if "recommended_actions" not in results:
            results["recommended_actions"] = []
        
        if "objectives" not in results:
            results["objectives"] = ["maintain_performance"]
        
        # Validate budget constraints
        for action in results["recommended_actions"]:
            if action.get("type") == "create_campaign":
                budget = action.get("parameters", {}).get("budget", 0)
                if budget > settings.MAX_DAILY_BUDGET:
                    action["parameters"]["budget"] = settings.MAX_DAILY_BUDGET
        
        return results
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status and state"""
        return {
            "state": asdict(self.state),
            "performance_summary": self._get_performance_summary(),
            "active_campaigns_count": len(self.state.active_campaigns),
            "last_iteration_time": self.state.last_action_time.isoformat(),
            "total_iterations": self.state.iteration
        }
    
    def _get_performance_summary(self) -> Dict[str, float]:
        """Get summary of recent performance"""
        if not self.performance_history:
            return {}
        
        latest = self.performance_history[-1]
        return latest.get("metrics", {})
    
    def reset_agent(self):
        """Reset agent state for new session"""
        self.state = AgentState(
            phase=AgentPhase.IDLE,
            iteration=0,
            last_action_time=datetime.now(),
            active_campaigns=[],
            performance_metrics={},
            reasoning_chain=[],
            current_objectives=[]
        )
        self.logger.log_action("agent_reset", {})


# Additional utility functions for the agent
def create_mock_response(action_type: str) -> Dict[str, Any]:
    """Create mock responses for testing"""
    return {
        "success": True,
        "action_type": action_type,
        "timestamp": datetime.now().isoformat(),
        "mock": True
    }