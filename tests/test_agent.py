"""
Unit tests for the E-commerce Growth Agent core functionality
"""
import pytest
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.agent import EcommerceGrowthAgent, AgentPhase, AgentState, AgentDecision
from src.data_pipeline import DataPipeline
from src.campaign_manager import CampaignManager
from src.evaluation import EvaluationSystem


class TestEcommerceGrowthAgent:
    """Test suite for the main agent class"""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing"""
        with patch('src.agent.DataPipeline') as mock_dp, \
             patch('src.agent.CampaignManager') as mock_cm, \
             patch('src.agent.EvaluationSystem') as mock_es, \
             patch('src.agent.openai') as mock_openai, \
             patch('src.agent.anthropic') as mock_anthropic:
            
            # Mock the dependencies
            mock_dp.return_value = Mock()
            mock_cm.return_value = Mock()
            mock_es.return_value = Mock()
            
            agent = EcommerceGrowthAgent()
            return agent
    
    def test_agent_initialization(self, mock_agent):
        """Test agent initializes correctly"""
        assert mock_agent is not None
        assert mock_agent.state.phase == AgentPhase.IDLE
        assert mock_agent.state.iteration == 0
        assert isinstance(mock_agent.state.active_campaigns, list)
        assert isinstance(mock_agent.state.performance_metrics, dict)
        assert isinstance(mock_agent.state.reasoning_chain, list)
    
    def test_agent_state_transitions(self, mock_agent):
        """Test agent state transitions through phases"""
        # Start in IDLE
        assert mock_agent.state.phase == AgentPhase.IDLE
        
        # Simulate phase transitions
        mock_agent.state.phase = AgentPhase.PLANNING
        assert mock_agent.state.phase == AgentPhase.PLANNING
        
        mock_agent.state.phase = AgentPhase.ACTING
        assert mock_agent.state.phase == AgentPhase.ACTING
        
        mock_agent.state.phase = AgentPhase.OBSERVING
        assert mock_agent.state.phase == AgentPhase.OBSERVING
    
    @patch('src.agent.EcommerceGrowthAgent._call_llm')
    def test_planning_phase(self, mock_llm, mock_agent):
        """Test the planning phase functionality"""
        # Mock LLM response
        mock_planning_response = {
            "objectives": ["optimize_campaigns", "improve_targeting"],
            "recommended_actions": [
                {
                    "type": "optimize_budget",
                    "priority": "high",
                    "parameters": {"campaign_id": "test_campaign"}
                }
            ],
            "summary": "Focus on budget optimization",
            "reasoning": "Current campaigns underperforming"
        }
        mock_llm.return_value = json.dumps(mock_planning_response)
        
        # Mock data pipeline responses
        mock_agent.data_pipeline.get_customer_segments.return_value = {
            "segments": {"seg_1": {"name": "High Value", "size": 1000}}
        }
        mock_agent.data_pipeline.get_performance_metrics.return_value = {
            "overall_roas": 2.5
        }
        mock_agent.data_pipeline.identify_opportunities.return_value = []
        
        # Execute planning phase
        result = mock_agent._planning_phase()
        
        # Assertions
        assert "objectives" in result
        assert "recommended_actions" in result
        assert len(result["recommended_actions"]) > 0
        assert result["recommended_actions"][0]["type"] == "optimize_budget"
        
        # Verify LLM was called
        mock_llm.assert_called_once()
    
    def test_acting_phase(self, mock_agent):
        """Test the acting phase functionality"""
        # Mock planning results
        planning_results = {
            "recommended_actions": [
                {
                    "type": "create_campaign",
                    "parameters": {
                        "campaign_type": "acquisition",
                        "budget": 1000,
                        "channels": ["google_ads"]
                    }
                }
            ]
        }
        
        # Mock campaign manager response
        mock_agent.campaign_manager.create_campaign.return_value = {
            "success": True,
            "campaign_id": "CAMP_TEST_001"
        }
        
        # Execute acting phase
        result = mock_agent._acting_phase(planning_results)
        
        # Assertions
        assert "actions_taken" in result
        assert "optimizations" in result
        assert len(result["actions_taken"]) > 0
        
        # Verify campaign manager was called
        mock_agent.campaign_manager.create_campaign.assert_called_once()
    
    def test_observation_phase(self, mock_agent):
        """Test the observation phase functionality"""
        # Mock action results
        action_results = {
            "actions_taken": [
                {
                    "action": {"type": "create_campaign"},
                    "result": {"success": True, "campaign_id": "CAMP_001"},
                    "status": "success"
                }
            ]
        }
        
        # Mock evaluation system responses
        mock_agent.evaluation_system.calculate_current_metrics.return_value = {
            "overall_roas": 3.2,
            "overall_ctr": 0.025,
            "total_revenue": 50000
        }
        mock_agent.evaluation_system.analyze_trends.return_value = {
            "overall_roas": {"direction": "increasing", "strength": 0.8}
        }
        mock_agent.evaluation_system.detect_anomalies.return_value = []
        
        # Execute observation phase
        result = mock_agent._observation_phase(action_results)
        
        # Assertions
        assert "current_metrics" in result
        assert "performance_trends" in result
        assert "insights" in result
        assert "anomalies" in result
        
        # Verify evaluation system methods were called
        mock_agent.evaluation_system.calculate_current_metrics.assert_called_once()
        mock_agent.evaluation_system.analyze_trends.assert_called_once()
    
    def test_run_iteration_complete_cycle(self, mock_agent):
        """Test a complete agent iteration cycle"""
        # Mock all dependencies for a complete cycle
        mock_agent.data_pipeline.get_customer_segments.return_value = {"segments": {}}
        mock_agent.data_pipeline.get_performance_metrics.return_value = {"overall_roas": 2.5}
        mock_agent.data_pipeline.identify_opportunities.return_value = []
        
        mock_agent.campaign_manager.create_campaign.return_value = {
            "success": True, "campaign_id": "CAMP_001"
        }
        
        mock_agent.evaluation_system.calculate_current_metrics.return_value = {
            "overall_roas": 3.0
        }
        mock_agent.evaluation_system.analyze_trends.return_value = {}
        mock_agent.evaluation_system.detect_anomalies.return_value = []
        
        # Mock LLM response
        with patch.object(mock_agent, '_call_llm') as mock_llm:
            mock_llm.return_value = json.dumps({
                "objectives": ["test_objective"],
                "recommended_actions": [],
                "summary": "Test summary"
            })
            
            # Execute complete iteration
            result = mock_agent.run_iteration()
            
            # Assertions
            assert "iteration" in result
            assert "duration" in result
            assert "planning" in result
            assert "actions" in result
            assert "observations" in result
            assert "state" in result
            
            # Verify iteration number incremented
            assert mock_agent.state.iteration == 1
            assert mock_agent.state.phase == AgentPhase.IDLE
    
    def test_agent_status_retrieval(self, mock_agent):
        """Test agent status information retrieval"""
        # Set some test state
        mock_agent.state.iteration = 5
        mock_agent.state.active_campaigns = ["CAMP_001", "CAMP_002"]
        mock_agent.state.performance_metrics = {"roas": 3.5}
        
        # Get status
        status = mock_agent.get_agent_status()
        
        # Assertions
        assert "state" in status
        assert "performance_summary" in status
        assert "active_campaigns_count" in status
        assert "total_iterations" in status
        
        assert status["total_iterations"] == 5
        assert status["active_campaigns_count"] == 2
    
    def test_agent_reset(self, mock_agent):
        """Test agent reset functionality"""
        # Set some state
        mock_agent.state.iteration = 10
        mock_agent.state.active_campaigns = ["CAMP_001"]
        mock_agent.state.reasoning_chain = ["step1", "step2"]
        
        # Reset agent
        mock_agent.reset_agent()
        
        # Verify reset
        assert mock_agent.state.iteration == 0
        assert len(mock_agent.state.active_campaigns) == 0
        assert len(mock_agent.state.reasoning_chain) == 0
        assert mock_agent.state.phase == AgentPhase.IDLE
    
    def test_decision_validation(self, mock_agent):
        """Test agent decision validation"""
        # Valid decision
        valid_decision = AgentDecision(
            decision_type="create_campaign",
            confidence=0.85,
            reasoning="High ROAS opportunity identified",
            parameters={"budget": 1000, "channel": "google_ads"},
            expected_outcome="20% improvement in ROAS",
            risk_assessment="Low risk - tested strategy"
        )
        
        # Assertions for valid decision
        assert valid_decision.decision_type == "create_campaign"
        assert 0 <= valid_decision.confidence <= 1
        assert isinstance(valid_decision.parameters, dict)
    
    @patch('src.agent.EcommerceGrowthAgent._call_llm')
    def test_llm_fallback_handling(self, mock_llm, mock_agent):
        """Test LLM fallback when AI response fails"""
        # Mock LLM to raise an exception
        mock_llm.side_effect = Exception("API Error")
        
        # Mock data pipeline responses
        mock_agent.data_pipeline.get_customer_segments.return_value = {"segments": {}}
        mock_agent.data_pipeline.get_performance_metrics.return_value = {"overall_roas": 2.5}
        mock_agent.data_pipeline.identify_opportunities.return_value = []
        
        # Execute planning phase (should handle LLM failure gracefully)
        result = mock_agent._planning_phase()
        
        # Should have fallback results
        assert "recommended_actions" in result
        assert "objectives" in result
        assert result.get("reasoning") == "AI analysis unavailable, using rule-based planning"
    
    def test_performance_metrics_tracking(self, mock_agent):
        """Test performance metrics tracking over time"""
        # Add some performance history
        test_metrics = {
            "timestamp": datetime.now().isoformat(),
            "metrics": {"roas": 3.0, "ctr": 0.025},
            "actions": {"count": 2},
            "insights": ["test insight"]
        }
        
        mock_agent.performance_history.append(test_metrics)
        
        # Verify tracking
        assert len(mock_agent.performance_history) == 1
        assert mock_agent.performance_history[0]["metrics"]["roas"] == 3.0
    
    def test_safety_guardrails(self, mock_agent):
        """Test safety guardrails and constraints"""
        # Test budget constraint validation in planning results
        planning_results = {
            "recommended_actions": [
                {
                    "type": "create_campaign",
                    "parameters": {
                        "budget": 999999  # Exceeds max budget
                    }
                }
            ]
        }
        
        # Validate planning results (should apply constraints)
        validated_results = mock_agent._validate_planning_results(planning_results)
        
        # Check that budget was capped
        action = validated_results["recommended_actions"][0]
        assert action["parameters"]["budget"] <= 1000.0  # MAX_DAILY_BUDGET from settings


class TestAgentDecision:
    """Test suite for AgentDecision dataclass"""
    
    def test_decision_creation(self):
        """Test decision object creation"""
        decision = AgentDecision(
            decision_type="optimize_budget",
            confidence=0.92,
            reasoning="Performance metrics indicate budget increase would improve ROAS",
            parameters={"campaign_id": "CAMP_001", "new_budget": 1500},
            expected_outcome="15% improvement in ROAS",
            risk_assessment="Low risk - historical data supports decision"
        )
        
        assert decision.decision_type == "optimize_budget"
        assert decision.confidence == 0.92
        assert "CAMP_001" in str(decision.parameters)
    
    def test_decision_serialization(self):
        """Test decision serialization for logging"""
        decision = AgentDecision(
            decision_type="pause_campaign",
            confidence=0.75,
            reasoning="Campaign performance below threshold",
            parameters={"campaign_id": "CAMP_002"},
            expected_outcome="Cost savings of $200/day",
            risk_assessment="Medium risk - may impact overall reach"
        )
        
        # Convert to dict for serialization
        decision_dict = {
            "decision_type": decision.decision_type,
            "confidence": decision.confidence,
            "reasoning": decision.reasoning,
            "parameters": decision.parameters,
            "expected_outcome": decision.expected_outcome,
            "risk_assessment": decision.risk_assessment
        }
        
        # Verify serialization
        assert isinstance(decision_dict, dict)
        assert decision_dict["decision_type"] == "pause_campaign"
        assert decision_dict["confidence"] == 0.75


class TestAgentIntegration:
    """Integration tests for agent with its components"""
    
    @pytest.fixture
    def integrated_agent(self):
        """Create agent with real component integration (but mocked external APIs)"""
        with patch('openai.ChatCompletion.create') as mock_openai, \
             patch('anthropic.Anthropic') as mock_anthropic:
            
            # Mock OpenAI response
            mock_openai.return_value = Mock()
            mock_openai.return_value.choices = [Mock()]
            mock_openai.return_value.choices[0].message.content = json.dumps({
                "objectives": ["test_objective"],
                "recommended_actions": [],
                "summary": "Test plan"
            })
            
            agent = EcommerceGrowthAgent()
            return agent
    
    def test_end_to_end_workflow(self, integrated_agent):
        """Test end-to-end agent workflow with real components"""
        # This would test the full workflow but with mocked external dependencies
        # Due to the complexity of setting up all real components, this is a placeholder
        # for more comprehensive integration testing
        
        assert integrated_agent is not None
        assert hasattr(integrated_agent, 'data_pipeline')
        assert hasattr(integrated_agent, 'campaign_manager')
        assert hasattr(integrated_agent, 'evaluation_system')


if __name__ == "__main__":
    pytest.main([__file__])