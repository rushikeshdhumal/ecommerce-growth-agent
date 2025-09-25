"""
Streamlit Demo Application for E-commerce Growth Agent
Interactive dashboard showcasing autonomous marketing campaign management
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import our modules
from config.settings import settings
from config.logging_config import setup_logging, get_agent_logger
from src.agent import EcommerceGrowthAgent
from src.data_pipeline import DataPipeline
from src.campaign_manager import CampaignManager
from src.evaluation import EvaluationSystem
from src.utils import (
    format_currency, format_percentage, format_large_number,
    calculate_percentage_change, generate_color_palette, MetricsCalculator
)


# Page configuration
st.set_page_config(
    page_title="E-commerce Growth Agent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
setup_logging()
logger = get_agent_logger("StreamlitApp")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-warning {
        color: #ffc107;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .agent-decision {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #17a2b8;
    }
    .small-font {
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_agent():
    """Initialize the E-commerce Growth Agent"""
    try:
        agent = EcommerceGrowthAgent()
        logger.log_action("streamlit_agent_initialized", {})
        return agent
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return None


@st.cache_resource
def initialize_components():
    """Initialize data pipeline and other components"""
    try:
        data_pipeline = DataPipeline()
        evaluation_system = EvaluationSystem()
        return data_pipeline, evaluation_system
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        return None, None


def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🚀 Autonomous E-commerce Growth Agent")
    st.markdown("**AI-Powered Marketing Campaign Orchestration & Optimization**")
    st.markdown("---")
    
    # Initialize components
    agent = initialize_agent()
    data_pipeline, evaluation_system = initialize_components()
    
    if not agent or not data_pipeline or not evaluation_system:
        st.error("Failed to initialize application components. Please check the configuration.")
        return
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Agent status in sidebar
    st.sidebar.markdown("### Agent Status")
    agent_status = agent.get_agent_status()
    
    phase = agent_status['state']['phase']
    if phase == 'idle':
        st.sidebar.markdown('<span class="status-success">🟢 IDLE</span>', unsafe_allow_html=True)
    elif phase == 'planning':
        st.sidebar.markdown('<span class="status-warning">🟡 PLANNING</span>', unsafe_allow_html=True)
    elif phase == 'acting':
        st.sidebar.markdown('<span class="status-warning">🟡 ACTING</span>', unsafe_allow_html=True)
    elif phase == 'observing':
        st.sidebar.markdown('<span class="status-warning">🟡 OBSERVING</span>', unsafe_allow_html=True)
    
    st.sidebar.metric("Total Iterations", agent_status['total_iterations'])
    st.sidebar.metric("Active Campaigns", agent_status['active_campaigns_count'])
    
    # Page selection
    page = st.sidebar.selectbox(
        "Select Page",
        [
            "Dashboard Overview",
            "Agent Control Panel", 
            "Campaign Management",
            "Performance Analytics",
            "Customer Insights",
            "A/B Testing",
            "System Monitoring",
            "Settings"
        ]
    )
    
    # Route to different pages
    if page == "Dashboard Overview":
        show_dashboard(agent, data_pipeline, evaluation_system)
    elif page == "Agent Control Panel":
        show_agent_control(agent)
    elif page == "Campaign Management":
        show_campaign_management(agent)
    elif page == "Performance Analytics":
        show_performance_analytics(data_pipeline, evaluation_system)
    elif page == "Customer Insights":
        show_customer_insights(data_pipeline)
    elif page == "A/B Testing":
        show_ab_testing(evaluation_system)
    elif page == "System Monitoring":
        show_system_monitoring(agent, evaluation_system)
    elif page == "Settings":
        show_settings()


def show_dashboard(agent, data_pipeline, evaluation_system):
    """Show main dashboard overview"""
    st.header("📊 Dashboard Overview")
    
    # Get current metrics
    current_metrics = evaluation_system.calculate_current_metrics()
    agent_status = agent.get_agent_status()
    
    # Key Performance Indicators
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        roas = current_metrics.get('overall_roas', 0)
        st.metric(
            "Overall ROAS",
            f"{roas:.2f}x",
            delta=f"{calculate_percentage_change(roas, 3.0):.1f}%" if roas > 0 else None
        )
    
    with col2:
        total_revenue = current_metrics.get('total_revenue', 0)
        st.metric(
            "Total Revenue",
            format_currency(total_revenue),
            delta=f"+{format_percentage(12.5)}" if total_revenue > 0 else None
        )
    
    with col3:
        active_customers = current_metrics.get('active_customers', 0)
        st.metric(
            "Active Customers",
            format_large_number(active_customers),
            delta=f"+{format_percentage(8.2)}" if active_customers > 0 else None
        )
    
    with col4:
        churn_rate = current_metrics.get('churn_rate', 0)
        st.metric(
            "Churn Rate",
            format_percentage(churn_rate * 100),
            delta=f"-{format_percentage(2.1)}" if churn_rate > 0 else None
        )
    
    # Performance Overview Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Performance Trends")
        trends = evaluation_system.analyze_trends()
        
        if trends:
            # Create trend chart
            trend_data = []
            for metric, trend_info in list(trends.items())[:5]:
                trend_data.append({
                    'Metric': metric.replace('overall_', '').replace('_', ' ').title(),
                    'Current Value': trend_info['current_value'],
                    'Trend': trend_info['direction'],
                    'Change %': trend_info['percentage_change']
                })
            
            if trend_data:
                df_trends = pd.DataFrame(trend_data)
                
                fig = px.bar(
                    df_trends,
                    x='Metric',
                    y='Change %',
                    color='Change %',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    title="Performance Change (%)"
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Channel Performance")
        
        # Channel performance data
        channel_data = []
        channel_performance = current_metrics.get('channel_performance', {})
        
        for channel, metrics in channel_performance.items():
            channel_data.append({
                'Channel': channel.replace('_', ' ').title(),
                'ROAS': metrics.get('roas', 0),
                'Spend': metrics.get('spend', 0),
                'Revenue': metrics.get('revenue', 0)
            })
        
        if channel_data:
            df_channels = pd.DataFrame(channel_data)
            
            fig = px.scatter(
                df_channels,
                x='Spend',
                y='Revenue',
                size='ROAS',
                color='Channel',
                title="Channel Performance (Size = ROAS)",
                hover_data=['ROAS']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent Agent Actions
    st.subheader("🤖 Recent Agent Activity")
    
    reasoning_chain = agent_status['state']['reasoning_chain']
    if reasoning_chain:
        for i, decision in enumerate(reasoning_chain[-5:]):  # Show last 5 decisions
            st.markdown(f"""
            <div class="agent-decision">
                <strong>Decision {len(reasoning_chain) - 4 + i}:</strong> {decision}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No recent agent activity to display.")
    
    # Quick Actions
    st.subheader("⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Run Agent Iteration", type="primary"):
            with st.spinner("Running agent iteration..."):
                try:
                    result = agent.run_iteration()
                    st.success(f"Iteration {result['iteration']} completed successfully!")
                    st.json(result)
                    st.rerun()
                except Exception as e:
                    st.error(f"Agent iteration failed: {str(e)}")
    
    with col2:
        if st.button("📊 Refresh Metrics"):
            st.cache_resource.clear()
            st.success("Metrics refreshed!")
            st.rerun()
    
    with col3:
        if st.button("🎯 Create Campaign"):
            st.switch_page("Campaign Management")
    
    with col4:
        if st.button("🔍 View Analysis"):
            st.switch_page("Performance Analytics")


def show_agent_control(agent):
    """Show agent control panel"""
    st.header("🤖 Agent Control Panel")
    
    # Agent Configuration
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Agent Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "AI Model",
            ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"],
            index=0 if settings.AGENT_MODEL == "gpt-4" else 1
        )
        
        # Temperature setting
        temperature = st.slider(
            "Model Temperature",
            min_value=0.0,
            max_value=1.0,
            value=settings.TEMPERATURE,
            step=0.1
        )
        
        # Max iterations
        max_iterations = st.number_input(
            "Max Iterations per Run",
            min_value=1,
            max_value=20,
            value=settings.MAX_ITERATIONS
        )
        
        # Budget constraints
        max_budget = st.number_input(
            "Max Daily Budget ($)",
            min_value=100,
            max_value=50000,
            value=int(settings.MAX_DAILY_BUDGET)
        )
    
    with col2:
        st.subheader("📊 Agent Status")
        
        agent_status = agent.get_agent_status()
        state = agent_status['state']
        
        # Status display
        st.json({
            "Current Phase": state['phase'],
            "Iteration": state['iteration'],
            "Active Campaigns": len(state['active_campaigns']),
            "Last Action": state['last_action_time'],
            "Current Objectives": state['current_objectives'][:3] if state['current_objectives'] else []
        })
        
        # Performance summary
        if agent_status['performance_summary']:
            st.subheader("📈 Performance Summary")
            perf_summary = agent_status['performance_summary']
            
            metrics_col1, metrics_col2 = st.columns(2)
            
            with metrics_col1:
                st.metric("ROAS", f"{perf_summary.get('overall_roas', 0):.2f}x")
                st.metric("CTR", format_percentage(perf_summary.get('overall_ctr', 0) * 100))
            
            with metrics_col2:
                st.metric("Conversion Rate", format_percentage(perf_summary.get('overall_conversion_rate', 0) * 100))
                st.metric("Total Spend", format_currency(perf_summary.get('total_spend', 0)))
    
    # Agent Controls
    st.subheader("🎮 Agent Controls")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("▶️ Run Single Iteration", type="primary"):
            with st.spinner("Executing agent iteration..."):
                try:
                    start_time = time.time()
                    result = agent.run_iteration()
                    execution_time = time.time() - start_time
                    
                    st.success(f"Iteration completed in {execution_time:.2f} seconds")
                    
                    # Show iteration results
                    with st.expander("View Iteration Details"):
                        st.json(result)
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Iteration failed: {str(e)}")
    
    with col2:
        if st.button("🔄 Reset Agent"):
            if st.confirm("Are you sure you want to reset the agent?"):
                agent.reset_agent()
                st.success("Agent reset successfully!")
                st.rerun()
    
    with col3:
        if st.button("⏸️ Pause Agent"):
            st.info("Agent paused (simulation)")
    
    with col4:
        if st.button("📥 Export Logs"):
            # Simulate log export
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "Download Logs",
                data="Agent logs would be exported here...",
                file_name=f"agent_logs_{timestamp}.txt",
                mime="text/plain"
            )
    
    # Real-time Decision Making
    st.subheader("🧠 Real-time Decision Making")
    
    # Show reasoning chain
    reasoning_chain = agent_status['state']['reasoning_chain']
    
    if reasoning_chain:
        with st.container():
            st.write("**Recent Reasoning Chain:**")
            for i, step in enumerate(reasoning_chain[-10:]):  # Last 10 steps
                step_number = len(reasoning_chain) - 9 + i
                st.markdown(f"""
                <div class="agent-decision">
                    <strong>Step {step_number}:</strong> {step}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No reasoning history available yet.")
    
    # Manual Decision Override
    st.subheader("🎯 Manual Decision Override")
    
    with st.form("manual_decision"):
        st.write("Override agent decision with manual input:")
        
        decision_type = st.selectbox(
            "Decision Type",
            ["Create Campaign", "Optimize Budget", "Pause Campaign", "Update Targeting"]
        )
        
        campaign_id = st.text_input("Campaign ID (if applicable)")
        parameters = st.text_area("Decision Parameters (JSON format)")
        reasoning = st.text_area("Reasoning for this decision")
        
        if st.form_submit_button("Execute Manual Decision"):
            try:
                params = json.loads(parameters) if parameters else {}
                
                # Simulate decision execution
                st.success(f"Manual decision '{decision_type}' executed successfully!")
                st.json({
                    "decision_type": decision_type,
                    "campaign_id": campaign_id,
                    "parameters": params,
                    "reasoning": reasoning,
                    "timestamp": datetime.now().isoformat()
                })
                
            except json.JSONDecodeError:
                st.error("Invalid JSON format in parameters")


def show_campaign_management(agent):
    """Show campaign management interface"""
    st.header("🎯 Campaign Management")
    
    # Campaign Overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Active Campaigns")
        
        # Get campaign data
        try:
            campaigns = agent.campaign_manager.get_all_campaigns()
            
            if campaigns['campaigns']:
                campaign_data = []
                for campaign_id, campaign_info in campaigns['campaigns'].items():
                    performance = campaign_info.get('performance', {})
                    campaign_data.append({
                        'Campaign ID': campaign_id,
                        'Name': campaign_info['name'],
                        'Type': campaign_info['type'].title(),
                        'Budget': format_currency(campaign_info['budget']),
                        'Status': campaign_info['status'].title(),
                        'ROAS': f"{performance.get('roas', 0):.2f}x",
                        'Spend': format_currency(performance.get('cost', 0)),
                        'Revenue': format_currency(performance.get('revenue', 0))
                    })
                
                df_campaigns = pd.DataFrame(campaign_data)
                st.dataframe(df_campaigns, use_container_width=True)
                
            else:
                st.info("No active campaigns found.")
                
        except Exception as e:
            st.error(f"Failed to load campaigns: {str(e)}")
    
    with col2:
        st.subheader("📊 Campaign Stats")
        
        # Campaign statistics
        try:
            campaigns = agent.campaign_manager.get_all_campaigns()
            
            st.metric("Total Campaigns", campaigns['active_campaigns'])
            
            # Calculate total metrics
            total_budget = 0
            total_spend = 0
            total_revenue = 0
            
            for campaign_info in campaigns['campaigns'].values():
                total_budget += campaign_info.get('budget', 0)
                performance = campaign_info.get('performance', {})
                total_spend += performance.get('cost', 0)
                total_revenue += performance.get('revenue', 0)
            
            st.metric("Total Budget", format_currency(total_budget))
            st.metric("Total Spend", format_currency(total_spend))
            st.metric("Total Revenue", format_currency(total_revenue))
            
            if total_spend > 0:
                overall_roas = total_revenue / total_spend
                st.metric("Portfolio ROAS", f"{overall_roas:.2f}x")
                
        except Exception as e:
            st.error(f"Failed to calculate stats: {str(e)}")
    
    # Create New Campaign
    st.subheader("➕ Create New Campaign")
    
    with st.form("create_campaign"):
        col1, col2 = st.columns(2)
        
        with col1:
            campaign_name = st.text_input("Campaign Name", placeholder="Summer Sale 2024")
            campaign_type = st.selectbox(
                "Campaign Type",
                ["acquisition", "retention", "winback", "upsell"]
            )
            budget = st.number_input(
                "Budget ($)",
                min_value=100,
                max_value=50000,
                value=1000,
                step=100
            )
            target_segment = st.selectbox(
                "Target Segment",
                ["High Value Loyal", "Active Customers", "At Risk / Churned", "One-time Buyers"]
            )
        
        with col2:
            channels = st.multiselect(
                "Channels",
                ["google_ads", "meta_ads", "email", "sms", "display"],
                default=["google_ads", "meta_ads"]
            )
            
            objectives = st.text_area(
                "Custom Objectives (one per line)",
                placeholder="Increase customer acquisition\nImprove brand awareness"
            )
            
            start_date = st.date_input("Start Date", value=datetime.now().date())
            
        if st.form_submit_button("🚀 Create Campaign", type="primary"):
            try:
                # Parse objectives
                custom_objectives = [obj.strip() for obj in objectives.split('\n') if obj.strip()] if objectives else None
                
                # Create campaign
                with st.spinner("Creating campaign..."):
                    result = agent.campaign_manager.create_campaign(
                        campaign_type=campaign_type,
                        target_segment=target_segment,
                        budget=budget,
                        channels=channels,
                        custom_objectives=custom_objectives
                    )
                
                if result.get('success'):
                    st.success(f"Campaign '{campaign_name}' created successfully!")
                    st.json(result)
                    st.rerun()
                else:
                    st.error(f"Campaign creation failed: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                st.error(f"Error creating campaign: {str(e)}")
    
    # Campaign Performance Analysis
    st.subheader("📊 Campaign Performance Analysis")
    
    # Performance visualization
    try:
        campaigns = agent.campaign_manager.get_all_campaigns()
        
        if campaigns['campaigns']:
            # Prepare data for visualization
            perf_data = []
            for campaign_id, campaign_info in campaigns['campaigns'].items():
                performance = campaign_info.get('performance', {})
                perf_data.append({
                    'Campaign': campaign_info['name'][:20] + '...' if len(campaign_info['name']) > 20 else campaign_info['name'],
                    'Type': campaign_info['type'].title(),
                    'ROAS': performance.get('roas', 0),
                    'CTR': performance.get('ctr', 0) * 100,
                    'Conversion Rate': performance.get('conversion_rate', 0) * 100,
                    'Spend': performance.get('cost', 0),
                    'Revenue': performance.get('revenue', 0)
                })
            
            if perf_data:
                df_perf = pd.DataFrame(perf_data)
                
                # ROAS comparison chart
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_roas = px.bar(
                        df_perf,
                        x='Campaign',
                        y='ROAS',
                        color='Type',
                        title="Campaign ROAS Comparison"
                    )
                    fig_roas.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_roas, use_container_width=True)
                
                with col2:
                    fig_revenue = px.scatter(
                        df_perf,
                        x='Spend',
                        y='Revenue',
                        size='ROAS',
                        color='Type',
                        hover_name='Campaign',
                        title="Spend vs Revenue (Size = ROAS)"
                    )
                    st.plotly_chart(fig_revenue, use_container_width=True)
                
    except Exception as e:
        st.error(f"Failed to create performance analysis: {str(e)}")


def show_performance_analytics(data_pipeline, evaluation_system):
    """Show performance analytics dashboard"""
    st.header("📊 Performance Analytics")
    
    # Get performance data
    current_metrics = evaluation_system.calculate_current_metrics()
    trends = evaluation_system.analyze_trends()
    
    # Performance Summary
    st.subheader("📈 Performance Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        roas = current_metrics.get('overall_roas', 0)
        st.metric("ROAS", f"{roas:.2f}x")
    
    with col2:
        ctr = current_metrics.get('overall_ctr', 0)
        st.metric("CTR", format_percentage(ctr * 100))
    
    with col3:
        conv_rate = current_metrics.get('overall_conversion_rate', 0)
        st.metric("Conversion Rate", format_percentage(conv_rate * 100))
    
    with col4:
        total_spend = current_metrics.get('total_spend', 0)
        st.metric("Total Spend", format_currency(total_spend))
    
    with col5:
        total_revenue = current_metrics.get('total_revenue', 0)
        st.metric("Total Revenue", format_currency(total_revenue))
    
    # Trend Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Performance Trends")
        
        if trends:
            trend_data = []
            for metric, trend_info in trends.items():
                trend_data.append({
                    'Metric': metric.replace('overall_', '').replace('_', ' ').title(),
                    'Current': trend_info['current_value'],
                    'Trend': trend_info['direction'],
                    'Change %': trend_info['percentage_change'],
                    'Confidence': trend_info['trend_confidence']
                })
            
            df_trends = pd.DataFrame(trend_data)
            
            # Color code by change direction
            colors = ['red' if x < 0 else 'green' if x > 0 else 'gray' for x in df_trends['Change %']]
            
            fig = px.bar(
                df_trends,
                x='Metric',
                y='Change %',
                color='Change %',
                color_continuous_scale=['red', 'yellow', 'green'],
                title="Performance Changes (%)",
                hover_data=['Trend', 'Confidence']
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Benchmark Comparison")
        
        # Benchmark comparison
        benchmarks = {
            'ROAS': 3.0,
            'CTR': 2.5,
            'Conversion Rate': 3.5,
            'Customer LTV': 250.0
        }
        
        current_values = {
            'ROAS': current_metrics.get('overall_roas', 0),
            'CTR': current_metrics.get('overall_ctr', 0) * 100,
            'Conversion Rate': current_metrics.get('overall_conversion_rate', 0) * 100,
            'Customer LTV': current_metrics.get('avg_clv', 0)
        }
        
        benchmark_data = []
        for metric in benchmarks:
            benchmark_data.append({
                'Metric': metric,
                'Current': current_values[metric],
                'Benchmark': benchmarks[metric],
                'vs Benchmark': ((current_values[metric] - benchmarks[metric]) / benchmarks[metric] * 100) if benchmarks[metric] > 0 else 0
            })
        
        df_benchmark = pd.DataFrame(benchmark_data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Current',
            x=df_benchmark['Metric'],
            y=df_benchmark['Current'],
            marker_color='lightblue'
        ))
        
        fig.add_trace(go.Bar(
            name='Benchmark',
            x=df_benchmark['Metric'],
            y=df_benchmark['Benchmark'],
            marker_color='darkblue'
        ))
        
        fig.update_layout(
            title="Current vs Industry Benchmarks",
            barmode='group',
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed Analytics
    st.subheader("🔍 Detailed Analytics")
    
    # Channel Performance
    channel_performance = current_metrics.get('channel_performance', {})
    
    if channel_performance:
        st.write("**Channel Performance Breakdown:**")
        
        channel_data = []
        for channel, metrics in channel_performance.items():
            channel_data.append({
                'Channel': channel.replace('_', ' ').title(),
                'Spend': format_currency(metrics.get('spend', 0)),
                'Revenue': format_currency(metrics.get('revenue', 0)),
                'ROAS': f"{metrics.get('roas', 0):.2f}x",
                'Conversions': format_large_number(metrics.get('conversions', 0)),
                'Campaigns': metrics.get('campaigns', 0)
            })
        
        df_channels = pd.DataFrame(channel_data)
        st.dataframe(df_channels, use_container_width=True)
        
        # Channel performance visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig_spend = px.pie(
                df_channels,
                values=[float(x.replace('$', '').replace(',', '')) for x in df_channels['Spend']],
                names='Channel',
                title="Spend Distribution by Channel"
            )
            st.plotly_chart(fig_spend, use_container_width=True)
        
        with col2:
            fig_roas = px.bar(
                df_channels,
                x='Channel',
                y=[float(x.replace('x', '')) for x in df_channels['ROAS']],
                title="ROAS by Channel",
                color=[float(x.replace('x', '')) for x in df_channels['ROAS']],
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_roas, use_container_width=True)
    
    # Anomaly Detection
    st.subheader("🚨 Anomaly Detection")
    
    anomalies = evaluation_system.detect_anomalies(current_metrics)
    
    if anomalies:
        st.warning(f"Detected {len(anomalies)} anomalies in current performance metrics.")
        
        anomaly_data = []
        for anomaly in anomalies:
            anomaly_data.append({
                'Metric': anomaly.metric_name.replace('_', ' ').title(),
                'Current Value': anomaly.current_value,
                'Expected Range': f"{anomaly.expected_range[0]:.2f} - {anomaly.expected_range[1]:.2f}",
                'Anomaly Score': f"{anomaly.anomaly_score:.2f}",
                'Severity': anomaly.severity.title(),
                'Recommended Action': anomaly.recommended_action.replace('_', ' ').title()
            })
        
        df_anomalies = pd.DataFrame(anomaly_data)
        st.dataframe(df_anomalies, use_container_width=True)
        
    else:
        st.success("No anomalies detected in current performance metrics.")


def show_customer_insights(data_pipeline):
    """Show customer insights and segmentation"""
    st.header("👥 Customer Insights")
    
    # Get customer segmentation data
    try:
        segments_data = data_pipeline.get_customer_segments()
        performance_metrics = data_pipeline.get_performance_metrics()
        
        # Segmentation Overview
        st.subheader("🎯 Customer Segmentation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Customers", format_large_number(segments_data['total_customers']))
        
        with col2:
            st.metric("Segments Identified", len(segments_data['segments']))
        
        with col3:
            quality_score = segments_data.get('segmentation_quality', 0)
            st.metric("Segmentation Quality", f"{quality_score:.3f}")
        
        # Segment Details
        st.subheader("📊 Segment Analysis")
        
        segment_comparison = []
        for segment_id, segment_info in segments_data['segments'].items():
            segment_comparison.append({
                'Segment': segment_info['name'],
                'Size': format_large_number(segment_info['size']),
                'Avg CLV': format_currency(segment_info['avg_clv']),
                'Churn Risk': format_percentage(segment_info['churn_risk'] * 100),
                'Recommended Channels': ', '.join(segment_info['recommended_channels'][:2]),
                'Suggested Campaigns': ', '.join(segment_info['suggested_campaigns'])
            })
        
        df_segments = pd.DataFrame(segment_comparison)
        st.dataframe(df_segments, use_container_width=True)
        
        # Segment Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Segment size distribution
            sizes = [info['size'] for info in segments_data['segments'].values()]
            names = [info['name'] for info in segments_data['segments'].values()]
            
            fig_pie = px.pie(
                values=sizes,
                names=names,
                title="Customer Distribution by Segment"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # CLV vs Churn Risk scatter
            clv_data = []
            for segment_info in segments_data['segments'].values():
                clv_data.append({
                    'Segment': segment_info['name'],
                    'Average CLV': segment_info['avg_clv'],
                    'Churn Risk': segment_info['churn_risk'] * 100,
                    'Size': segment_info['size']
                })
            
            df_clv = pd.DataFrame(clv_data)
            
            fig_scatter = px.scatter(
                df_clv,
                x='Churn Risk',
                y='Average CLV',
                size='Size',
                color='Segment',
                title="CLV vs Churn Risk by Segment",
                hover_name='Segment'
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Detailed Segment Analysis
        st.subheader("🔍 Detailed Segment Characteristics")
        
        selected_segment = st.selectbox(
            "Select Segment for Detailed Analysis",
            list(segments_data['segments'].keys()),
            format_func=lambda x: segments_data['segments'][x]['name']
        )
        
        if selected_segment:
            segment_detail = segments_data['segments'][selected_segment]
            characteristics = segment_detail['characteristics']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Segment Characteristics:**")
                st.json({
                    'Average Recency (days)': round(characteristics['avg_recency'], 1),
                    'Average Frequency': round(characteristics['avg_frequency'], 1),
                    'Average Monetary Value': format_currency(characteristics['avg_monetary']),
                    'Average Order Value': format_currency(characteristics['avg_order_value']),
                    'Dominant Category': characteristics['dominant_category'],
                    'Dominant Channel': characteristics['dominant_channel'],
                    'Churn Rate': format_percentage(characteristics['churn_rate'] * 100)
                })
            
            with col2:
                st.write("**Recommendations:**")
                st.write("🎯 **Recommended Channels:**")
                for channel in segment_detail['recommended_channels']:
                    st.write(f"- {channel.replace('_', ' ').title()}")
                
                st.write("📢 **Suggested Campaigns:**")
                for campaign in segment_detail['suggested_campaigns']:
                    st.write(f"- {campaign.replace('_', ' ').title()}")
        
        # Customer Lifetime Value Analysis
        st.subheader("💰 Customer Lifetime Value Analysis")
        
        clv_metrics = {
            'Average CLV': performance_metrics.get('avg_clv', 0),
            'Active Customer Rate': performance_metrics.get('active_customer_rate', 0),
            'Churn Rate': performance_metrics.get('churn_rate', 0),
            'Average Order Value': performance_metrics.get('avg_order_value', 0)
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg CLV", format_currency(clv_metrics['Average CLV']))
        
        with col2:
            st.metric("Active Rate", format_percentage(clv_metrics['Active Customer Rate'] * 100))
        
        with col3:
            st.metric("Churn Rate", format_percentage(clv_metrics['Churn Rate'] * 100))
        
        with col4:
            st.metric("Avg Order Value", format_currency(clv_metrics['Average Order Value']))
        
    except Exception as e:
        st.error(f"Failed to load customer insights: {str(e)}")


def show_ab_testing(evaluation_system):
    """Show A/B testing interface"""
    st.header("🧪 A/B Testing")
    
    st.subheader("🔬 Create New A/B Test")
    
    with st.form("create_ab_test"):
        col1, col2 = st.columns(2)
        
        with col1:
            test_name = st.text_input("Test Name", placeholder="Email Subject Line Test")
            campaign_id = st.text_input("Campaign ID", placeholder="CAMP_12345")
            success_metric = st.selectbox(
                "Success Metric",
                ["conversion_rate", "click_through_rate", "revenue_per_visitor", "engagement_rate"]
            )
            duration_days = st.number_input("Test Duration (days)", min_value=1, max_value=30, value=14)
        
        with col2:
            traffic_split = st.slider("Traffic Split (% for Variant A)", min_value=10, max_value=90, value=50) / 100
            confidence_level = st.slider("Confidence Level", min_value=0.80, max_value=0.99, value=0.95, step=0.01)
            minimum_sample_size = st.number_input("Minimum Sample Size", min_value=100, max_value=10000, value=1000)
        
        variant_a_config = st.text_area("Variant A Configuration (JSON)", placeholder='{"subject": "Original Subject"}')
        variant_b_config = st.text_area("Variant B Configuration (JSON)", placeholder='{"subject": "New Subject"}')
        
        if st.form_submit_button("🚀 Create A/B Test"):
            try:
                # Parse configurations
                variant_a = json.loads(variant_a_config) if variant_a_config else {}
                variant_b = json.loads(variant_b_config) if variant_b_config else {}
                
                # Create test
                test_config = {
                    'duration_days': duration_days,
                    'traffic_split': traffic_split,
                    'success_metric': success_metric,
                    'minimum_sample_size': minimum_sample_size,
                    'confidence_level': confidence_level,
                    'variant_a': variant_a,
                    'variant_b': variant_b
                }
                
                test_id = evaluation_system.create_ab_test(test_name, campaign_id, test_config)
                
                st.success(f"A/B Test '{test_name}' created successfully!")
                st.code(f"Test ID: {test_id}")
                
            except json.JSONDecodeError:
                st.error("Invalid JSON format in variant configurations")
            except Exception as e:
                st.error(f"Failed to create A/B test: {str(e)}")
    
    # Simulate A/B Test Results
    st.subheader("📊 A/B Test Results")
    
    # Generate mock test results
    if st.button("🎲 Generate Sample Test Results"):
        # Mock data for demonstration
        variant_a_data = {
            'visitors': 5000,
            'conversions': 250,
            'revenue': 15000,
            'bounce_rate': 0.35
        }
        
        variant_b_data = {
            'visitors': 4950,
            'conversions': 297,
            'revenue': 17820,
            'bounce_rate': 0.32
        }
        
        try:
            # Analyze the test
            result = evaluation_system.analyze_ab_test("MOCK_TEST_001", variant_a_data, variant_b_data)
            
            # Display results
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("🅰️ Variant A")
                st.metric("Visitors", format_large_number(variant_a_data['visitors']))
                st.metric("Conversions", format_large_number(variant_a_data['conversions']))
                st.metric("Conversion Rate", format_percentage(result.variant_a_performance['conversion_rate'] * 100))
                st.metric("Revenue", format_currency(variant_a_data['revenue']))
            
            with col2:
                st.subheader("🅱️ Variant B")
                st.metric("Visitors", format_large_number(variant_b_data['visitors']))
                st.metric("Conversions", format_large_number(variant_b_data['conversions']))
                st.metric("Conversion Rate", format_percentage(result.variant_b_performance['conversion_rate'] * 100))
                st.metric("Revenue", format_currency(variant_b_data['revenue']))
            
            with col3:
                st.subheader("📈 Test Results")
                
                winner_color = "🟢" if result.winner == "B" else "🔵" if result.winner == "A" else "🟡"
                st.metric("Winner", f"{winner_color} Variant {result.winner}")
                
                st.metric("Confidence Level", format_percentage(result.confidence_level * 100))
                
                significance_color = "🟢" if result.statistical_significance else "🔴"
                st.metric("Statistical Significance", f"{significance_color} {'Yes' if result.statistical_significance else 'No'}")
                
                st.metric("Lift", f"{result.lift_percentage:+.1f}%")
            
            # Visualization
            comparison_data = pd.DataFrame([
                {'Variant': 'A', 'Conversion Rate': result.variant_a_performance['conversion_rate'] * 100},
                {'Variant': 'B', 'Conversion Rate': result.variant_b_performance['conversion_rate'] * 100}
            ])
            
            fig = px.bar(
                comparison_data,
                x='Variant',
                y='Conversion Rate',
                title="Conversion Rate Comparison",
                color='Conversion Rate',
                color_continuous_scale=['red', 'green']
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Test summary
            st.subheader("📋 Test Summary")
            st.json({
                "Test ID": result.test_id,
                "Test Name": result.test_name,
                "Winner": result.winner,
                "Lift Percentage": f"{result.lift_percentage:.2f}%",
                "Statistical Significance": result.statistical_significance,
                "Confidence Level": f"{result.confidence_level:.2%}",
                "Sample Size A": result.sample_size_a,
                "Sample Size B": result.sample_size_b
            })
            
        except Exception as e:
            st.error(f"Failed to analyze A/B test: {str(e)}")


def show_system_monitoring(agent, evaluation_system):
    """Show system monitoring dashboard"""
    st.header("🖥️ System Monitoring")
    
    # System Health Overview
    st.subheader("🏥 System Health")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "🟢 Healthy")
    
    with col2:
        st.metric("Agent Uptime", "99.8%")
    
    with col3:
        st.metric("API Response Time", "245ms")
    
    with col4:
        st.metric("Error Rate", "0.02%")
    
    # Performance Monitoring
    st.subheader("📊 Performance Monitoring")
    
    # Generate mock monitoring data
    import random
    
    # System metrics over time
    times = pd.date_range(start=datetime.now() - timedelta(hours=24), end=datetime.now(), freq='1H')
    monitoring_data = pd.DataFrame({
        'Time': times,
        'CPU Usage (%)': [random.uniform(20, 80) for _ in times],
        'Memory Usage (%)': [random.uniform(30, 70) for _ in times],
        'API Response Time (ms)': [random.uniform(200, 400) for _ in times],
        'Request Rate (req/min)': [random.uniform(50, 200) for _ in times]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_cpu = px.line(
            monitoring_data,
            x='Time',
            y='CPU Usage (%)',
            title="CPU Usage (24h)"
        )
        st.plotly_chart(fig_cpu, use_container_width=True)
        
        fig_api = px.line(
            monitoring_data,
            x='Time',
            y='API Response Time (ms)',
            title="API Response Time (24h)"
        )
        st.plotly_chart(fig_api, use_container_width=True)
    
    with col2:
        fig_memory = px.line(
            monitoring_data,
            x='Time',
            y='Memory Usage (%)',
            title="Memory Usage (24h)"
        )
        st.plotly_chart(fig_memory, use_container_width=True)
        
        fig_requests = px.line(
            monitoring_data,
            x='Time',
            y='Request Rate (req/min)',
            title="Request Rate (24h)"
        )
        st.plotly_chart(fig_requests, use_container_width=True)
    
    # Error Monitoring
    st.subheader("🚨 Error Monitoring")
    
    # Mock error data
    error_data = [
        {'Time': '2024-01-15 14:23:15', 'Level': 'ERROR', 'Component': 'Campaign Manager', 'Message': 'API rate limit exceeded'},
        {'Time': '2024-01-15 12:45:32', 'Level': 'WARNING', 'Component': 'Data Pipeline', 'Message': 'High customer churn detected'},
        {'Time': '2024-01-15 11:12:08', 'Level': 'INFO', 'Component': 'Agent Core', 'Message': 'Optimization iteration completed'},
        {'Time': '2024-01-15 09:34:22', 'Level': 'ERROR', 'Component': 'Evaluation System', 'Message': 'Anomaly detection threshold exceeded'}
    ]
    
    df_errors = pd.DataFrame(error_data)
    st.dataframe(df_errors, use_container_width=True)
    
    # Agent Performance Metrics
    st.subheader("🤖 Agent Performance Metrics")
    
    agent_status = agent.get_agent_status()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Agent Statistics")
        st.json({
            "Total Iterations": agent_status['total_iterations'],
            "Active Campaigns": agent_status['active_campaigns_count'],
            "Current Phase": agent_status['state']['phase'],
            "Last Action": agent_status['last_iteration_time'],
            "Decision Accuracy": "94.2%",
            "Optimization Success Rate": "87.5%"
        })
    
    with col2:
        st.subheader("Performance Trends")
        
        # Mock performance trend data
        trend_dates = pd.date_range(start=datetime.now() - timedelta(days=7), end=datetime.now(), freq='1D')
        trend_data = pd.DataFrame({
            'Date': trend_dates,
            'ROAS Improvement': [random.uniform(-5, 15) for _ in trend_dates],
            'Cost Efficiency': [random.uniform(-3, 12) for _ in trend_dates],
            'Conversion Rate': [random.uniform(-2, 8) for _ in trend_dates]
        })
        
        fig_trends = px.line(
            trend_data,
            x='Date',
            y=['ROAS Improvement', 'Cost Efficiency', 'Conversion Rate'],
            title="Agent Performance Trends (% improvement)"
        )
        st.plotly_chart(fig_trends, use_container_width=True)
    
    # System Alerts
    st.subheader("🔔 System Alerts")
    
    alerts = [
        {"Level": "🟡 Warning", "Message": "Campaign budget utilization above 85%", "Time": "5 minutes ago"},
        {"Level": "🟢 Info", "Message": "New customer segment identified", "Time": "23 minutes ago"},
        {"Level": "🔴 Critical", "Message": "ROAS dropped below threshold for campaign CAMP_001", "Time": "1 hour ago"},
        {"Level": "🟢 Info", "Message": "A/B test completed with significant results", "Time": "2 hours ago"}
    ]
    
    for alert in alerts:
        st.markdown(f"""
        <div style="padding: 0.5rem; margin: 0.25rem 0; border-left: 4px solid #007bff; background-color: #f8f9fa;">
            <strong>{alert['Level']}</strong>: {alert['Message']} <span style="color: #6c757d; font-size: 0.8rem;">({alert['Time']})</span>
        </div>
        """, unsafe_allow_html=True)


def show_settings():
    """Show settings and configuration"""
    st.header("⚙️ Settings & Configuration")
    
    st.subheader("🔧 Agent Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**AI Model Settings**")
        
        # Model selection
        model = st.selectbox("AI Model", ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"], index=0)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        max_iterations = st.number_input("Max Iterations", 1, 20, 10)
        
        st.write("**Budget Constraints**")
        max_daily_budget = st.number_input("Max Daily Budget ($)", 100, 50000, 1000)
        min_roas_threshold = st.number_input("Min ROAS Threshold", 1.0, 10.0, 2.0, 0.1)
        
        st.write("**Performance Thresholds**")
        min_ctr = st.number_input("Min CTR (%)", 0.1, 10.0, 1.0, 0.1) / 100
        min_conversion_rate = st.number_input("Min Conversion Rate (%)", 0.1, 20.0, 2.0, 0.1) / 100
    
    with col2:
        st.write("**API Configuration**")
        
        # API keys (masked for security)
        openai_key = st.text_input("OpenAI API Key", value="sk-***", type="password")
        anthropic_key = st.text_input("Anthropic API Key", value="ant-***", type="password")
        
        st.write("**Logging Configuration**")
        log_level = st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
        
        st.write("**Data Configuration**")
        sample_data_size = st.number_input("Sample Data Size", 1000, 50000, 10000)
        segment_count = st.number_input("Customer Segments", 3, 10, 5)
    
    # Save settings
    if st.button("💾 Save Configuration", type="primary"):
        st.success("Configuration saved successfully!")
        st.info("Note: Some changes may require an application restart to take effect.")
    
    # Export/Import Configuration
    st.subheader("📁 Configuration Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📤 Export Configuration"):
            config = {
                "agent_model": model,
                "temperature": temperature,
                "max_iterations": max_iterations,
                "max_daily_budget": max_daily_budget,
                "min_roas_threshold": min_roas_threshold,
                "min_ctr": min_ctr,
                "min_conversion_rate": min_conversion_rate,
                "log_level": log_level,
                "sample_data_size": sample_data_size,
                "segment_count": segment_count
            }
            
            st.download_button(
                "Download Configuration",
                data=json.dumps(config, indent=2),
                file_name="agent_config.json",
                mime="application/json"
            )
    
    with col2:
        uploaded_config = st.file_uploader("📥 Import Configuration", type="json")
        if uploaded_config:
            try:
                config = json.load(uploaded_config)
                st.success("Configuration imported successfully!")
                st.json(config)
            except:
                st.error("Invalid configuration file")
    
    # System Information
    st.subheader("ℹ️ System Information")
    
    system_info = {
        "Application Version": "1.0.0",
        "Python Version": "3.9+",
        "Streamlit Version": st.__version__,
        "Database": "SQLite",
        "Last Updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    st.json(system_info)


if __name__ == "__main__":
    main()