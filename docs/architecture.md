# Architecture Documentation

## System Overview

The E-commerce Growth Agent is a sophisticated AI-powered system designed to autonomously manage and optimize marketing campaigns across multiple channels. The system follows a **plan→act→observe** loop, continuously learning and improving performance.

## Core Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       STREAMLIT UI                          │
│                  (streamlit_app.py)                         │
└─────────────────┬───────────────────────────────────────────┘
                  │
┌─────────────────┼───────────────────────────────────────────┐
│                 │        AGENT CORE                         │
│                 │       (src/agent.py)                      │
│                 │                                           │
│    ┌────────────┼────────────┐                             │
│    │     PLAN   │   ACT      │   OBSERVE                   │
│    │            │            │                             │
│    │ • Analyze  │ • Execute  │ • Monitor                   │
│    │ • Decide   │ • Create   │ • Learn                     │
│    │ • Strategy │ • Optimize │ • Adapt                     │
│    └────────────┼────────────┘                             │
└─────────────────┼───────────────────────────────────────────┘
                  │
┌─────────────────┼───────────────────────────────────────────┐
│    Data Pipeline     │   Campaign Manager   │   Evaluation    │
│  (data_pipeline.py)  │ (campaign_manager.py)│ (evaluation.py) │
│                      │                      │                 │
│ • Customer Seg.      │ • Multi-channel      │ • A/B Tests     │
│ • Behavioral Data    │ • Creative Gen.      │ • Metrics       │
│ • Market Analysis    │ • Budget Optim.      │ • Anomalies     │
│ • Opportunity ID     │ • Performance Track  │ • ROI Analysis  │
└─────────────────┬────┴──────────────────────┴─────────────────┘
                  │
┌─────────────────┼───────────────────────────────────────────┐
│                       API INTEGRATIONS                      │
│     (src/integrations/)                                     │
│                                                             │
│  Google Ads    │  Meta Ads     │  Klaviyo     │  Shopify    │
│  (Mock/Real)   │  (Mock/Real)  │  (Mock/Real) │  (Mock)     │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Agent Core (`src/agent.py`)

The heart of the system implementing autonomous decision-making:

#### Plan Phase
- **Data Analysis**: Ingests customer behavior, campaign performance, and market data
- **Opportunity Identification**: Uses ML and statistical analysis to find optimization opportunities
- **Strategic Planning**: Generates comprehensive marketing strategies using LLMs
- **Risk Assessment**: Evaluates potential risks and constraints

#### Act Phase
- **Campaign Creation**: Automatically creates multi-channel campaigns
- **Budget Optimization**: Allocates budgets across channels using performance data
- **Creative Generation**: Uses AI to generate ad copy, email content, and targeting parameters
- **Execution**: Implements decisions through platform APIs

#### Observe Phase
- **Performance Monitoring**: Tracks KPIs across all campaigns and channels
- **Anomaly Detection**: Identifies unusual patterns or performance issues
- **Learning Integration**: Updates models and strategies based on results
- **Feedback Loops**: Adjusts future decisions based on observed outcomes

### 2. Data Pipeline (`src/data_pipeline.py`)

Comprehensive data processing and analytics engine:

#### Customer Segmentation
- **RFM Analysis**: Recency, Frequency, Monetary value segmentation
- **K-means Clustering**: ML-powered customer grouping
- **Behavioral Patterns**: Purchase behavior and lifecycle analysis
- **Predictive Modeling**: Customer lifetime value and churn prediction

#### Data Processing
- **ETL Operations**: Extract, Transform, Load data from multiple sources
- **Data Quality**: Validation, cleaning, and integrity checks
- **Feature Engineering**: Create derived metrics and indicators
- **Trend Analysis**: Statistical trend detection and forecasting

### 3. Campaign Manager (`src/campaign_manager.py`)

Multi-channel campaign orchestration:

#### Creative Generation
- **AI-Powered Copy**: LLM-generated ad copy and email content
- **A/B Variant Creation**: Automatic generation of test variants
- **Channel Adaptation**: Content optimization for specific platforms
- **Brand Consistency**: Maintains brand voice across channels

#### Budget Optimization
- **Performance-Based Allocation**: Budget distribution based on ROAS
- **Dynamic Adjustment**: Real-time budget reallocation
- **Constraint Management**: Respects budget limits and performance thresholds
- **ROI Maximization**: Continuous optimization for maximum return

### 4. Evaluation System (`src/evaluation.py`)

Performance tracking and optimization:

#### A/B Testing
- **Automated Test Creation**: Statistical test design and setup
- **Significance Testing**: Chi-square and t-test analysis
- **Winner Determination**: Statistical significance assessment
- **Performance Tracking**: Comprehensive test result analysis

#### Anomaly Detection
- **Statistical Methods**: Z-score and IQR-based outlier detection
- **Threshold Monitoring**: Business rule-based alerts
- **Trend Analysis**: Time series anomaly identification
- **Alert Generation**: Automatic notification of issues

## Data Models

### Customer Segmentation Model

```python
@dataclass
class CustomerSegment:
    segment_id: str                    # Unique identifier
    name: str                         # Human-readable name
    size: int                         # Number of customers
    characteristics: Dict[str, Any]    # RFM and behavioral metrics
    avg_clv: float                    # Average Customer Lifetime Value
    churn_risk: float                 # Predicted churn probability
    recommended_channels: List[str]    # Optimal marketing channels
    suggested_campaigns: List[str]     # Recommended campaign types
```

### Campaign Structure

```python
@dataclass
class Campaign:
    campaign_id: str                   # Unique identifier
    name: str                         # Campaign name
    campaign_type: str                # acquisition, retention, winback, upsell
    channel: str                      # Primary channel
    target_segment: str               # Target customer segment
    budget: float                     # Total campaign budget
    daily_budget: float               # Daily spending limit
    start_date: datetime              # Campaign start
    end_date: datetime                # Campaign end
    status: str                       # active, paused, completed
    objectives: List[str]             # Campaign goals
    targeting_parameters: Dict        # Audience targeting config
    creative_assets: List[CreativeAsset]  # Ad creatives and copy
    performance_metrics: Dict[str, float]  # Current performance data
    optimization_history: List[Dict]  # History of optimizations
```

### Performance Metrics

```python
performance_metrics = {
    # Campaign Performance
    "roas": float,                    # Return on Ad Spend
    "ctr": float,                     # Click-Through Rate
    "conversion_rate": float,         # Conversion Rate
    "cpc": float,                     # Cost Per Click
    "cpm": float,                     # Cost Per Mille
    
    # Customer Metrics
    "customer_ltv": float,            # Customer Lifetime Value
    "cac": float,                     # Customer Acquisition Cost
    "churn_rate": float,              # Customer Churn Rate
    "active_rate": float,             # Active Customer Rate
    
    # Business Metrics
    "total_revenue": float,           # Total Revenue
    "total_spend": float,             # Total Ad Spend
    "total_conversions": int,         # Total Conversions
    "avg_order_value": float          # Average Order Value
}
```

## Decision-Making Framework

### 1. Data-Driven Decisions

The agent makes decisions based on:
- **Historical Performance**: Past campaign and channel performance
- **Statistical Significance**: A/B test results and confidence intervals
- **Predictive Models**: Customer behavior and performance forecasts
- **Market Trends**: Industry benchmarks and competitive analysis

### 2. Safety Guardrails

Built-in safety mechanisms:
- **Budget Constraints**: Hard limits on daily and total spending
- **Performance Thresholds**: Minimum ROAS and conversion rate requirements
- **Anomaly Detection**: Automatic detection of unusual patterns
- **Human Oversight**: Optional manual approval for major decisions

### 3. Learning Integration

Continuous improvement through:
- **Performance Feedback**: Learning from campaign results
- **A/B Test Insights**: Incorporating test findings into future decisions
- **Trend Adaptation**: Adjusting to changing market conditions
- **Model Updates**: Improving prediction accuracy over time

## API Integration Architecture

### Mock vs Real APIs

The system supports both mock and real API integrations:

#### Mock APIs (Default)
- Fully functional simulation of real platforms
- Realistic performance data generation
- No external dependencies or costs
- Perfect for development and demonstration

#### Real APIs (Production)
- Google Ads API integration
- Meta Marketing API (Facebook/Instagram)
- Klaviyo Email Marketing API
- Shopify Admin API

### Integration Pattern

```python
class PlatformInterface:
    """Abstract interface for marketing platforms"""
    
    def create_campaign(self, campaign_data: Dict) -> Dict:
        pass
    
    def get_campaign_metrics(self, campaign_id: str) -> Dict:
        pass
    
    def update_campaign_budget(self, campaign_id: str, budget: float) -> Dict:
        pass
    
    def pause_campaign(self, campaign_id: str) -> Dict:
        pass

class GoogleAdsIntegration(PlatformInterface):
    """Google Ads API implementation"""
    # Real or mock implementation

class MetaAdsIntegration(PlatformInterface):
    """Meta Ads API implementation"""
    # Real or mock implementation
```

## Performance Optimization

### 1. Caching Strategy

- **Metrics Caching**: Cache performance data for faster dashboard loading
- **Segment Caching**: Store customer segments to avoid recalculation
- **API Response Caching**: Cache platform API responses when appropriate

### 2. Database Optimization

- **Indexing**: Proper database indexes for query performance
- **Data Partitioning**: Separate historical and current data
- **Query Optimization**: Efficient SQL queries for large datasets

### 3. Asynchronous Operations

- **Background Processing**: Long-running operations in background tasks
- **Queue Management**: Task queues for campaign creation and optimization
- **Real-time Updates**: WebSocket connections for live dashboard updates

## Security Considerations

### 1. API Key Management

- **Environment Variables**: Store sensitive data in environment variables
- **Key Rotation**: Regular rotation of API keys and secrets
- **Access Control**: Role-based access to different system components

### 2. Data Protection

- **Encryption**: Encrypt sensitive customer data at rest
- **Access Logging**: Log all data access and modifications
- **Privacy Compliance**: GDPR and CCPA compliance measures

### 3. System Security

- **Input Validation**: Validate all user inputs and API responses
- **Rate Limiting**: Protect against API abuse and DoS attacks
- **Monitoring**: Continuous security monitoring and alerting

## Scalability Design

### 1. Horizontal Scaling

- **Microservices**: Separate services for different components
- **Load Balancing**: Distribute load across multiple instances
- **Database Sharding**: Split large datasets across multiple databases

### 2. Cloud Architecture

- **Container Deployment**: Docker containers for easy scaling
- **Auto-scaling**: Automatic scaling based on demand
- **Cloud Services**: Leverage cloud-native services for scalability

### 3. Performance Monitoring

- **Metrics Collection**: Comprehensive system and business metrics
- **Performance Alerts**: Automatic alerts for performance issues
- **Capacity Planning**: Proactive scaling based on usage trends

## Deployment Architecture

### Development Environment
```
Local Machine
├── Streamlit App (localhost:8501)
├── SQLite Database
├── Mock APIs
└── File-based Logging
```

### Production Environment
```
Cloud Infrastructure
├── Load Balancer
├── Application Servers (Multiple Instances)
├── Database Cluster (PostgreSQL/MySQL)
├── Cache Layer (Redis)
├── Message Queue (RabbitMQ/Celery)
├── Monitoring (Prometheus/Grafana)
└── Log Aggregation (ELK Stack)
```

## Future Enhancements

### Phase 2: Advanced Features
- **Real-time Optimization**: Sub-hourly campaign adjustments
- **Advanced ML Models**: Deep learning for customer behavior prediction
- **Voice Interface**: Voice commands for campaign management
- **Mobile Application**: Native mobile app for monitoring

### Phase 3: Enterprise Features
- **Multi-tenant Architecture**: Support for multiple clients
- **Advanced Security**: Enterprise-grade security features
- **Custom Integrations**: Support for custom platform APIs
- **Advanced Analytics**: Machine learning insights and predictions

## Conclusion

The E-commerce Growth Agent represents a sophisticated approach to autonomous marketing optimization. By combining AI-powered decision making with comprehensive data analysis and multi-channel execution, the system delivers significant improvements in marketing ROI while reducing manual effort.

The modular architecture ensures scalability and maintainability, while the extensive testing and safety features provide reliability and trustworthiness. The system is designed to grow with your business needs, from small-scale optimization to enterprise-level marketing automation.