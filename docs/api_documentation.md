# API Documentation - E-commerce Growth Agent

Comprehensive documentation for all mock API integrations and system interfaces.

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Google Ads Mock API](#google-ads-mock-api)
4. [Meta Ads Mock API](#meta-ads-mock-api)
5. [Klaviyo Mock API](#klaviyo-mock-api)
6. [Shopify Mock API](#shopify-mock-api)
7. [Agent Core API](#agent-core-api)
8. [Data Pipeline API](#data-pipeline-api)
9. [Evaluation System API](#evaluation-system-api)
10. [Error Handling](#error-handling)
11. [Rate Limiting](#rate-limiting)
12. [Examples](#examples)

## Overview

The E-commerce Growth Agent provides mock API integrations for major marketing platforms. These APIs simulate real platform behavior for development, testing, and demonstration purposes.

### Base URLs

- **Local Development**: `http://localhost:8501`
- **API Version**: `v1`
- **Mock APIs**: Fully functional without external dependencies

### Response Format

All APIs return JSON responses with consistent structure:

```json
{
  "success": true,
  "data": {...},
  "error": null,
  "timestamp": "2024-01-15T10:30:00Z",
  "request_id": "req_abc123"
}
```

## Authentication

### API Key Authentication

Mock APIs use simulated authentication but don't require real API keys:

```python
# Environment variables (optional for mocks)
GOOGLE_ADS_API_KEY=mock_key_google
META_ADS_API_KEY=mock_key_meta
KLAVIYO_API_KEY=mock_key_klaviyo
SHOPIFY_ACCESS_TOKEN=mock_token_shopify
```

### Usage Example

```python
from src.integrations import get_platform_client

# Create authenticated client
client = get_platform_client('google_ads')
account_info = client.get_account_info()
```

## Google Ads Mock API

### Overview

Simulates Google Ads API for search and display campaign management.

#### Base Client

```python
from src.integrations.google_ads_mock import GoogleAdsMock

client = GoogleAdsMock()
```

### Endpoints

#### 1. Account Information

**Method**: `get_account_info()`

**Description**: Retrieve account details

**Parameters**: None

**Response**:
```json
{
  "account_id": "123-456-7890",
  "account_name": "Demo E-commerce Account",
  "currency": "USD",
  "time_zone": "America/New_York",
  "total_campaigns": 5,
  "active_campaigns": 3,
  "paused_campaigns": 2
}
```

**Example**:
```python
account_info = client.get_account_info()
print(f"Account: {account_info['account_name']}")
```

#### 2. Create Campaign

**Method**: `create_campaign(campaign_data)`

**Description**: Create a new Google Ads campaign

**Parameters**:
- `campaign_data` (dict): Campaign configuration

**Request**:
```python
campaign_data = {
    'name': 'Summer Sale 2024',
    'budget': 2000.0,
    'targeting': {
        'keywords': ['summer sale', 'discount clothes'],
        'location': 'United States',
        'age_range': '25-55'
    },
    'creative_assets': [
        {
            'asset_type': 'search_ad',
            'content': '{"headline": "Summer Sale", "description": "Up to 50% off!"}'
        }
    ]
}
```

**Response**:
```json
{
  "success": true,
  "platform_campaign_id": "GADS_A1B2C3D4",
  "campaign_name": "Summer Sale 2024",
  "status": "ENABLED",
  "message": "Campaign created successfully"
}
```

**Example**:
```python
result = client.create_campaign(campaign_data)
if result['success']:
    campaign_id = result['platform_campaign_id']
    print(f"Created campaign: {campaign_id}")
```

#### 3. Get Campaign Metrics

**Method**: `get_campaign_metrics(campaign_id)`

**Description**: Retrieve performance metrics for a campaign

**Parameters**:
- `campaign_id` (str): Campaign identifier

**Response**:
```json
{
  "impressions": 15420,
  "clicks": 387,
  "conversions": 23,
  "cost": 486.50,
  "revenue": 1840.00,
  "ctr": 0.0251,
  "cpc": 1.26,
  "conversion_rate": 0.0594,
  "roas": 3.78,
  "quality_score": 7.2
}
```

**Example**:
```python
metrics = client.get_campaign_metrics('GADS_A1B2C3D4')
print(f"ROAS: {metrics['roas']:.2f}x")
```

#### 4. Update Campaign Budget

**Method**: `update_campaign_budget(campaign_id, new_budget)`

**Description**: Update campaign budget

**Parameters**:
- `campaign_id` (str): Campaign identifier
- `new_budget` (float): New budget amount

**Response**:
```json
{
  "success": true,
  "campaign_id": "GADS_A1B2C3D4",
  "old_budget": 2000.0,
  "new_budget": 2500.0,
  "message": "Budget updated successfully"
}
```

#### 5. Campaign Management

**Methods**: 
- `pause_campaign(campaign_id)` - Pause campaign
- `enable_campaign(campaign_id)` - Enable campaign

**Response**:
```json
{
  "success": true,
  "campaign_id": "GADS_A1B2C3D4",
  "status": "PAUSED",
  "message": "Campaign paused successfully"
}
```

#### 6. Keyword Management

**Method**: `get_keyword_performance(campaign_id)`

**Description**: Get keyword-level performance data

**Response**:
```json
{
  "campaign_id": "GADS_A1B2C3D4",
  "keyword_performance": [
    {
      "keyword_id": "KW_12345678",
      "keyword_text": "summer sale",
      "match_type": "EXACT",
      "impressions": 5420,
      "clicks": 127,
      "conversions": 8,
      "cost": 156.70,
      "ctr": 0.0234,
      "conversion_rate": 0.0630,
      "quality_score": 8.5
    }
  ]
}
```

## Meta Ads Mock API

### Overview

Simulates Meta Marketing API for Facebook and Instagram advertising.

#### Base Client

```python
from src.integrations.meta_ads_mock import MetaAdsMock

client = MetaAdsMock()
```

### Endpoints

#### 1. Account Information

**Method**: `get_account_info()`

**Response**:
```json
{
  "account_id": "act_1234567890",
  "account_name": "Demo E-commerce Business",
  "currency": "USD",
  "account_status": "ACTIVE",
  "business_id": "1234567890",
  "total_campaigns": 8,
  "active_campaigns": 6,
  "paused_campaigns": 2
}
```

#### 2. Create Campaign

**Method**: `create_campaign(campaign_data)`

**Request**:
```python
campaign_data = {
    'name': 'Instagram Fashion Campaign',
    'budget': 1500.0,
    'targeting': {
        'demographics': {
            'age_min': 18,
            'age_max': 35,
            'genders': [2]  # Female
        },
        'interests': ['fashion', 'shopping'],
        'geographic': 'United States'
    },
    'optimization_goal': 'conversions',
    'creative_assets': [
        {
            'asset_type': 'news_feed_ad',
            'content': '{"primary_text": "New collection available!", "headline": "Shop Now"}'
        }
    ]
}
```

**Response**:
```json
{
  "success": true,
  "platform_campaign_id": "META_X9Y8Z7W6",
  "campaign_name": "Instagram Fashion Campaign",
  "objective": "CONVERSIONS",
  "status": "ACTIVE",
  "message": "Meta campaign created successfully"
}
```

#### 3. Get Campaign Metrics

**Method**: `get_campaign_metrics(campaign_id)`

**Response**:
```json
{
  "impressions": 28540,
  "clicks": 342,
  "conversions": 18,
  "spend": 487.30,
  "revenue": 1620.00,
  "post_engagement": 156,
  "video_views": 2840,
  "page_views": 298,
  "add_to_cart": 45,
  "ctr": 0.0120,
  "cpc": 1.43,
  "conversion_rate": 0.0526,
  "cpm": 17.08,
  "roas": 3.32,
  "frequency": 1.8,
  "reach": 15855
}
```

#### 4. Audience Management

**Method**: `create_custom_audience(audience_data)`

**Request**:
```python
audience_data = {
    'name': 'Website Visitors',
    'subtype': 'WEBSITE',
    'retention_days': 180
}
```

**Response**:
```json
{
  "success": true,
  "audience_id": "CA_ABCD1234",
  "name": "Website Visitors",
  "approximate_count": 15420,
  "message": "Custom audience created successfully"
}
```

**Method**: `create_lookalike_audience(source_audience_id, country, ratio)`

**Parameters**:
- `source_audience_id` (str): Source audience ID
- `country` (str): Target country (default: 'US')
- `ratio` (float): Audience size ratio (default: 0.01 = 1%)

**Response**:
```json
{
  "success": true,
  "audience_id": "LAL_EFGH5678",
  "name": "Lookalike_CA_ABCD1234_1%",
  "approximate_count": 2100000,
  "message": "Lookalike audience created successfully"
}
```

#### 5. Audience Insights

**Method**: `get_audience_insights(audience_id)`

**Response**:
```json
{
  "audience_id": "CA_ABCD1234",
  "insights": {
    "audience_size": 15420,
    "age_distribution": {
      "18-24": 0.22,
      "25-34": 0.31,
      "35-44": 0.28,
      "45-54": 0.14,
      "55+": 0.05
    },
    "gender_distribution": {
      "male": 0.48,
      "female": 0.52
    },
    "top_interests": ["Shopping", "Fashion", "Technology", "Travel", "Food"],
    "estimated_reach": 12500
  }
}
```

## Klaviyo Mock API

### Overview

Simulates Klaviyo API for email and SMS marketing automation.

#### Base Client

```python
from src.integrations.klaviyo_mock import KlaviyoMock

client = KlaviyoMock()
```

### Endpoints

#### 1. Account Information

**Method**: `get_account_info()`

**Response**:
```json
{
  "account_id": "PK_12345abcdef",
  "account_name": "Demo E-commerce Store",
  "company_name": "Demo Company",
  "time_zone": "America/New_York",
  "total_email_campaigns": 15,
  "total_sms_campaigns": 8,
  "total_subscribers": 5000,
  "email_subscribers": 4200,
  "sms_subscribers": 1500
}
```

#### 2. Create Email Campaign

**Method**: `create_email_campaign(campaign_data)`

**Request**:
```python
campaign_data = {
    'name': 'Weekly Newsletter',
    'targeting': {
        'segment_filters': {
            'engagement_level': 'high',
            'purchase_history': 'any'
        }
    },
    'creative_assets': [
        {
            'asset_type': 'email_campaign',
            'content': '{"subject_line": "Weekly Deals Inside!", "from_name": "Demo Store"}'
        }
    ]
}
```

**Response**:
```json
{
  "success": true,
  "platform_campaign_id": "EMAIL_ABC123DEF",
  "campaign_name": "Weekly Newsletter",
  "campaign_type": "email",
  "status": "DRAFT",
  "recipient_count": 850,
  "message": "Email campaign created successfully"
}
```

#### 3. Create SMS Campaign

**Method**: `create_sms_campaign(campaign_data)`

**Request**:
```python
campaign_data = {
    'name': 'Flash Sale Alert',
    'targeting': {
        'segment_filters': {
            'engagement_level': 'all'
        }
    },
    'creative_assets': [
        {
            'asset_type': 'sms_message',
            'content': '{"message": "Flash Sale! 20% off everything. Code: FLASH20"}'
        }
    ]
}
```

**Response**:
```json
{
  "success": true,
  "platform_campaign_id": "SMS_XYZ789GHI",
  "campaign_name": "Flash Sale Alert",
  "campaign_type": "sms",
  "status": "DRAFT",
  "recipient_count": 320,
  "message": "SMS campaign created successfully"
}
```

#### 4. Get Campaign Metrics

**Method**: `get_campaign_metrics(campaign_id)`

**Email Campaign Response**:
```json
{
  "recipients": 850,
  "delivered": 842,
  "bounced": 8,
  "opened": 236,
  "clicked": 47,
  "conversions": 12,
  "unsubscribed": 3,
  "marked_as_spam": 1,
  "revenue": 480.00,
  "delivery_rate": 0.991,
  "open_rate": 0.280,
  "click_rate": 0.056,
  "click_to_open_rate": 0.199,
  "conversion_rate": 0.255,
  "unsubscribe_rate": 0.004,
  "revenue_per_recipient": 0.56
}
```

**SMS Campaign Response**:
```json
{
  "recipients": 320,
  "delivered": 318,
  "failed": 2,
  "clicked": 28,
  "conversions": 6,
  "opt_outs": 1,
  "revenue": 240.00,
  "delivery_rate": 0.994,
  "click_rate": 0.088,
  "conversion_rate": 0.214,
  "opt_out_rate": 0.003,
  "revenue_per_recipient": 0.75
}
```

#### 5. Send Campaign

**Method**: `send_campaign(campaign_id, send_time)`

**Parameters**:
- `campaign_id` (str): Campaign identifier
- `send_time` (str, optional): ISO timestamp for scheduled sending

**Response**:
```json
{
  "success": true,
  "campaign_id": "EMAIL_ABC123DEF",
  "status": "SENT",
  "send_time": "2024-01-15T14:30:00Z",
  "message": "Email campaign sent successfully"
}
```

#### 6. List Management

**Method**: `create_list(list_data)`

**Request**:
```python
list_data = {
    'name': 'VIP Customers',
    'opt_in_process': 'DOUBLE_OPT_IN',
    'list_type': 'REGULAR'
}
```

**Response**:
```json
{
  "success": true,
  "list_id": "LIST_VIP12345",
  "name": "VIP Customers",
  "message": "List created successfully"
}
```

#### 7. Segmentation

**Method**: `create_segment(segment_data)`

**Request**:
```python
segment_data = {
    'name': 'High Value Customers',
    'definition': {
        'total_spent': {'greater_than': 500},
        'orders_count': {'greater_than': 3}
    }
}
```

**Response**:
```json
{
  "success": true,
  "segment_id": "SEG_HVC67890",
  "name": "High Value Customers",
  "estimated_size": 420,
  "message": "Segment created successfully"
}
```

#### 8. Subscriber Data

**Method**: `get_subscriber_data(filters)`

**Request**:
```python
filters = {
    'email_consent': True,
    'min_orders': 1
}
```

**Response**:
```json
{
  "total_subscribers": 3200,
  "email_subscribers": 2800,
  "sms_subscribers": 1200,
  "email_opt_in_rate": 0.875,
  "sms_opt_in_rate": 0.375,
  "subscribers": [...]  # First 100 subscribers
}
```

## Shopify Mock API

### Overview

Simulates Shopify Admin API for e-commerce store management.

#### Base Client

```python
from src.integrations.shopify_mock import ShopifyMock

client = ShopifyMock()
```

### Endpoints

#### 1. Store Information

**Method**: `get_store_info()`

**Response**:
```json
{
  "id": "shop_12345",
  "name": "Demo E-commerce Store",
  "domain": "demo-store.myshopify.com",
  "email": "owner@demo-store.com",
  "currency": "USD",
  "timezone": "America/New_York",
  "plan_name": "Shopify Plan",
  "created_at": "2023-01-15T10:30:00Z",
  "total_products": 100,
  "total_orders": 1000,
  "total_customers": 500,
  "active_products": 85
}
```

#### 2. Products Management

**Method**: `get_products(limit, status)`

**Parameters**:
- `limit` (int, optional): Maximum products to return (default: 50)
- `status` (str, optional): Filter by status ('active', 'draft', 'archived')

**Response**:
```json
{
  "products": [
    {
      "id": "PROD_A1B2C3D4",
      "title": "Premium T-Shirt",
      "handle": "premium-t-shirt",
      "product_type": "Clothing",
      "vendor": "Fashion Forward",
      "status": "active",
      "price": 29.99,
      "compare_at_price": 39.99,
      "inventory_quantity": 150,
      "sku": "SKU-00001",
      "weight": 0.2,
      "tags": ["clothing", "fashion-forward"],
      "created_at": "2023-06-15T14:20:00Z",
      "updated_at": "2024-01-10T09:15:00Z"
    }
  ],
  "count": 50,
  "total_available": 100
}
```

**Method**: `get_product(product_id)`

**Method**: `create_product(product_data)`

**Request**:
```python
product_data = {
    'title': 'New Product',
    'product_type': 'Electronics',
    'vendor': 'TechWorld',
    'price': 199.99,
    'inventory_quantity': 50,
    'sku': 'TECH-001'
}
```

**Method**: `update_product(product_id, updates)`

#### 3. Orders Management

**Method**: `get_orders(limit, status)`

**Response**:
```json
{
  "orders": [
    {
      "id": "ORDER_X9Y8Z7W6",
      "order_number": "#1001",
      "email": "customer@example.com",
      "customer_id": "CUST_12345678",
      "financial_status": "paid",
      "fulfillment_status": "fulfilled",
      "total_price": 89.97,
      "subtotal_price": 79.97,
      "total_tax": 7.20,
      "total_discounts": 0.00,
      "line_items_count": 3,
      "created_at": "2024-01-14T16:45:00Z",
      "updated_at": "2024-01-15T10:30:00Z"
    }
  ],
  "count": 50,
  "total_available": 1000
}
```

**Method**: `get_order(order_id)`

Returns detailed order with line items and shipping information.

#### 4. Customers Management

**Method**: `get_customers(limit)`

**Response**:
```json
{
  "customers": [
    {
      "id": "CUST_12345678",
      "email": "customer@example.com",
      "first_name": "John",
      "last_name": "Doe",
      "phone": "+1-555-123-4567",
      "accepts_marketing": true,
      "total_spent": 450.00,
      "orders_count": 5,
      "state": "enabled",
      "tags": ["VIP", "Newsletter"],
      "created_at": "2023-03-20T12:00:00Z",
      "updated_at": "2024-01-12T14:30:00Z"
    }
  ],
  "count": 50,
  "total_available": 500
}
```

**Method**: `get_customer(customer_id)`

Returns detailed customer with recent orders.

#### 5. Analytics

**Method**: `get_analytics(period)`

**Parameters**:
- `period` (str): Time period ('7d', '30d', '90d')

**Response**:
```json
{
  "period": "30d",
  "total_sales": 12450.00,
  "total_orders": 156,
  "average_order_value": 79.81,
  "unique_customers": 98,
  "conversion_rate": 15.60,
  "top_products": [
    {
      "product_id": "PROD_A1B2C3D4",
      "title": "Premium T-Shirt",
      "quantity_sold": 45,
      "revenue": 1349.55
    }
  ],
  "daily_breakdown": [
    {
      "date": "2024-01-01",
      "sales": 420.50,
      "orders": 6
    }
  ]
}
```

#### 6. Inventory Report

**Method**: `get_inventory_report()`

**Response**:
```json
{
  "total_products": 85,
  "total_inventory_value": 25420.50,
  "low_stock_count": 12,
  "out_of_stock_count": 3,
  "low_stock_products": [...],
  "out_of_stock_products": [...]
}
```

## Agent Core API

### Overview

Core agent functionality for autonomous marketing optimization.

#### Base Client

```python
from src.agent import EcommerceGrowthAgent

agent = EcommerceGrowthAgent()
```

### Methods

#### 1. Run Iteration

**Method**: `run_iteration()`

**Description**: Execute one complete plan→act→observe cycle

**Response**:
```json
{
  "iteration": 5,
  "duration": 12.3,
  "planning": {
    "objectives": ["optimize_budget", "improve_targeting"],
    "recommended_actions": [...]
  },
  "actions": {
    "actions_taken": [...],
    "optimizations": [...]
  },
  "observations": {
    "current_metrics": {...},
    "insights": [...]
  },
  "state": {
    "phase": "idle",
    "iteration": 5,
    "active_campaigns": ["CAMP_001", "CAMP_002"]
  }
}
```

#### 2. Get Agent Status

**Method**: `get_agent_status()`

**Response**:
```json
{
  "state": {
    "phase": "idle",
    "iteration": 5,
    "last_action_time": "2024-01-15T14:30:00Z",
    "active_campaigns": ["CAMP_001", "CAMP_002"],
    "performance_metrics": {...},
    "reasoning_chain": [...],
    "current_objectives": [...]
  },
  "performance_summary": {...},
  "active_campaigns_count": 2,
  "last_iteration_time": "2024-01-15T14:30:00Z",
  "total_iterations": 5
}
```

#### 3. Reset Agent

**Method**: `reset_agent()`

**Description**: Reset agent state for new session

## Data Pipeline API

### Overview

Customer segmentation and data analysis functionality.

#### Base Client

```python
from src.data_pipeline import DataPipeline

pipeline = DataPipeline()
```

### Methods

#### 1. Get Customer Segments

**Method**: `get_customer_segments()`

**Response**:
```json
{
  "segments": {
    "segment_0": {
      "segment_id": "segment_0",
      "name": "High Value Loyal",
      "size": 850,
      "characteristics": {
        "avg_clv": 650.0,
        "churn_rate": 0.12,
        "avg_frequency": 8.5
      },
      "avg_clv": 650.0,
      "churn_risk": 0.12,
      "recommended_channels": ["email", "google_ads"],
      "suggested_campaigns": ["retention", "upsell"]
    }
  },
  "total_customers": 10000,
  "segmentation_quality": 0.724
}
```

#### 2. Get Performance Metrics

**Method**: `get_performance_metrics()`

**Response**:
```json
{
  "overall_roas": 3.45,
  "total_spend": 15420.50,
  "total_revenue": 53200.75,
  "total_conversions": 342,
  "channel_performance": {
    "Google Ads": {
      "spend": 8500.00,
      "revenue": 31200.00,
      "roas": 3.67,
      "conversions": 185,
      "campaigns": 3
    }
  },
  "active_customers": 2450,
  "churned_customers": 380,
  "churn_rate": 0.134,
  "avg_clv": 425.50,
  "avg_order_value": 75.30,
  "total_customers": 10000
}
```

#### 3. Identify Opportunities

**Method**: `identify_opportunities()`

**Response**:
```json
[
  {
    "opportunity_id": "OPP_001",
    "opportunity_type": "segment_expansion",
    "description": "Expand high-value customer segment through lookalike targeting",
    "potential_revenue": 25000.0,
    "confidence_score": 0.85,
    "target_segments": ["High Value Loyal"],
    "recommended_actions": [
      {
        "action": "create_lookalike_campaign",
        "budget": 2000,
        "channel": "meta_ads"
      }
    ],
    "priority": "high"
  }
]
```

## Evaluation System API

### Overview

Performance evaluation, A/B testing, and anomaly detection.

#### Base Client

```python
from src.evaluation import EvaluationSystem

evaluator = EvaluationSystem()
```

### Methods

#### 1. Calculate Current Metrics

**Method**: `calculate_current_metrics()`

Returns comprehensive current performance metrics across all campaigns.

#### 2. Create A/B Test

**Method**: `create_ab_test(test_name, campaign_id, test_config)`

**Request**:
```python
test_config = {
    'duration_days': 14,
    'traffic_split': 0.5,
    'success_metric': 'conversion_rate',
    'minimum_sample_size': 1000,
    'confidence_level': 0.95,
    'variant_a': {'subject': 'Original Subject'},
    'variant_b': {'subject': 'New Subject'}
}
```

**Response**:
```json
{
  "test_id": "TEST_20240115143000",
  "test_name": "Email Subject Test",
  "status": "running",
  "start_date": "2024-01-15T14:30:00Z",
  "end_date": "2024-01-29T14:30:00Z"
}
```

#### 3. Analyze A/B Test

**Method**: `analyze_ab_test(test_id, variant_a_data, variant_b_data)`

**Response**:
```json
{
  "test_id": "TEST_20240115143000",
  "test_name": "Email Subject Test",
  "variant_a_performance": {
    "conversion_rate": 0.0425,
    "conversions": 85,
    "visitors": 2000
  },
  "variant_b_performance": {
    "conversion_rate": 0.0512,
    "conversions": 102,
    "visitors": 1995
  },
  "winner": "B",
  "confidence_level": 0.934,
  "statistical_significance": true,
  "lift_percentage": 20.5,
  "sample_size_a": 2000,
  "sample_size_b": 1995
}
```

#### 4. Detect Anomalies

**Method**: `detect_anomalies(current_metrics)`

**Response**:
```json
[
  {
    "metric_name": "overall_roas",
    "current_value": 1.2,
    "expected_range": [2.8, 4.2],
    "anomaly_score": 0.89,
    "is_anomaly": true,
    "severity": "high",
    "recommended_action": "immediate_investigation_required"
  }
]
```

## Error Handling

### Standard Error Response

```json
{
  "success": false,
  "error": {
    "code": "INVALID_CAMPAIGN_ID",
    "message": "Campaign not found",
    "details": "Campaign ID 'INVALID_123' does not exist"
  },
  "timestamp": "2024-01-15T14:30:00Z",
  "request_id": "req_error_123"
}
```

### Common Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `INVALID_CAMPAIGN_ID` | Campaign not found | 404 |
| `INSUFFICIENT_BUDGET` | Budget below minimum | 400 |
| `API_RATE_LIMIT` | Too many requests | 429 |
| `INVALID_PARAMETERS` | Invalid request parameters | 400 |
| `AUTHENTICATION_FAILED` | Invalid API key | 401 |
| `INTERNAL_ERROR` | Server error | 500 |

### Error Handling Example

```python
try:
    result = client.create_campaign(campaign_data)
    if result['success']:
        print(f"Campaign created: {result['platform_campaign_id']}")
    else:
        print(f"Error: {result['error']['message']}")
except Exception as e:
    print(f"Exception: {e}")
```

## Rate Limiting

### Mock Rate Limits

Mock APIs simulate real platform rate limits:

- **Google Ads**: 1000 requests/hour
- **Meta Ads**: 200 requests/hour  
- **Klaviyo**: 150 requests/10 minutes
- **Shopify**: 40 requests/second

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642248000
```

### Handling Rate Limits

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def create_campaign_with_retry(client, campaign_data):
    result = client.create_campaign(campaign_data)
    if not result['success'] and 'rate_limit' in result.get('error', {}):
        raise Exception("Rate limit exceeded")
    return result
```

## Examples

### Complete Campaign Workflow

```python
from src.integrations import get_platform_client
from src.agent import EcommerceGrowthAgent

# Initialize clients
google_client = get_platform_client('google_ads')
meta_client = get_platform_client('meta_ads')
agent = EcommerceGrowthAgent()

# Create multi-platform campaign
campaign_data = {
    'name': 'Holiday Sale 2024',
    'budget': 5000.0,
    'targeting': {
        'keywords': ['holiday sale', 'christmas deals'],
        'demographics': {'age_min': 25, 'age_max': 55}
    }
}

# Google Ads campaign
google_result = google_client.create_campaign(campaign_data)
if google_result['success']:
    google_campaign_id = google_result['platform_campaign_id']
    print(f"Google campaign: {google_campaign_id}")

# Meta Ads campaign
meta_result = meta_client.create_campaign(campaign_data)
if meta_result['success']:
    meta_campaign_id = meta_result['platform_campaign_id']
    print(f"Meta campaign: {meta_campaign_id}")

# Monitor performance
import time
time.sleep(60)  # Wait for initial data

google_metrics = google_client.get_campaign_metrics(google_campaign_id)
meta_metrics = meta_client.get_campaign_metrics(meta_campaign_id)

print(f"Google ROAS: {google_metrics['roas']:.2f}")
print(f"Meta ROAS: {meta_metrics['roas']:.2f}")

# Run agent optimization
agent_result = agent.run_iteration()
print(f"Agent optimizations: {len(agent_result['actions']['optimizations'])}")
```

### Customer Segmentation Analysis

```python
from src.data_pipeline import DataPipeline

pipeline = DataPipeline()

# Get customer segments
segments = pipeline.get_customer_segments()
print(f"Found {len(segments['segments'])} segments")

for segment_id, segment in segments['segments'].items():
    print(f"Segment: {segment['name']}")
    print(f"  Size: {segment['size']} customers")
    print(f"  Avg CLV: ${segment['avg_clv']:.2f}")
    print(f"  Recommended channels: {segment['recommended_channels']}")

# Identify opportunities
opportunities = pipeline.identify_opportunities()
for opp in opportunities:
    print(f"Opportunity: {opp.description}")
    print(f"  Potential revenue: ${opp.potential_revenue:,.2f}")
    print(f"  Confidence: {opp.confidence_score:.1%}")
```

### A/B Testing Workflow

```python
from src.evaluation import EvaluationSystem
from src.integrations import get_platform_client

evaluator = EvaluationSystem()
klaviyo_client = get_platform_client('klaviyo')

# Create A/B test
test_config = {
    'duration_days': 14,
    'success_metric': 'conversion_rate',
    'variant_a': {'subject': 'Don\'t Miss Out!'},
    'variant_b': {'subject': 'Last Chance to Save!'}
}

test_id = evaluator.create_ab_test(
    'Email Subject Test',
    'CAMP_EMAIL_001',
    test_config
)

print(f"Created A/B test: {test_id}")

# Simulate test results after campaign runs
variant_a_data = {
    'visitors': 5000,
    'conversions': 150
}

variant_b_data = {
    'visitors': 4950,
    'conversions': 180
}

# Analyze results
result = evaluator.analyze_ab_test(test_id, variant_a_data, variant_b_data)

print(f"Winner: Variant {result.winner}")
print(f"Lift: {result.lift_percentage:.1f}%")
print(f"Significant: {result.statistical_significance}")
print(f"Confidence: {result.confidence_level:.1%}")
```

## Support

For API questions and issues:

- **GitHub Issues**: [https://github.com/rushikeshdhumal/ecommerce-growth-agent/issues](https://github.com/rushikeshdhumal/ecommerce-growth-agent/issues)
- **Documentation**: See [README.md](../README.md) and [setup_guide.md](setup_guide.md)
- **Examples**: Check the `examples/` directory for more code samples

---

*This documentation covers the mock API implementations. For production use, replace mock clients with real platform integrations following the same interface patterns.*