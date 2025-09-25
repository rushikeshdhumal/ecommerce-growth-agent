# Setup Guide - E-commerce Growth Agent

Complete step-by-step guide for setting up the E-commerce Growth Agent on your system.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Configuration](#configuration)
4. [Database Setup](#database-setup)
5. [API Keys Configuration](#api-keys-configuration)
6. [Running the Application](#running-the-application)
7. [Verification](#verification)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Configuration](#advanced-configuration)
10. [Production Deployment](#production-deployment)

## System Requirements

### Minimum Requirements

- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **Python**: 3.9 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space
- **Internet**: Required for AI API calls and package downloads

### Recommended Requirements

- **Python**: 3.10 or 3.11
- **RAM**: 16GB for large datasets
- **Storage**: 10GB for full development setup
- **CPU**: Multi-core processor for better performance

### Software Dependencies

```bash
# Check Python version
python --version  # Should be 3.9+

# Check pip version
pip --version

# Git (for cloning repository)
git --version
```

## Installation Methods

### Method 1: Automated Setup (Recommended)

The easiest way to get started:

```bash
# 1. Clone the repository
git clone https://github.com/rushikeshdhumal/ecommerce-growth-agent.git
cd ecommerce-growth-agent

# 2. Run automated setup
python setup.py
```

The setup script will:
- ✅ Check Python version compatibility
- ✅ Create necessary directories
- ✅ Install all dependencies
- ✅ Initialize the database
- ✅ Generate sample data
- ✅ Create startup scripts

### Method 2: Manual Setup

For advanced users or custom configurations:

```bash
# 1. Clone repository
git clone https://github.com/rushikeshdhumal/ecommerce-growth-agent.git
cd ecommerce-growth-agent

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 4. Upgrade pip
pip install --upgrade pip

# 5. Install dependencies
pip install -r requirements.txt

# 6. Create environment file
cp .env.example .env

# 7. Initialize database
python -c "from data import initialize_database; initialize_database()"

# 8. Generate sample data
python -c "from data import generate_mock_data; generate_mock_data()"
```

### Method 3: Docker Setup

Using Docker for isolated deployment:

```bash
# 1. Clone repository
git clone https://github.com/rushikeshdhumal/ecommerce-growth-agent.git
cd ecommerce-growth-agent

# 2. Build Docker image
docker build -t ecommerce-growth-agent .

# 3. Run container
docker run -p 8501:8501 -v $(pwd)/data:/app/data ecommerce-growth-agent
```

## Configuration

### Environment Variables

Edit the `.env` file with your specific configuration:

```bash
# Core Settings
AGENT_MODEL=gpt-4                    # AI model to use
TEMPERATURE=0.7                      # Model creativity (0-1)
MAX_ITERATIONS=10                    # Max agent iterations
MAX_DAILY_BUDGET=1000.0             # Budget limit ($)

# Data Settings
SAMPLE_DATA_SIZE=10000              # Number of sample customers
CUSTOMER_SEGMENT_COUNT=5            # Customer segments

# Performance Thresholds
MIN_ROAS_THRESHOLD=2.0              # Minimum ROAS
MIN_CTR=0.01                        # Minimum CTR
MIN_CONVERSION_RATE=0.02            # Minimum conversion rate

# Logging
LOG_LEVEL=INFO                      # Logging level
LOG_FILE=logs/agent.log             # Log file path
```

### Configuration Validation

Test your configuration:

```bash
python -c "
from config import validate_config, get_config_summary
result = validate_config()
print('Valid:', result['valid'])
if result['errors']:
    print('Errors:', result['errors'])
if result['warnings']:
    print('Warnings:', result['warnings'])
print('Summary:', get_config_summary())
"
```

## Database Setup

### Automatic Setup

The setup script handles database initialization automatically. To manually initialize:

```bash
python -c "
from data import initialize_database, validate_database
success = initialize_database()
if success:
    validation = validate_database()
    print('Database valid:', validation['valid'])
    print('Table counts:', validation['table_counts'])
else:
    print('Database initialization failed')
"
```

### Database Structure

The system creates these tables:
- **customers**: Customer profiles and segments
- **transactions**: Purchase history
- **campaign_performance**: Campaign metrics
- **performance_history**: Time-series performance data
- **ab_test_results**: A/B test outcomes
- **anomaly_log**: Performance anomalies

### Sample Data Generation

Generate realistic sample data:

```bash
python -c "
from data import generate_mock_data
result = generate_mock_data(customer_count=5000)
if result['success']:
    print('Generated:')
    for table, count in result['generated_counts'].items():
        print(f'  {table}: {count:,}')
else:
    print('Errors:', result['errors'])
"
```

### Database Management

Common database operations:

```bash
# View database statistics
python -c "
from data import get_database_stats
stats = get_database_stats()
print(f'Database size: {stats[\"file_size_mb\"]} MB')
print('Table counts:', stats['table_counts'])
"

# Backup database
python -c "
from data import backup_database
result = backup_database()
if result['success']:
    print('Backup created:', result['backup_file'])
"

# Reset database
python -c "
from data import reset_database
success = reset_database(generate_data=True)
print('Reset successful:', success)
"
```

## API Keys Configuration

### OpenAI API Setup

1. **Create OpenAI Account**
   - Go to [OpenAI Platform](https://platform.openai.com/)
   - Create account or sign in
   - Navigate to API Keys section

2. **Generate API Key**
   - Click "Create new secret key"
   - Copy the key (starts with `sk-`)
   - Add to `.env` file:
     ```bash
     OPENAI_API_KEY=sk-your-key-here
     ```

3. **Verify API Key**
   ```bash
   python -c "
   import openai
   from config import settings
   openai.api_key = settings.OPENAI_API_KEY
   try:
       # Test API call
       response = openai.ChatCompletion.create(
           model='gpt-3.5-turbo',
           messages=[{'role': 'user', 'content': 'Hello'}],
           max_tokens=10
       )
       print('OpenAI API: Working')
   except Exception as e:
       print('OpenAI API Error:', e)
   "
   ```

### Anthropic API Setup

1. **Create Anthropic Account**
   - Go to [Anthropic Console](https://console.anthropic.com/)
   - Create account or sign in
   - Navigate to API Keys

2. **Generate API Key**
   - Create new API key
   - Copy the key (starts with `ant-`)
   - Add to `.env` file:
     ```bash
     ANTHROPIC_API_KEY=ant-your-key-here
     ```

3. **Verify API Key**
   ```bash
   python -c "
   import anthropic
   from config import settings
   client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
   try:
       response = client.messages.create(
           model='claude-3-sonnet-20240229',
           max_tokens=10,
           messages=[{'role': 'user', 'content': 'Hello'}]
       )
       print('Anthropic API: Working')
   except Exception as e:
       print('Anthropic API Error:', e)
   "
   ```

### API Key Security

**Important Security Practices:**

1. **Never commit API keys to version control**
2. **Use environment variables only**
3. **Rotate keys regularly**
4. **Set usage limits in API dashboards**
5. **Monitor usage and costs**

## Running the Application

### Quick Start

```bash
# Option 1: Use startup script
./start.sh          # Linux/macOS
start.bat           # Windows

# Option 2: Direct command
streamlit run streamlit_app.py

# Option 3: Custom port
streamlit run streamlit_app.py --server.port 8502
```

### Development Mode

For development with auto-reload:

```bash
# Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with auto-reload
streamlit run streamlit_app.py --server.runOnSave true
```

### Background Execution

Run as background service:

```bash
# Using nohup (Linux/macOS)
nohup streamlit run streamlit_app.py > app.log 2>&1 &

# Using screen
screen -S ecommerce-agent
streamlit run streamlit_app.py
# Press Ctrl+A, then D to detach

# Using systemd (Linux)
sudo cp ecommerce-agent.service /etc/systemd/system/
sudo systemctl enable ecommerce-agent
sudo systemctl start ecommerce-agent
```

## Verification

### System Health Check

Run comprehensive system verification:

```bash
python -c "
from src import get_system_status, validate_system
print('=== System Status ===')
status = get_system_status()
print(f'Version: {status[\"version\"]}')
print(f'Core Ready: {status[\"core_components_ready\"]}')
print(f'System Ready: {status[\"system_ready\"]}')

print('\n=== Component Status ===')
for component, available in status['components_available'].items():
    status_icon = '✅' if available else '❌'
    print(f'{status_icon} {component}')

print('\n=== Validation ===')
validation = validate_system()
print(f'Valid: {validation[\"valid\"]}')
if validation['errors']:
    print('Errors:', validation['errors'])
if validation['warnings']:
    print('Warnings:', validation['warnings'])
"
```

### Agent Test

Test agent functionality:

```bash
python -c "
from src import create_agent_instance
print('Creating agent instance...')
agent = create_agent_instance()
if agent:
    print('✅ Agent created successfully')
    status = agent.get_agent_status()
    print(f'Agent phase: {status[\"state\"][\"phase\"]}')
    print(f'Iterations: {status[\"total_iterations\"]}')
else:
    print('❌ Failed to create agent')
"
```

### Web Interface Test

1. **Access Application**
   - Open browser to http://localhost:8501
   - Should see E-commerce Growth Agent dashboard

2. **Test Navigation**
   - Navigate through all pages
   - Check for errors in browser console
   - Verify data loads properly

3. **Test Agent Functionality**
   - Go to "Agent Control Panel"
   - Click "Run Single Iteration"
   - Verify agent executes successfully

## Troubleshooting

### Common Issues

#### 1. Python Version Issues

```bash
# Problem: Python version too old
# Solution: Update Python
python --version  # Check current version

# Install Python 3.9+ from python.org
# Or use pyenv for version management
pyenv install 3.10.12
pyenv local 3.10.12
```

#### 2. Package Installation Failures

```bash
# Problem: Package installation fails
# Solutions:

# Update pip
pip install --upgrade pip

# Clear cache
pip cache purge

# Install with no cache
pip install --no-cache-dir -r requirements.txt

# Use conda instead
conda create -n ecommerce-agent python=3.10
conda activate ecommerce-agent
pip install -r requirements.txt
```

#### 3. Database Issues

```bash
# Problem: Database errors
# Solutions:

# Reset database
rm data/ecommerce_agent.db
python -c "from data import initialize_database; initialize_database()"

# Check permissions
ls -la data/
chmod 755 data/
chmod 644 data/*.db

# Check disk space
df -h
```

#### 4. Port Already in Use

```bash
# Problem: Port 8501 already in use
# Solutions:

# Find process using port
lsof -i :8501  # macOS/Linux
netstat -ano | findstr 8501  # Windows

# Kill process
kill -9 <PID>  # macOS/Linux

# Use different port
streamlit run streamlit_app.py --server.port 8502
```

#### 5. API Key Issues

```bash
# Problem: API key not working
# Solutions:

# Check .env file exists and has correct format
cat .env | grep API_KEY

# Verify no extra spaces or quotes
# Correct: OPENAI_API_KEY=sk-abc123
# Wrong: OPENAI_API_KEY="sk-abc123"
# Wrong: OPENAI_API_KEY = sk-abc123

# Test API key directly
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('OPENAI_API_KEY')
print('Key length:', len(key) if key else 'None')
print('Key prefix:', key[:10] if key else 'None')
"
```

#### 6. Memory Issues

```bash
# Problem: Out of memory
# Solutions:

# Reduce sample data size
export SAMPLE_DATA_SIZE=1000

# Monitor memory usage
python -c "
import psutil
memory = psutil.virtual_memory()
print(f'Available: {memory.available / 1024**3:.1f} GB')
print(f'Used: {memory.percent}%')
"

# Use smaller model
export AGENT_MODEL=gpt-3.5-turbo
```

### Performance Optimization

#### 1. Startup Time

```bash
# Skip data generation on startup
export SKIP_DATA_GENERATION=true

# Use smaller dataset
export SAMPLE_DATA_SIZE=1000

# Enable caching
export ENABLE_CACHING=true
```

#### 2. Response Time

```bash
# Use faster model
export AGENT_MODEL=gpt-3.5-turbo
export TEMPERATURE=0.5

# Reduce max iterations
export MAX_ITERATIONS=5
```

#### 3. Memory Usage

```bash
# Optimize Streamlit
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=10
export STREAMLIT_SERVER_MAX_MESSAGE_SIZE=10

# Clear logs regularly
find logs/ -name "*.log" -mtime +7 -delete
```

### Log Analysis

Check logs for issues:

```bash
# View recent logs
tail -f logs/agent.log

# Search for errors
grep -i error logs/agent.log
grep -i warning logs/agent.log

# Analyze log patterns
python -c "
import re
with open('logs/agent.log', 'r') as f:
    content = f.read()
    errors = re.findall(r'ERROR.*', content)
    warnings = re.findall(r'WARNING.*', content)
    print(f'Errors: {len(errors)}')
    print(f'Warnings: {len(warnings)}')
    if errors:
        print('Recent errors:')
        for error in errors[-5:]:
            print(f'  {error}')
"
```

## Advanced Configuration

### Custom Models

Configure custom AI models:

```bash
# In .env file
AGENT_MODEL=custom-model
CUSTOM_LLM_ENDPOINT=http://localhost:8000/v1/chat/completions
CUSTOM_LLM_API_KEY=your-custom-key

# Test custom endpoint
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-custom-key" \
  -d '{"messages":[{"role":"user","content":"Hello"}]}'
```

### Database Configuration

Use external database:

```bash
# PostgreSQL
export DATABASE_URL=postgresql://user:pass@localhost/ecommerce_agent

# MySQL
export DATABASE_URL=mysql://user:pass@localhost/ecommerce_agent

# Install database drivers
pip install psycopg2-binary  # PostgreSQL
pip install mysql-connector-python  # MySQL
```

### Performance Tuning

```python
# config/settings.py - Add custom settings
PERFORMANCE_SETTINGS = {
    'enable_multiprocessing': True,
    'worker_processes': 4,
    'cache_size': 1000,
    'batch_size': 100
}

# Enable parallel processing
ENABLE_PARALLEL_PROCESSING = True
MAX_WORKERS = 4
```

### Feature Flags

Control feature availability:

```bash
# In .env file
ENABLE_AB_TESTING=true
ENABLE_ADVANCED_SEGMENTATION=true
ENABLE_PREDICTIVE_ANALYTICS=false
ENABLE_REAL_TIME_OPTIMIZATION=true
```

### Monitoring Setup

Configure monitoring and alerting:

```bash
# Monitoring settings
ENABLE_MONITORING=true
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Error tracking
SENTRY_DSN=your-sentry-dsn-here

# Log aggregation
ELASTICSEARCH_URL=http://localhost:9200
KIBANA_URL=http://localhost:5601
```

## Production Deployment

### Environment Setup

1. **Server Requirements**
   - Ubuntu 20.04+ or CentOS 8+
   - 8GB+ RAM
   - 50GB+ storage
   - SSL certificate

2. **Security Configuration**
   ```bash
   # Firewall rules
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw allow 22/tcp
   sudo ufw enable
   
   # SSL setup with Let's Encrypt
   sudo apt install certbot nginx
   sudo certbot --nginx -d yourdomain.com
   ```

3. **Database Setup**
   ```bash
   # PostgreSQL installation
   sudo apt install postgresql postgresql-contrib
   sudo -u postgres createdb ecommerce_agent
   sudo -u postgres createuser --interactive
   
   # Update DATABASE_URL
   export DATABASE_URL=postgresql://username:password@localhost/ecommerce_agent
   ```

### Docker Deployment

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=postgresql://postgres:password@db:5432/ecommerce_agent
    depends_on:
      - db
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: ecommerce_agent
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - app

volumes:
  postgres_data:
```

### Cloud Deployment

#### AWS Deployment

```bash
# Using AWS CLI
aws configure

# Deploy to EC2
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-groups your-security-group

# Deploy to ECS
aws ecs create-cluster --cluster-name ecommerce-agent
```

#### Google Cloud Deployment

```bash
# Using Cloud Run
gcloud run deploy ecommerce-agent \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Monitoring and Logging

```bash
# Set up log rotation
sudo tee /etc/logrotate.d/ecommerce-agent << EOF
/path/to/logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    create 0644 app app
}
EOF

# Set up health checks
curl -f http://localhost:8501/_stcore/health || exit 1
```

### Backup Strategy

```bash
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Database backup
pg_dump ecommerce_agent > $BACKUP_DIR/db_backup_$DATE.sql

# Code backup
tar -czf $BACKUP_DIR/code_backup_$DATE.tar.gz /app

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "*.sql" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

---

## Need Help?

- **Documentation**: Check the [README.md](../README.md) for overview
- **Architecture**: See [architecture.md](architecture.md) for system details
- **Issues**: Report problems on [GitHub Issues](https://github.com/rushikeshdhumal/ecommerce-growth-agent/issues)
- **Community**: Join discussions on [GitHub Discussions](https://github.com/rushikeshdhumal/ecommerce-growth-agent/discussions)

For immediate help, run the built-in diagnostics:

```bash
python -c "
from src import quick_demo
result = quick_demo()
print('Demo successful:', result['success'])
if not result['success']:
    print('Errors:', result['errors'])
"
```