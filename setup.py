"""
Setup script for E-commerce Growth Agent
Automated initialization and configuration
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
import sqlite3
import json


def print_header():
    """Print setup header"""
    print("=" * 60)
    print("🚀 E-commerce Growth Agent Setup")
    print("=" * 60)
    print("Initializing your autonomous marketing system...")
    print()


def check_python_version():
    """Check Python version compatibility"""
    print("📋 Checking Python version...")
    
    if sys.version_info < (3, 9):
        print("❌ Error: Python 3.9 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    print()


def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = [
        'data',
        'logs', 
        'exports',
        'cache',
        'config',
        'tests',
        'docs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created: {directory}/")
    
    print()


def setup_environment_file():
    """Setup environment configuration file"""
    print("⚙️ Setting up environment configuration...")
    
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        print("✅ Created .env file from template")
        print("⚠️  Please edit .env file with your API keys")
    elif env_file.exists():
        print("✅ .env file already exists")
    else:
        # Create basic .env file
        with open(env_file, 'w') as f:
            f.write("""# E-commerce Growth Agent Configuration
AGENT_MODEL=gpt-4
TEMPERATURE=0.7
MAX_ITERATIONS=10
MAX_DAILY_BUDGET=1000.0
MIN_ROAS_THRESHOLD=2.0
SAMPLE_DATA_SIZE=10000
LOG_LEVEL=INFO

# Add your API keys here:
# OPENAI_API_KEY=your_openai_api_key_here
# ANTHROPIC_API_KEY=your_anthropic_api_key_here
""")
        print("✅ Created basic .env file")
        print("⚠️  Please add your API keys to .env file")
    
    print()


def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing dependencies...")
    
    try:
        # Check if requirements.txt exists
        if not Path('requirements.txt').exists():
            print("❌ requirements.txt not found")
            return False
        
        # Install requirements
        result = subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
        else:
            print("❌ Failed to install dependencies")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    
    print()
    return True


def initialize_database():
    """Initialize SQLite database"""
    print("🗄️ Initializing database...")
    
    try:
        # Create data directory
        Path('data').mkdir(exist_ok=True)
        
        # Initialize database
        db_path = 'data/ecommerce_agent.db'
        conn = sqlite3.connect(db_path)
        
        # Create basic tables
        cursor = conn.cursor()
        
        # Customers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id TEXT PRIMARY KEY,
                first_name TEXT,
                last_name TEXT,
                email TEXT,
                registration_date DATE,
                last_purchase_date DATE,
                total_orders INTEGER,
                total_revenue REAL,
                avg_order_value REAL,
                days_since_last_purchase INTEGER,
                preferred_category TEXT,
                acquisition_channel TEXT,
                geographic_region TEXT,
                age_group TEXT,
                segment_id TEXT
            )
        """)
        
        # Campaign performance table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS campaign_performance (
                campaign_id TEXT PRIMARY KEY,
                campaign_name TEXT,
                channel TEXT,
                start_date DATE,
                end_date DATE,
                budget REAL,
                spend REAL,
                impressions INTEGER,
                clicks INTEGER,
                conversions INTEGER,
                revenue REAL,
                target_segment TEXT
            )
        """)
        
        # Performance history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                campaign_id TEXT,
                channel TEXT,
                metric_name TEXT,
                metric_value REAL,
                benchmark_value REAL,
                context TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
        print("✅ Database initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        return False
    
    print()
    return True


def generate_sample_data():
    """Generate sample data for demonstration"""
    print("📊 Generating sample data...")
    
    try:
        # Import and run data generation
        from src.data_pipeline import DataPipeline
        
        # Initialize data pipeline (will generate sample data)
        pipeline = DataPipeline()
        
        print("✅ Sample data generated successfully")
        print(f"   - Customers: {len(pipeline.customers_df):,}")
        print(f"   - Transactions: {len(pipeline.transactions_df):,}")
        print(f"   - Campaigns: {len(pipeline.campaigns_df):,}")
        
        # Close connection
        pipeline.close_connection()
        
    except Exception as e:
        print(f"❌ Error generating sample data: {e}")
        print("   Sample data will be generated on first run")
        return False
    
    print()
    return True


def verify_installation():
    """Verify installation by importing main components"""
    print("🔍 Verifying installation...")
    
    try:
        # Test imports
        from config.settings import settings
        from config.logging_config import setup_logging
        from src.agent import EcommerceGrowthAgent
        from src.data_pipeline import DataPipeline
        from src.campaign_manager import CampaignManager
        from src.evaluation import EvaluationSystem
        
        print("✅ All core components imported successfully")
        
        # Test basic functionality
        setup_logging()
        print("✅ Logging system initialized")
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False
    
    print()
    return True


def create_startup_script():
    """Create startup script for easy launching"""
    print("🚀 Creating startup script...")
    
    startup_script = """#!/bin/bash
# E-commerce Growth Agent Startup Script

echo "🚀 Starting E-commerce Growth Agent..."
echo "Opening browser at http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo ""

streamlit run streamlit_app.py
"""
    
    try:
        with open('start.sh', 'w') as f:
            f.write(startup_script)
        
        # Make executable on Unix systems
        if os.name != 'nt':
            os.chmod('start.sh', 0o755)
        
        print("✅ Created startup script: start.sh")
        
        # Create Windows batch file
        windows_script = """@echo off
echo 🚀 Starting E-commerce Growth Agent...
echo Opening browser at http://localhost:8501
echo Press Ctrl+C to stop the application
echo.

streamlit run streamlit_app.py
pause
"""
        
        with open('start.bat', 'w') as f:
            f.write(windows_script)
        
        print("✅ Created Windows startup script: start.bat")
        
    except Exception as e:
        print(f"❌ Error creating startup script: {e}")
        return False
    
    print()
    return True


def print_next_steps():
    """Print next steps for the user"""
    print("🎉 Setup Complete!")
    print("=" * 60)
    print()
    print("Next Steps:")
    print("1. 📝 Edit .env file with your API keys:")
    print("   - OpenAI API key (for GPT models)")
    print("   - Anthropic API key (for Claude models)")
    print()
    print("2. 🚀 Start the application:")
    print("   Linux/Mac:  ./start.sh")
    print("   Windows:    start.bat")
    print("   Manual:     streamlit run streamlit_app.py")
    print()
    print("3. 🌐 Open your browser to:")
    print("   http://localhost:8501")
    print()
    print("4. 📖 Check the documentation:")
    print("   README.md - Full documentation")
    print("   docs/ - Additional guides")
    print()
    print("🔧 Configuration Files:")
    print("   .env - Environment variables")
    print("   config/settings.py - Application settings")
    print()
    print("🧪 Testing:")
    print("   Run tests: pytest tests/")
    print("   Coverage:  pytest --cov=src tests/")
    print()
    print("💡 Need help?")
    print("   - Check README.md for detailed instructions")
    print("   - Visit GitHub issues for support")
    print("   - Review example configurations in docs/")
    print()
    print("=" * 60)
    print("Happy automating! 🤖✨")


def main():
    """Main setup function"""
    print_header()
    
    # Run setup steps
    steps = [
        ("Checking Python version", check_python_version),
        ("Creating directories", create_directories),
        ("Setting up environment", setup_environment_file),
        ("Installing dependencies", install_dependencies),
        ("Initializing database", initialize_database),
        ("Generating sample data", generate_sample_data),
        ("Verifying installation", verify_installation),
        ("Creating startup scripts", create_startup_script)
    ]
    
    success_count = 0
    
    for step_name, step_function in steps:
        try:
            result = step_function()
            if result is not False:
                success_count += 1
        except Exception as e:
            print(f"❌ Error in {step_name}: {e}")
    
    print(f"Setup completed: {success_count}/{len(steps)} steps successful")
    print()
    
    if success_count == len(steps):
        print_next_steps()
    else:
        print("⚠️ Some setup steps failed. Please check the errors above.")
        print("You may need to complete the setup manually.")


if __name__ == "__main__":
    main()