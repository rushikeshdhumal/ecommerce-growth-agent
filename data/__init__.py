"""
Data package for E-commerce Growth Agent
Handles all data operations, database management, and sample data generation
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import json
import logging

# Package version and metadata
__version__ = "1.0.0"
__author__ = "E-commerce Growth Agent Team"

# Get logger
logger = logging.getLogger(__name__)

# Database configuration
DEFAULT_DB_PATH = "data/ecommerce_agent.db"
BACKUP_DIR = "data/backups"
EXPORT_DIR = "data/exports"

# Data schemas
CUSTOMER_SCHEMA = {
    'customer_id': 'TEXT PRIMARY KEY',
    'first_name': 'TEXT',
    'last_name': 'TEXT', 
    'email': 'TEXT',
    'registration_date': 'DATE',
    'last_purchase_date': 'DATE',
    'total_orders': 'INTEGER',
    'total_revenue': 'REAL',
    'avg_order_value': 'REAL',
    'days_since_last_purchase': 'INTEGER',
    'preferred_category': 'TEXT',
    'acquisition_channel': 'TEXT',
    'geographic_region': 'TEXT',
    'age_group': 'TEXT',
    'segment_id': 'TEXT'
}

TRANSACTION_SCHEMA = {
    'transaction_id': 'TEXT PRIMARY KEY',
    'customer_id': 'TEXT',
    'order_date': 'DATE',
    'order_value': 'REAL',
    'product_category': 'TEXT',
    'product_count': 'INTEGER',
    'discount_used': 'REAL',
    'channel': 'TEXT'
}

CAMPAIGN_SCHEMA = {
    'campaign_id': 'TEXT PRIMARY KEY',
    'campaign_name': 'TEXT',
    'channel': 'TEXT',
    'start_date': 'DATE',
    'end_date': 'DATE',
    'budget': 'REAL',
    'spend': 'REAL',
    'impressions': 'INTEGER',
    'clicks': 'INTEGER',
    'conversions': 'INTEGER',
    'revenue': 'REAL',
    'target_segment': 'TEXT'
}

# Package exports
__all__ = [
    # Core functions
    'initialize_database',
    'get_database_connection',
    'validate_database',
    'reset_database',
    
    # Data management
    'load_sample_data',
    'generate_mock_data',
    'export_data',
    'import_data',
    
    # Database utilities
    'backup_database',
    'restore_database',
    'get_database_stats',
    
    # Data validation
    'validate_data_integrity',
    'check_data_quality',
    
    # Constants
    'DEFAULT_DB_PATH',
    'CUSTOMER_SCHEMA',
    'TRANSACTION_SCHEMA',
    'CAMPAIGN_SCHEMA'
]


def initialize_database(db_path: str = DEFAULT_DB_PATH, force_recreate: bool = False) -> bool:
    """
    Initialize the SQLite database with all required tables
    
    Args:
        db_path: Path to the database file
        force_recreate: If True, recreate database even if it exists
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create data directory if it doesn't exist
        Path(db_path).parent.mkdir(exist_ok=True)
        
        # Remove existing database if force recreate
        if force_recreate and Path(db_path).exists():
            Path(db_path).unlink()
            logger.info(f"Removed existing database: {db_path}")
        
        # Create database connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create customers table
        customers_sql = f"""
            CREATE TABLE IF NOT EXISTS customers (
                {', '.join([f'{col} {dtype}' for col, dtype in CUSTOMER_SCHEMA.items()])}
            )
        """
        cursor.execute(customers_sql)
        
        # Create transactions table
        transactions_sql = f"""
            CREATE TABLE IF NOT EXISTS transactions (
                {', '.join([f'{col} {dtype}' for col, dtype in TRANSACTION_SCHEMA.items()])},
                FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
            )
        """
        cursor.execute(transactions_sql)
        
        # Create campaign performance table
        campaigns_sql = f"""
            CREATE TABLE IF NOT EXISTS campaign_performance (
                {', '.join([f'{col} {dtype}' for col, dtype in CAMPAIGN_SCHEMA.items()])}
            )
        """
        cursor.execute(campaigns_sql)
        
        # Create performance history table
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
        
        # Create A/B test results table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS ab_test_results (
                test_id TEXT PRIMARY KEY,
                test_name TEXT,
                campaign_id TEXT,
                start_date DATETIME,
                end_date DATETIME,
                variant_a_data TEXT,
                variant_b_data TEXT,
                winner TEXT,
                confidence_level REAL,
                statistical_significance BOOLEAN,
                lift_percentage REAL
            )
        """)
        
        # Create anomaly detection log
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS anomaly_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME,
                campaign_id TEXT,
                metric_name TEXT,
                current_value REAL,
                expected_value REAL,
                anomaly_score REAL,
                severity TEXT,
                action_taken TEXT
            )
        """)
        
        # Create indexes for better performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_customer_segment ON customers(segment_id)",
            "CREATE INDEX IF NOT EXISTS idx_customer_channel ON customers(acquisition_channel)",
            "CREATE INDEX IF NOT EXISTS idx_transaction_customer ON transactions(customer_id)",
            "CREATE INDEX IF NOT EXISTS idx_transaction_date ON transactions(order_date)",
            "CREATE INDEX IF NOT EXISTS idx_campaign_channel ON campaign_performance(channel)",
            "CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_history(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_performance_campaign ON performance_history(campaign_id)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        # Commit changes
        conn.commit()
        conn.close()
        
        logger.info(f"Database initialized successfully: {db_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def get_database_connection(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Get a database connection with proper configuration
    
    Args:
        db_path: Path to the database file
        
    Returns:
        SQLite connection object
    """
    try:
        conn = sqlite3.connect(db_path)
        
        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")
        
        # Set row factory for dict-like access
        conn.row_factory = sqlite3.Row
        
        return conn
        
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise


def validate_database(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Validate database structure and integrity
    
    Args:
        db_path: Path to the database file
        
    Returns:
        Validation results dictionary
    """
    validation_results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'table_counts': {},
        'missing_tables': [],
        'missing_indexes': []
    }
    
    try:
        if not Path(db_path).exists():
            validation_results['valid'] = False
            validation_results['errors'].append(f"Database file does not exist: {db_path}")
            return validation_results
        
        conn = get_database_connection(db_path)
        cursor = conn.cursor()
        
        # Check required tables exist
        required_tables = [
            'customers', 'transactions', 'campaign_performance', 
            'performance_history', 'ab_test_results', 'anomaly_log'
        ]
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        for table in required_tables:
            if table in existing_tables:
                # Count records in table
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                validation_results['table_counts'][table] = count
            else:
                validation_results['missing_tables'].append(table)
                validation_results['valid'] = False
        
        # Check for data quality issues
        if 'customers' in existing_tables:
            # Check for duplicate customer IDs
            cursor.execute("SELECT customer_id, COUNT(*) FROM customers GROUP BY customer_id HAVING COUNT(*) > 1")
            duplicates = cursor.fetchall()
            if duplicates:
                validation_results['warnings'].append(f"Found {len(duplicates)} duplicate customer IDs")
            
            # Check for invalid data
            cursor.execute("SELECT COUNT(*) FROM customers WHERE total_revenue < 0")
            negative_revenue = cursor.fetchone()[0]
            if negative_revenue > 0:
                validation_results['errors'].append(f"Found {negative_revenue} customers with negative revenue")
        
        conn.close()
        
    except Exception as e:
        validation_results['valid'] = False
        validation_results['errors'].append(f"Database validation error: {e}")
    
    return validation_results


def load_sample_data(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Load sample data from CSV files if they exist
    
    Args:
        db_path: Path to the database file
        
    Returns:
        Loading results dictionary
    """
    results = {
        'success': True,
        'loaded_files': [],
        'errors': [],
        'record_counts': {}
    }
    
    try:
        conn = get_database_connection(db_path)
        
        # Define CSV files and corresponding tables
        csv_files = {
            'data/sample_customers.csv': 'customers',
            'data/sample_transactions.csv': 'transactions', 
            'data/sample_campaigns.csv': 'campaign_performance'
        }
        
        for csv_file, table_name in csv_files.items():
            if Path(csv_file).exists():
                try:
                    df = pd.read_csv(csv_file)
                    df.to_sql(table_name, conn, if_exists='replace', index=False)
                    
                    results['loaded_files'].append(csv_file)
                    results['record_counts'][table_name] = len(df)
                    
                    logger.info(f"Loaded {len(df)} records from {csv_file}")
                    
                except Exception as e:
                    results['errors'].append(f"Error loading {csv_file}: {e}")
        
        conn.close()
        
        if results['errors']:
            results['success'] = False
        
    except Exception as e:
        results['success'] = False
        results['errors'].append(f"Database connection error: {e}")
    
    return results


def generate_mock_data(db_path: str = DEFAULT_DB_PATH, customer_count: int = 10000) -> Dict[str, Any]:
    """
    Generate realistic mock data for demonstration
    
    Args:
        db_path: Path to the database file
        customer_count: Number of customers to generate
        
    Returns:
        Generation results dictionary
    """
    results = {
        'success': True,
        'generated_counts': {},
        'errors': []
    }
    
    try:
        from faker import Faker
        fake = Faker()
        
        conn = get_database_connection(db_path)
        
        # Generate customers
        logger.info(f"Generating {customer_count} mock customers...")
        
        customers_data = []
        for i in range(customer_count):
            registration_date = fake.date_between(start_date='-2y', end_date='today')
            last_purchase = fake.date_between(
                start_date=registration_date,
                end_date='today'
            ) if np.random.random() > 0.15 else None  # 15% churn rate
            
            total_orders = np.random.poisson(lam=8) + 1
            avg_order_value = max(10, np.random.normal(75, 25))
            total_revenue = total_orders * avg_order_value
            
            days_since_last = (datetime.now().date() - last_purchase).days if last_purchase else 999
            
            customer = {
                'customer_id': f"CUST_{i:06d}",
                'first_name': fake.first_name(),
                'last_name': fake.last_name(),
                'email': fake.email(),
                'registration_date': registration_date.isoformat(),
                'last_purchase_date': last_purchase.isoformat() if last_purchase else None,
                'total_orders': total_orders,
                'total_revenue': round(total_revenue, 2),
                'avg_order_value': round(avg_order_value, 2),
                'days_since_last_purchase': days_since_last,
                'preferred_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports']),
                'acquisition_channel': np.random.choice(['Organic', 'Paid Search', 'Social', 'Email', 'Referral'],
                                                       p=[0.3, 0.25, 0.2, 0.15, 0.1]),
                'geographic_region': np.random.choice(['North', 'South', 'East', 'West']),
                'age_group': np.random.choice(['18-25', '26-35', '36-45', '46-55', '55+']),
                'segment_id': None  # Will be assigned during segmentation
            }
            customers_data.append(customer)
        
        # Insert customers into database
        df_customers = pd.DataFrame(customers_data)
        df_customers.to_sql('customers', conn, if_exists='replace', index=False)
        results['generated_counts']['customers'] = len(customers_data)
        
        # Generate transactions
        logger.info("Generating mock transactions...")
        
        transactions_data = []
        transaction_id = 0
        
        for customer in customers_data:
            if customer['last_purchase_date']:  # Only for active customers
                for order_num in range(customer['total_orders']):
                    transaction_date = fake.date_between(
                        start_date=datetime.fromisoformat(customer['registration_date']).date(),
                        end_date=datetime.fromisoformat(customer['last_purchase_date']).date()
                    )
                    
                    transaction = {
                        'transaction_id': f"TXN_{transaction_id:08d}",
                        'customer_id': customer['customer_id'],
                        'order_date': transaction_date.isoformat(),
                        'order_value': max(5, round(np.random.normal(customer['avg_order_value'], 15), 2)),
                        'product_category': customer['preferred_category'] if np.random.random() > 0.3
                                          else np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports']),
                        'product_count': np.random.randint(1, 6),
                        'discount_used': round(np.random.exponential(5), 2) if np.random.random() > 0.7 else 0,
                        'channel': np.random.choice(['Website', 'Mobile App', 'Marketplace'])
                    }
                    transactions_data.append(transaction)
                    transaction_id += 1
        
        # Insert transactions
        df_transactions = pd.DataFrame(transactions_data)
        df_transactions.to_sql('transactions', conn, if_exists='replace', index=False)
        results['generated_counts']['transactions'] = len(transactions_data)
        
        # Generate sample campaigns
        logger.info("Generating mock campaigns...")
        
        campaigns_data = []
        for i in range(50):  # 50 historical campaigns
            start_date = fake.date_between(start_date='-6m', end_date='today')
            end_date = start_date + timedelta(days=np.random.randint(7, 30))
            budget = np.random.uniform(500, 5000)
            spend = budget * np.random.uniform(0.8, 1.0)
            
            # Generate realistic metrics
            impressions = int(spend * np.random.uniform(20, 100))
            clicks = int(impressions * np.random.uniform(0.01, 0.05))
            conversions = int(clicks * np.random.uniform(0.02, 0.08))
            revenue = conversions * np.random.uniform(50, 200)
            
            campaign = {
                'campaign_id': f"CAMP_{i:04d}",
                'campaign_name': f"Campaign_{i}_{np.random.choice(['Acquisition', 'Retention', 'Winback'])}",
                'channel': np.random.choice(['Google Ads', 'Meta Ads', 'Email', 'Display']),
                'start_date': start_date.isoformat(),
                'end_date': end_date.isoformat(),
                'budget': round(budget, 2),
                'spend': round(spend, 2),
                'impressions': impressions,
                'clicks': clicks,
                'conversions': conversions,
                'revenue': round(revenue, 2),
                'target_segment': np.random.choice(['All', 'High Value', 'At Risk', 'New'])
            }
            campaigns_data.append(campaign)
        
        # Insert campaigns
        df_campaigns = pd.DataFrame(campaigns_data)
        df_campaigns.to_sql('campaign_performance', conn, if_exists='replace', index=False)
        results['generated_counts']['campaigns'] = len(campaigns_data)
        
        conn.close()
        
        logger.info(f"Mock data generation completed successfully")
        
    except Exception as e:
        results['success'] = False
        results['errors'].append(f"Mock data generation failed: {e}")
        logger.error(f"Mock data generation failed: {e}")
    
    return results


def export_data(db_path: str = DEFAULT_DB_PATH, export_format: str = 'csv') -> Dict[str, Any]:
    """
    Export database data to files
    
    Args:
        db_path: Path to the database file
        export_format: Export format ('csv', 'json', 'xlsx')
        
    Returns:
        Export results dictionary
    """
    results = {
        'success': True,
        'exported_files': [],
        'errors': []
    }
    
    try:
        # Create export directory
        Path(EXPORT_DIR).mkdir(exist_ok=True)
        
        conn = get_database_connection(db_path)
        
        # Tables to export
        tables = ['customers', 'transactions', 'campaign_performance', 'performance_history']
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for table in tables:
            try:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                
                if export_format == 'csv':
                    filename = f"{EXPORT_DIR}/{table}_{timestamp}.csv"
                    df.to_csv(filename, index=False)
                elif export_format == 'json':
                    filename = f"{EXPORT_DIR}/{table}_{timestamp}.json"
                    df.to_json(filename, orient='records', indent=2)
                elif export_format == 'xlsx':
                    filename = f"{EXPORT_DIR}/{table}_{timestamp}.xlsx"
                    df.to_excel(filename, index=False)
                else:
                    raise ValueError(f"Unsupported export format: {export_format}")
                
                results['exported_files'].append(filename)
                logger.info(f"Exported {len(df)} records from {table} to {filename}")
                
            except Exception as e:
                results['errors'].append(f"Error exporting {table}: {e}")
        
        conn.close()
        
        if results['errors']:
            results['success'] = False
        
    except Exception as e:
        results['success'] = False
        results['errors'].append(f"Export failed: {e}")
    
    return results


def backup_database(db_path: str = DEFAULT_DB_PATH, backup_name: str = None) -> Dict[str, Any]:
    """
    Create a backup of the database
    
    Args:
        db_path: Path to the database file
        backup_name: Custom backup name (optional)
        
    Returns:
        Backup results dictionary
    """
    results = {
        'success': True,
        'backup_file': None,
        'errors': []
    }
    
    try:
        # Create backup directory
        Path(BACKUP_DIR).mkdir(exist_ok=True)
        
        # Generate backup filename
        if not backup_name:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"ecommerce_agent_backup_{timestamp}.db"
        
        backup_path = f"{BACKUP_DIR}/{backup_name}"
        
        # Copy database file
        import shutil
        shutil.copy2(db_path, backup_path)
        
        results['backup_file'] = backup_path
        logger.info(f"Database backed up to: {backup_path}")
        
    except Exception as e:
        results['success'] = False
        results['errors'].append(f"Backup failed: {e}")
    
    return results


def get_database_stats(db_path: str = DEFAULT_DB_PATH) -> Dict[str, Any]:
    """
    Get comprehensive database statistics
    
    Args:
        db_path: Path to the database file
        
    Returns:
        Database statistics dictionary
    """
    stats = {
        'file_size_mb': 0,
        'table_counts': {},
        'date_ranges': {},
        'data_quality': {},
        'performance_metrics': {}
    }
    
    try:
        # Get file size
        if Path(db_path).exists():
            stats['file_size_mb'] = round(Path(db_path).stat().st_size / (1024 * 1024), 2)
        
        conn = get_database_connection(db_path)
        cursor = conn.cursor()
        
        # Get table counts
        tables = ['customers', 'transactions', 'campaign_performance', 'performance_history']
        for table in tables:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                stats['table_counts'][table] = cursor.fetchone()[0]
            except:
                stats['table_counts'][table] = 0
        
        # Get date ranges
        try:
            cursor.execute("SELECT MIN(registration_date), MAX(registration_date) FROM customers")
            result = cursor.fetchone()
            if result and result[0]:
                stats['date_ranges']['customers'] = {'min': result[0], 'max': result[1]}
        except:
            pass
        
        try:
            cursor.execute("SELECT MIN(order_date), MAX(order_date) FROM transactions")
            result = cursor.fetchone()
            if result and result[0]:
                stats['date_ranges']['transactions'] = {'min': result[0], 'max': result[1]}
        except:
            pass
        
        # Get data quality stats
        try:
            cursor.execute("SELECT COUNT(*) FROM customers WHERE total_revenue > 0")
            active_customers = cursor.fetchone()[0]
            stats['data_quality']['active_customers'] = active_customers
            
            cursor.execute("SELECT COUNT(*) FROM customers WHERE days_since_last_purchase > 90")
            churned_customers = cursor.fetchone()[0]
            stats['data_quality']['churned_customers'] = churned_customers
        except:
            pass
        
        conn.close()
        
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        stats['error'] = str(e)
    
    return stats


def reset_database(db_path: str = DEFAULT_DB_PATH, generate_data: bool = True) -> bool:
    """
    Reset database by dropping all data and optionally regenerating
    
    Args:
        db_path: Path to the database file
        generate_data: Whether to generate new mock data
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Backup existing database first
        backup_result = backup_database(db_path, f"pre_reset_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
        if backup_result['success']:
            logger.info(f"Created backup before reset: {backup_result['backup_file']}")
        
        # Initialize fresh database
        success = initialize_database(db_path, force_recreate=True)
        if not success:
            return False
        
        # Generate new data if requested
        if generate_data:
            from config.settings import settings
            data_size = getattr(settings, 'SAMPLE_DATA_SIZE', 10000)
            generate_result = generate_mock_data(db_path, data_size)
            if not generate_result['success']:
                logger.warning("Failed to generate mock data after reset")
                return False
        
        logger.info("Database reset completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database reset failed: {e}")
        return False


# Initialize database on package import if it doesn't exist
def _auto_initialize():
    """Auto-initialize database if it doesn't exist"""
    try:
        if not Path(DEFAULT_DB_PATH).exists():
            logger.info("Database not found, initializing...")
            initialize_database(DEFAULT_DB_PATH)
            
            # Generate sample data
            from config.settings import settings
            data_size = getattr(settings, 'SAMPLE_DATA_SIZE', 10000)
            generate_mock_data(DEFAULT_DB_PATH, data_size)
            
    except Exception as e:
        logger.warning(f"Auto-initialization failed: {e}")


# Auto-initialize on import
try:
    _auto_initialize()
except Exception as e:
    logger.warning(f"Failed to auto-initialize data package: {e}")

# Package metadata
version = __version__