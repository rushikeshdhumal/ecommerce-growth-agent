"""
Mock Shopify Admin API integration for e-commerce data and store management
"""
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import numpy as np

from config.logging_config import get_agent_logger


@dataclass
class ShopifyStore:
    """Shopify store structure"""
    store_id: str
    name: str
    domain: str
    email: str
    currency: str
    timezone: str
    plan: str
    created_at: datetime


@dataclass
class ShopifyProduct:
    """Shopify product structure"""
    product_id: str
    title: str
    handle: str
    product_type: str
    vendor: str
    status: str
    price: float
    compare_at_price: Optional[float]
    inventory_quantity: int
    sku: str
    weight: float
    tags: List[str]
    created_at: datetime
    updated_at: datetime


@dataclass
class ShopifyOrder:
    """Shopify order structure"""
    order_id: str
    order_number: str
    email: str
    customer_id: Optional[str]
    financial_status: str
    fulfillment_status: str
    total_price: float
    subtotal_price: float
    total_tax: float
    total_discounts: float
    line_items: List[Dict[str, Any]]
    shipping_address: Dict[str, str]
    created_at: datetime
    updated_at: datetime


@dataclass
class ShopifyCustomer:
    """Shopify customer structure"""
    customer_id: str
    email: str
    first_name: str
    last_name: str
    phone: Optional[str]
    accepts_marketing: bool
    total_spent: float
    orders_count: int
    state: str
    tags: List[str]
    created_at: datetime
    updated_at: datetime


class ShopifyMock:
    """
    Mock Shopify Admin API client for e-commerce store management
    """
    
    def __init__(self):
        self.logger = get_agent_logger("ShopifyMock")
        
        # Mock store data
        self.store = ShopifyStore(
            store_id="shop_12345",
            name="Demo E-commerce Store",
            domain="demo-store.myshopify.com",
            email="owner@demo-store.com",
            currency="USD",
            timezone="America/New_York",
            plan="Shopify Plan",
            created_at=datetime.now() - timedelta(days=365)
        )
        
        # Data storage
        self.products: Dict[str, ShopifyProduct] = {}
        self.orders: Dict[str, ShopifyOrder] = {}
        self.customers: Dict[str, ShopifyCustomer] = {}
        
        # Analytics data
        self.analytics_data = {}
        
        # Generate initial mock data
        self._generate_mock_data()
        
        self.logger.log_action("shopify_mock_initialized", {
            "store_name": self.store.name,
            "products_count": len(self.products),
            "orders_count": len(self.orders),
            "customers_count": len(self.customers)
        })
    
    def _generate_mock_data(self):
        """Generate realistic mock store data"""
        # Generate customers
        self._generate_mock_customers(500)
        
        # Generate products
        self._generate_mock_products(100)
        
        # Generate orders
        self._generate_mock_orders(1000)
    
    def _generate_mock_customers(self, count: int):
        """Generate mock customers"""
        from faker import Faker
        fake = Faker()
        
        for i in range(count):
            customer_id = f"CUST_{uuid.uuid4().hex[:8].upper()}"
            
            orders_count = np.random.poisson(3)
            total_spent = orders_count * np.random.uniform(25, 200)
            
            customer = ShopifyCustomer(
                customer_id=customer_id,
                email=fake.email(),
                first_name=fake.first_name(),
                last_name=fake.last_name(),
                phone=fake.phone_number() if random.random() > 0.3 else None,
                accepts_marketing=random.choice([True, False]),
                total_spent=round(total_spent, 2),
                orders_count=orders_count,
                state=random.choice(['enabled', 'disabled']),
                tags=random.sample(['VIP', 'Returning', 'Newsletter', 'Discount'], k=random.randint(0, 2)),
                created_at=fake.date_time_between(start_date='-2y', end_date='now'),
                updated_at=fake.date_time_between(start_date='-1m', end_date='now')
            )
            
            self.customers[customer_id] = customer
    
    def _generate_mock_products(self, count: int):
        """Generate mock products"""
        from faker import Faker
        fake = Faker()
        
        product_categories = [
            'Electronics', 'Clothing', 'Home & Garden', 'Sports & Outdoors',
            'Books', 'Health & Beauty', 'Toys & Games', 'Automotive'
        ]
        
        vendors = [
            'Demo Electronics Co', 'Fashion Forward', 'Home Essentials',
            'ActiveGear', 'Beauty Plus', 'TechWorld', 'StyleHouse'
        ]
        
        for i in range(count):
            product_id = f"PROD_{uuid.uuid4().hex[:8].upper()}"
            category = random.choice(product_categories)
            vendor = random.choice(vendors)
            
            title = fake.catch_phrase()
            price = round(random.uniform(9.99, 299.99), 2)
            compare_price = round(price * random.uniform(1.1, 1.5), 2) if random.random() > 0.7 else None
            
            product = ShopifyProduct(
                product_id=product_id,
                title=title,
                handle=title.lower().replace(' ', '-').replace(',', ''),
                product_type=category,
                vendor=vendor,
                status=random.choice(['active', 'draft', 'archived']),
                price=price,
                compare_at_price=compare_price,
                inventory_quantity=random.randint(0, 500),
                sku=f"SKU-{i:05d}",
                weight=round(random.uniform(0.1, 5.0), 2),
                tags=[category.lower(), vendor.lower().replace(' ', '-')],
                created_at=fake.date_time_between(start_date='-1y', end_date='now'),
                updated_at=fake.date_time_between(start_date='-1m', end_date='now')
            )
            
            self.products[product_id] = product
    
    def _generate_mock_orders(self, count: int):
        """Generate mock orders"""
        from faker import Faker
        fake = Faker()
        
        order_statuses = [
            ('paid', 'fulfilled'),
            ('paid', 'partial'),
            ('pending', 'unfulfilled'),
            ('refunded', 'fulfilled')
        ]
        
        for i in range(count):
            order_id = f"ORDER_{uuid.uuid4().hex[:8].upper()}"
            order_number = f"#{1000 + i}"
            
            # Random customer or guest
            customer_id = random.choice(list(self.customers.keys())) if random.random() > 0.3 else None
            customer = self.customers.get(customer_id) if customer_id else None
            email = customer.email if customer else fake.email()
            
            # Generate line items
            num_items = random.randint(1, 5)
            line_items = []
            subtotal = 0
            
            available_products = [p for p in self.products.values() if p.status == 'active']
            selected_products = random.sample(available_products, min(num_items, len(available_products)))
            
            for product in selected_products:
                quantity = random.randint(1, 3)
                line_total = product.price * quantity
                subtotal += line_total
                
                line_items.append({
                    'product_id': product.product_id,
                    'title': product.title,
                    'quantity': quantity,
                    'price': product.price,
                    'total': line_total,
                    'sku': product.sku
                })
            
            # Calculate totals
            discount = subtotal * random.uniform(0, 0.2) if random.random() > 0.8 else 0
            tax_rate = random.uniform(0.05, 0.12)
            tax = (subtotal - discount) * tax_rate
            total = subtotal - discount + tax
            
            financial_status, fulfillment_status = random.choice(order_statuses)
            
            order = ShopifyOrder(
                order_id=order_id,
                order_number=order_number,
                email=email,
                customer_id=customer_id,
                financial_status=financial_status,
                fulfillment_status=fulfillment_status,
                total_price=round(total, 2),
                subtotal_price=round(subtotal, 2),
                total_tax=round(tax, 2),
                total_discounts=round(discount, 2),
                line_items=line_items,
                shipping_address={
                    'first_name': fake.first_name(),
                    'last_name': fake.last_name(),
                    'address1': fake.street_address(),
                    'city': fake.city(),
                    'province': fake.state(),
                    'country': 'United States',
                    'zip': fake.zipcode()
                },
                created_at=fake.date_time_between(start_date='-6m', end_date='now'),
                updated_at=fake.date_time_between(start_date='-1m', end_date='now')
            )
            
            self.orders[order_id] = order
    
    def get_store_info(self) -> Dict[str, Any]:
        """Get store information"""
        return {
            'id': self.store.store_id,
            'name': self.store.name,
            'domain': self.store.domain,
            'email': self.store.email,
            'currency': self.store.currency,
            'timezone': self.store.timezone,
            'plan_name': self.store.plan,
            'created_at': self.store.created_at.isoformat(),
            'total_products': len(self.products),
            'total_orders': len(self.orders),
            'total_customers': len(self.customers),
            'active_products': len([p for p in self.products.values() if p.status == 'active'])
        }
    
    def get_products(self, limit: int = 50, status: str = None) -> Dict[str, Any]:
        """Get products list"""
        products_list = list(self.products.values())
        
        if status:
            products_list = [p for p in products_list if p.status == status]
        
        products_list = products_list[:limit]
        
        return {
            'products': [
                {
                    'id': p.product_id,
                    'title': p.title,
                    'handle': p.handle,
                    'product_type': p.product_type,
                    'vendor': p.vendor,
                    'status': p.status,
                    'price': p.price,
                    'compare_at_price': p.compare_at_price,
                    'inventory_quantity': p.inventory_quantity,
                    'sku': p.sku,
                    'weight': p.weight,
                    'tags': p.tags,
                    'created_at': p.created_at.isoformat(),
                    'updated_at': p.updated_at.isoformat()
                }
                for p in products_list
            ],
            'count': len(products_list),
            'total_available': len(self.products)
        }
    
    def get_product(self, product_id: str) -> Dict[str, Any]:
        """Get single product"""
        if product_id not in self.products:
            return {'error': 'Product not found'}
        
        product = self.products[product_id]
        return {
            'product': {
                'id': product.product_id,
                'title': product.title,
                'handle': product.handle,
                'product_type': product.product_type,
                'vendor': product.vendor,
                'status': product.status,
                'price': product.price,
                'compare_at_price': product.compare_at_price,
                'inventory_quantity': product.inventory_quantity,
                'sku': product.sku,
                'weight': product.weight,
                'tags': product.tags,
                'created_at': product.created_at.isoformat(),
                'updated_at': product.updated_at.isoformat()
            }
        }
    
    def create_product(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create new product"""
        product_id = f"PROD_{uuid.uuid4().hex[:8].upper()}"
        
        try:
            product = ShopifyProduct(
                product_id=product_id,
                title=product_data.get('title', 'New Product'),
                handle=product_data.get('handle', 'new-product'),
                product_type=product_data.get('product_type', 'General'),
                vendor=product_data.get('vendor', 'Default Vendor'),
                status=product_data.get('status', 'draft'),
                price=float(product_data.get('price', 0.0)),
                compare_at_price=product_data.get('compare_at_price'),
                inventory_quantity=int(product_data.get('inventory_quantity', 0)),
                sku=product_data.get('sku', f'SKU-{product_id}'),
                weight=float(product_data.get('weight', 0.0)),
                tags=product_data.get('tags', []),
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            
            self.products[product_id] = product
            
            self.logger.log_action("shopify_product_created", {
                'product_id': product_id,
                'title': product.title,
                'price': product.price
            })
            
            return {
                'success': True,
                'product': {
                    'id': product_id,
                    'title': product.title,
                    'status': product.status,
                    'price': product.price
                }
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def update_product(self, product_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update existing product"""
        if product_id not in self.products:
            return {'success': False, 'error': 'Product not found'}
        
        try:
            product = self.products[product_id]
            
            # Update fields
            for field, value in updates.items():
                if hasattr(product, field):
                    setattr(product, field, value)
            
            product.updated_at = datetime.now()
            
            self.logger.log_action("shopify_product_updated", {
                'product_id': product_id,
                'updates': list(updates.keys())
            })
            
            return {'success': True, 'product_id': product_id}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_orders(self, limit: int = 50, status: str = None) -> Dict[str, Any]:
        """Get orders list"""
        orders_list = list(self.orders.values())
        
        if status:
            orders_list = [o for o in orders_list if o.financial_status == status]
        
        # Sort by creation date (newest first)
        orders_list.sort(key=lambda x: x.created_at, reverse=True)
        orders_list = orders_list[:limit]
        
        return {
            'orders': [
                {
                    'id': o.order_id,
                    'order_number': o.order_number,
                    'email': o.email,
                    'customer_id': o.customer_id,
                    'financial_status': o.financial_status,
                    'fulfillment_status': o.fulfillment_status,
                    'total_price': o.total_price,
                    'subtotal_price': o.subtotal_price,
                    'total_tax': o.total_tax,
                    'total_discounts': o.total_discounts,
                    'line_items_count': len(o.line_items),
                    'created_at': o.created_at.isoformat(),
                    'updated_at': o.updated_at.isoformat()
                }
                for o in orders_list
            ],
            'count': len(orders_list),
            'total_available': len(self.orders)
        }
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """Get single order"""
        if order_id not in self.orders:
            return {'error': 'Order not found'}
        
        order = self.orders[order_id]
        return {
            'order': {
                'id': order.order_id,
                'order_number': order.order_number,
                'email': order.email,
                'customer_id': order.customer_id,
                'financial_status': order.financial_status,
                'fulfillment_status': order.fulfillment_status,
                'total_price': order.total_price,
                'subtotal_price': order.subtotal_price,
                'total_tax': order.total_tax,
                'total_discounts': order.total_discounts,
                'line_items': order.line_items,
                'shipping_address': order.shipping_address,
                'created_at': order.created_at.isoformat(),
                'updated_at': order.updated_at.isoformat()
            }
        }
    
    def get_customers(self, limit: int = 50) -> Dict[str, Any]:
        """Get customers list"""
        customers_list = list(self.customers.values())
        
        # Sort by total spent (highest first)
        customers_list.sort(key=lambda x: x.total_spent, reverse=True)
        customers_list = customers_list[:limit]
        
        return {
            'customers': [
                {
                    'id': c.customer_id,
                    'email': c.email,
                    'first_name': c.first_name,
                    'last_name': c.last_name,
                    'phone': c.phone,
                    'accepts_marketing': c.accepts_marketing,
                    'total_spent': c.total_spent,
                    'orders_count': c.orders_count,
                    'state': c.state,
                    'tags': c.tags,
                    'created_at': c.created_at.isoformat(),
                    'updated_at': c.updated_at.isoformat()
                }
                for c in customers_list
            ],
            'count': len(customers_list),
            'total_available': len(self.customers)
        }
    
    def get_customer(self, customer_id: str) -> Dict[str, Any]:
        """Get single customer"""
        if customer_id not in self.customers:
            return {'error': 'Customer not found'}
        
        customer = self.customers[customer_id]
        
        # Get customer's orders
        customer_orders = [o for o in self.orders.values() if o.customer_id == customer_id]
        
        return {
            'customer': {
                'id': customer.customer_id,
                'email': customer.email,
                'first_name': customer.first_name,
                'last_name': customer.last_name,
                'phone': customer.phone,
                'accepts_marketing': customer.accepts_marketing,
                'total_spent': customer.total_spent,
                'orders_count': customer.orders_count,
                'state': customer.state,
                'tags': customer.tags,
                'created_at': customer.created_at.isoformat(),
                'updated_at': customer.updated_at.isoformat(),
                'recent_orders': [
                    {
                        'id': o.order_id,
                        'order_number': o.order_number,
                        'total_price': o.total_price,
                        'created_at': o.created_at.isoformat()
                    }
                    for o in sorted(customer_orders, key=lambda x: x.created_at, reverse=True)[:5]
                ]
            }
        }
    
    def get_analytics(self, period: str = '30d') -> Dict[str, Any]:
        """Get store analytics"""
        # Calculate date range
        if period == '7d':
            start_date = datetime.now() - timedelta(days=7)
        elif period == '30d':
            start_date = datetime.now() - timedelta(days=30)
        elif period == '90d':
            start_date = datetime.now() - timedelta(days=90)
        else:
            start_date = datetime.now() - timedelta(days=30)
        
        # Filter orders by date range
        period_orders = [o for o in self.orders.values() if o.created_at >= start_date]
        paid_orders = [o for o in period_orders if o.financial_status == 'paid']
        
        # Calculate metrics
        total_sales = sum(o.total_price for o in paid_orders)
        total_orders = len(paid_orders)
        avg_order_value = total_sales / total_orders if total_orders > 0 else 0
        
        # Customer metrics
        unique_customers = len(set(o.customer_id for o in paid_orders if o.customer_id))
        
        # Product metrics
        product_sales = {}
        for order in paid_orders:
            for item in order.line_items:
                product_id = item['product_id']
                if product_id not in product_sales:
                    product_sales[product_id] = {'quantity': 0, 'revenue': 0, 'title': item['title']}
                product_sales[product_id]['quantity'] += item['quantity']
                product_sales[product_id]['revenue'] += item['total']
        
        # Top products
        top_products = sorted(product_sales.items(), key=lambda x: x[1]['revenue'], reverse=True)[:5]
        
        # Daily sales data for the period
        daily_sales = {}
        for order in paid_orders:
            date_key = order.created_at.strftime('%Y-%m-%d')
            if date_key not in daily_sales:
                daily_sales[date_key] = {'sales': 0, 'orders': 0}
            daily_sales[date_key]['sales'] += order.total_price
            daily_sales[date_key]['orders'] += 1
        
        return {
            'period': period,
            'total_sales': round(total_sales, 2),
            'total_orders': total_orders,
            'average_order_value': round(avg_order_value, 2),
            'unique_customers': unique_customers,
            'conversion_rate': round((total_orders / len(period_orders)) * 100, 2) if period_orders else 0,
            'top_products': [
                {
                    'product_id': pid,
                    'title': data['title'],
                    'quantity_sold': data['quantity'],
                    'revenue': round(data['revenue'], 2)
                }
                for pid, data in top_products
            ],
            'daily_breakdown': [
                {
                    'date': date,
                    'sales': round(data['sales'], 2),
                    'orders': data['orders']
                }
                for date, data in sorted(daily_sales.items())
            ]
        }
    
    def get_inventory_report(self) -> Dict[str, Any]:
        """Get inventory status report"""
        active_products = [p for p in self.products.values() if p.status == 'active']
        
        low_stock = [p for p in active_products if p.inventory_quantity <= 10]
        out_of_stock = [p for p in active_products if p.inventory_quantity == 0]
        
        total_inventory_value = sum(p.price * p.inventory_quantity for p in active_products)
        
        return {
            'total_products': len(active_products),
            'total_inventory_value': round(total_inventory_value, 2),
            'low_stock_count': len(low_stock),
            'out_of_stock_count': len(out_of_stock),
            'low_stock_products': [
                {
                    'id': p.product_id,
                    'title': p.title,
                    'sku': p.sku,
                    'quantity': p.inventory_quantity,
                    'price': p.price
                }
                for p in low_stock[:10]  # Top 10
            ],
            'out_of_stock_products': [
                {
                    'id': p.product_id,
                    'title': p.title,
                    'sku': p.sku,
                    'price': p.price
                }
                for p in out_of_stock[:10]  # Top 10
            ]
        }
    
    # Campaign management methods (for interface compatibility)
    def create_campaign(self, campaign_data: Dict[str, Any]) -> Dict[str, Any]:
        """Shopify doesn't support campaigns directly"""
        return {
            'success': False,
            'error': 'Shopify does not support marketing campaigns directly. Use Shopify marketing apps or external platforms.',
            'suggestion': 'Consider integrating with Shopify marketing apps or using customer data for external campaign targeting.'
        }
    
    def get_campaign_metrics(self, campaign_id: str) -> Dict[str, float]:
        """Shopify doesn't have campaign metrics"""
        return {
            'error': 'Shopify does not support campaign metrics directly',
            'alternative': 'Use get_analytics() for store performance metrics'
        }
    
    def update_campaign_budget(self, campaign_id: str, new_budget: float) -> Dict[str, Any]:
        """Shopify doesn't support campaigns"""
        return {
            'success': False,
            'error': 'Shopify does not support marketing campaigns directly'
        }
    
    def pause_campaign(self, campaign_id: str) -> Dict[str, Any]:
        """Shopify doesn't support campaigns"""
        return {
            'success': False,
            'error': 'Shopify does not support marketing campaigns directly'
        }
    
    def enable_campaign(self, campaign_id: str) -> Dict[str, Any]:
        """Shopify doesn't support campaigns"""
        return {
            'success': False,
            'error': 'Shopify does not support marketing campaigns directly'
        }
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get account information (store info)"""
        return self.get_store_info()
    
    def simulate_webhook(self, event_type: str, data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Simulate Shopify webhook events"""
        webhook_data = {
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'shop_domain': self.store.domain,
            'data': data or {}
        }
        
        self.logger.log_action("shopify_webhook_simulated", {
            'event_type': event_type,
            'timestamp': webhook_data['timestamp']
        })
        
        return {
            'success': True,
            'webhook': webhook_data,
            'message': f'Webhook {event_type} simulated successfully'
        }
    
    def export_customer_data(self, format: str = 'csv') -> Dict[str, Any]:
        """Export customer data for external marketing tools"""
        customers_data = []
        
        for customer in self.customers.values():
            customers_data.append({
                'customer_id': customer.customer_id,
                'email': customer.email,
                'first_name': customer.first_name,
                'last_name': customer.last_name,
                'phone': customer.phone,
                'accepts_marketing': customer.accepts_marketing,
                'total_spent': customer.total_spent,
                'orders_count': customer.orders_count,
                'tags': ','.join(customer.tags),
                'created_at': customer.created_at.isoformat(),
                'state': customer.state
            })
        
        return {
            'success': True,
            'format': format,
            'customer_count': len(customers_data),
            'data': customers_data[:100],  # Limit for demo
            'message': f'Customer data exported in {format} format'
        }