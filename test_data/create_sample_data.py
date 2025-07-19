#!/usr/bin/env python3
"""
Sample Data Generator for Agent Testing

This script creates sample datasets for testing the multi-agent system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

def create_sales_data_small():
    """Create a small sales dataset (~1000 rows)."""
    np.random.seed(42)  # For reproducible results
    
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(1000)]
    
    regions = ['North', 'South', 'East', 'West']
    products = ['Product_A', 'Product_B', 'Product_C', 'Product_D']
    sales_reps = [f'Rep_{i:02d}' for i in range(1, 21)]
    
    data = {
        'date': dates,
        'revenue': np.random.normal(1000, 200, 1000).round(2),
        'units_sold': np.random.poisson(50, 1000),
        'region': np.random.choice(regions, 1000),
        'product': np.random.choice(products, 1000),
        'sales_rep': np.random.choice(sales_reps, 1000),
        'customer_id': np.random.randint(1000, 5000, 1000),
        'discount_rate': np.random.uniform(0, 0.3, 1000).round(3)
    }
    
    df = pd.DataFrame(data)
    
    # Add some seasonal trends
    df['month'] = pd.to_datetime(df['date']).dt.month
    df.loc[df['month'].isin([11, 12]), 'revenue'] *= 1.3  # Holiday boost
    df.loc[df['month'].isin([6, 7, 8]), 'units_sold'] *= 1.1  # Summer boost
    
    # Add some missing values and outliers for testing cleaning
    missing_indices = np.random.choice(df.index, 50, replace=False)
    df.loc[missing_indices[:25], 'revenue'] = np.nan
    df.loc[missing_indices[25:], 'units_sold'] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(df.index, 20, replace=False)
    df.loc[outlier_indices, 'revenue'] *= 5
    
    return df

def create_sales_data_medium():
    """Create a medium sales dataset (~100K rows)."""
    np.random.seed(42)
    
    # Create 3 years of daily data for multiple stores
    start_date = datetime(2021, 1, 1)
    end_date = datetime(2023, 12, 31)
    date_range = pd.date_range(start_date, end_date, freq='D')
    
    stores = [f'Store_{i:03d}' for i in range(1, 101)]  # 100 stores
    regions = ['North', 'South', 'East', 'West', 'Central']
    products = [f'Product_{chr(65+i)}' for i in range(20)]  # 20 products A-T
    categories = ['Electronics', 'Clothing', 'Food', 'Home', 'Sports']
    
    data = []
    for date in date_range:
        for store in np.random.choice(stores, 30):  # ~30 stores per day
            for _ in range(np.random.randint(1, 5)):  # 1-4 transactions per store
                data.append({
                    'date': date,
                    'store_id': store,
                    'region': np.random.choice(regions),
                    'product': np.random.choice(products),
                    'category': np.random.choice(categories),
                    'revenue': np.random.lognormal(6, 1),  # Lognormal for realistic revenue
                    'units_sold': np.random.poisson(10),
                    'customer_age': np.random.normal(40, 15),
                    'customer_gender': np.random.choice(['M', 'F', 'Other']),
                    'payment_method': np.random.choice(['Cash', 'Credit', 'Debit', 'Digital']),
                    'promotion_active': np.random.choice([True, False], p=[0.3, 0.7])
                })
    
    df = pd.DataFrame(data)
    
    # Add seasonal and trend effects
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Holiday effects
    df.loc[df['month'].isin([11, 12]), 'revenue'] *= 1.4
    df.loc[df['month'].isin([6, 7]), 'revenue'] *= 1.1
    
    # Weekend effects
    df.loc[df['day_of_week'].isin([5, 6]), 'revenue'] *= 1.2
    
    # Growth trend over years
    df.loc[df['year'] == 2022, 'revenue'] *= 1.05
    df.loc[df['year'] == 2023, 'revenue'] *= 1.1
    
    return df

def create_market_data_json():
    """Create market data in JSON format."""
    np.random.seed(42)
    
    start_date = datetime(2023, 1, 1)
    dates = [(start_date + timedelta(days=i)).isoformat() for i in range(365)]
    
    market_data = {
        "metadata": {
            "source": "Sample Market Data",
            "created": datetime.now().isoformat(),
            "description": "Daily market indicators for testing"
        },
        "data": []
    }
    
    for i, date in enumerate(dates):
        market_data["data"].append({
            "date": date,
            "stock_indices": {
                "SP500": 4000 + np.random.normal(0, 50),
                "NASDAQ": 12000 + np.random.normal(0, 200),
                "DOW": 33000 + np.random.normal(0, 300)
            },
            "currencies": {
                "USD_EUR": 0.85 + np.random.normal(0, 0.02),
                "USD_GBP": 0.75 + np.random.normal(0, 0.01),
                "USD_JPY": 110 + np.random.normal(0, 2)
            },
            "commodities": {
                "oil_wti": 70 + np.random.normal(0, 5),
                "gold": 2000 + np.random.normal(0, 30),
                "silver": 25 + np.random.normal(0, 2)
            },
            "economic_indicators": {
                "unemployment_rate": 4.0 + np.random.normal(0, 0.2),
                "inflation_rate": 3.0 + np.random.normal(0, 0.3),
                "gdp_growth": 2.5 + np.random.normal(0, 0.5)
            }
        })
    
    return market_data

def create_malformed_data():
    """Create intentionally malformed data for testing error handling."""
    data = {
        'date': ['2023-01-01', '2023-01-02', 'invalid-date', '2023-01-04'],
        'revenue': [1000, None, 'not_a_number', 1500],
        'region': ['North', '', None, 'InvalidRegion'],
        'special_chars': ['normal', 'with,comma', 'with"quote', 'with\ttab'],
        'mixed_types': [1, '2', 3.5, 'four']
    }
    
    return pd.DataFrame(data)

def main():
    """Generate all sample datasets."""
    # Create test_data directory if it doesn't exist
    os.makedirs('test_data', exist_ok=True)
    
    print("🔄 Generating sample datasets...")
    
    # Small dataset
    print("📊 Creating small sales dataset...")
    small_data = create_sales_data_small()
    small_data.to_csv('test_data/sales_data_small.csv', index=False)
    print(f"✅ Created sales_data_small.csv: {small_data.shape}")
    
    # Medium dataset
    print("📊 Creating medium sales dataset...")
    medium_data = create_sales_data_medium()
    medium_data.to_csv('test_data/sales_data_medium.csv', index=False)
    print(f"✅ Created sales_data_medium.csv: {medium_data.shape}")
    
    # JSON dataset
    print("📊 Creating JSON market data...")
    json_data = create_market_data_json()
    with open('test_data/market_data.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"✅ Created market_data.json: {len(json_data['data'])} records")
    
    # Malformed dataset
    print("📊 Creating malformed dataset...")
    malformed_data = create_malformed_data()
    malformed_data.to_csv('test_data/malformed_data.csv', index=False)
    print(f"✅ Created malformed_data.csv: {malformed_data.shape}")
    
    print("\n🎉 All sample datasets created successfully!")
    print("\nDatasets created:")
    print("- test_data/sales_data_small.csv (for quick testing)")
    print("- test_data/sales_data_medium.csv (for performance testing)")
    print("- test_data/market_data.json (for JSON testing)")
    print("- test_data/malformed_data.csv (for error handling testing)")

if __name__ == "__main__":
    main() 