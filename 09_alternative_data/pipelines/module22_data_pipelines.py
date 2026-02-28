"""
Module 22: Data Engineering Pipelines
=====================================
ETL pipelines, data quality, feature stores
"""

import pandas as pd
import numpy as np
from datetime import datetime


class DataPipeline:
    """ETL pipeline for quantitative data."""
    
    def __init__(self):
        self.data_quality_checks = []
    
    def extract(self, source):
        """Extract data from source."""
        # Simulate data extraction
        data = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=100),
            'price': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100)
        })
        return data
    
    def validate_schema(self, df, required_columns):
        """Validate data schema."""
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            return {'valid': False, 'missing': missing}
        return {'valid': True}
    
    def check_null_values(self, df):
        """Check for null values."""
        null_counts = df.isnull().sum()
        return {'null_counts': null_counts.to_dict(), 'has_nulls': null_counts.sum() > 0}
    
    def check_data_range(self, df, column, min_val, max_val):
        """Check if data is within expected range."""
        out_of_range = ((df[column] < min_val) | (df[column] > max_val)).sum()
        return {'out_of_range': out_of_range, 'valid': out_of_range == 0}
    
    def transform(self, df):
        """Transform data."""
        df['returns'] = df['price'].pct_change()
        df['log_volume'] = np.log(df['volume'])
        df['ma_5'] = df['price'].rolling(window=5).mean()
        df['ma_20'] = df['price'].rolling(window=20).mean()
        return df
    
    def load(self, df, destination):
        """Load data to destination."""
        # Simulate loading
        print(f"  Loading {len(df)} rows to {destination}")
        return {'status': 'success', 'rows': len(df)}


if __name__ == "__main__":
    print("=" * 70)
    print("  MODULE 22: DATA ENGINEERING PIPELINES")
    print("=" * 70)
    
    pipeline = DataPipeline()
    
    print(f"\n── Extract ──")
    data = pipeline.extract('market_data')
    print(f"  Extracted {len(data)} rows")
    
    print(f"\n── Validate ──")
    schema_check = pipeline.validate_schema(data, ['date', 'price', 'volume'])
    print(f"  Schema valid: {schema_check['valid']}")
    
    null_check = pipeline.check_null_values(data)
    print(f"  Null values: {null_check['has_nulls']}")
    
    range_check = pipeline.check_data_range(data, 'price', 50, 150)
    print(f"  Range check: {'PASS' if range_check['valid'] else 'FAIL'}")
    
    print(f"\n── Transform ──")
    transformed = pipeline.transform(data)
    print(f"  Added features: returns, log_volume, ma_5, ma_20")
    
    print(f"\n── Load ──")
    result = pipeline.load(transformed, 'feature_store')
    print(f"  Status: {result['status']}")
    
    print(f"\n✓ Module 22 complete")
