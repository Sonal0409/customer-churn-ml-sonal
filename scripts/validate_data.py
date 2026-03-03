"""
Data Schema Validation Script

This script validates that the data conforms to expected schema:
- Correct column names
- Correct data types
- Value ranges are valid
- Required columns present

Used in CI/CD pipeline to catch data issues early.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Schema definition for customer churn data
EXPECTED_SCHEMA = {
    'age': {
        'dtype': 'int64',
        'min': 18,
        'max': 100,
        'nullable': False
    },
    'tenure_months': {
        'dtype': 'int64',
        'min': 0,
        'max': 120,
        'nullable': False
    },
    'monthly_charges': {
        'dtype': 'float64',
        'min': 0.0,
        'max': 500.0,
        'nullable': False
    },
    'support_calls': {
        'dtype': 'int64',
        'min': 0,
        'max': 50,
        'nullable': False
    },
    'churn': {
        'dtype': 'int64',
        'min': 0,
        'max': 1,
        'nullable': False
    }
}


def generate_sample_data(n_samples=1000):
    """
    Generate synthetic data for validation
    (In production, this would load from actual data source)
    """
    np.random.seed(42)

    data = {
        'age': np.random.randint(18, 70, n_samples),
        'tenure_months': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'support_calls': np.random.randint(0, 10, n_samples)
    }

    df = pd.DataFrame(data)

    # Create target
    churn_prob = (
        0.3 * (df['support_calls'] > 5).astype(int) +
        0.2 * (df['tenure_months'] < 12).astype(int)
    )
    df['churn'] = (churn_prob > 0.3).astype(int)

    return df


def validate_columns(df):
    """
    Validate that all required columns are present
    """
    expected_columns = set(EXPECTED_SCHEMA.keys())
    actual_columns = set(df.columns)

    missing_columns = expected_columns - actual_columns
    extra_columns = actual_columns - expected_columns

    if missing_columns:
        print(f"❌ Missing columns: {missing_columns}")
        return False

    if extra_columns:
        print(f"⚠️  Extra columns (will be ignored): {extra_columns}")

    print("✅ All required columns present")
    return True


def validate_dtypes(df):
    """
    Validate data types match expected schema
    """
    validation_passed = True

    for col, schema in EXPECTED_SCHEMA.items():
        if col not in df.columns:
            continue

        expected_dtype = schema['dtype']
        actual_dtype = str(df[col].dtype)

        # Allow some flexibility in integer types
        if expected_dtype.startswith('int') and actual_dtype.startswith('int'):
            print(f"✅ Column '{col}': dtype OK ({actual_dtype})")
            continue

        # Allow some flexibility in float types
        if expected_dtype.startswith('float') and actual_dtype.startswith('float'):
            print(f"✅ Column '{col}': dtype OK ({actual_dtype})")
            continue

        if actual_dtype != expected_dtype:
            print(f"❌ Column '{col}': expected {expected_dtype}, got {actual_dtype}")
            validation_passed = False
        else:
            print(f"✅ Column '{col}': dtype OK")

    return validation_passed


def validate_value_ranges(df):
    """
    Validate that values are within expected ranges
    """
    validation_passed = True

    for col, schema in EXPECTED_SCHEMA.items():
        if col not in df.columns:
            continue

        # Check min value
        if 'min' in schema:
            min_val = df[col].min()
            expected_min = schema['min']
            if min_val < expected_min:
                print(f"❌ Column '{col}': min value {min_val} < expected {expected_min}")
                validation_passed = False
            else:
                print(f"✅ Column '{col}': min value OK ({min_val} >= {expected_min})")

        # Check max value
        if 'max' in schema:
            max_val = df[col].max()
            expected_max = schema['max']
            if max_val > expected_max:
                print(f"❌ Column '{col}': max value {max_val} > expected {expected_max}")
                validation_passed = False
            else:
                print(f"✅ Column '{col}': max value OK ({max_val} <= {expected_max})")

    return validation_passed


def validate_nulls(df):
    """
    Validate that non-nullable columns don't contain nulls
    """
    validation_passed = True

    for col, schema in EXPECTED_SCHEMA.items():
        if col not in df.columns:
            continue

        nullable = schema.get('nullable', True)
        has_nulls = df[col].isnull().any()

        if not nullable and has_nulls:
            null_count = df[col].isnull().sum()
            print(f"❌ Column '{col}': contains {null_count} null values (not allowed)")
            validation_passed = False
        elif has_nulls:
            null_count = df[col].isnull().sum()
            print(f"⚠️  Column '{col}': contains {null_count} null values (allowed)")
        else:
            print(f"✅ Column '{col}': no null values")

    return validation_passed


def main():
    """
    Main validation function
    """
    print("=" * 60)
    print("DATA SCHEMA VALIDATION")
    print("=" * 60)
    print()

    # Generate or load data
    print("📊 Loading data...")
    try:
        df = generate_sample_data()
        print(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
        print()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    # Run all validations
    all_passed = True

    print("🔍 Validating columns...")
    if not validate_columns(df):
        all_passed = False
    print()

    print("🔍 Validating data types...")
    if not validate_dtypes(df):
        all_passed = False
    print()

    print("🔍 Validating value ranges...")
    if not validate_value_ranges(df):
        all_passed = False
    print()

    print("🔍 Validating null values...")
    if not validate_nulls(df):
        all_passed = False
    print()

    # Final result
    print("=" * 60)
    if all_passed:
        print("✅ SCHEMA VALIDATION PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ SCHEMA VALIDATION FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()