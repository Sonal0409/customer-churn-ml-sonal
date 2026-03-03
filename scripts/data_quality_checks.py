"""
Data Quality Checks Script

This script performs comprehensive data quality checks:
- Missing value analysis
- Outlier detection
- Distribution checks
- Feature correlations
- Data consistency checks

Used in CI/CD pipeline to ensure data quality before training.
"""

import sys
import pandas as pd
import numpy as np
from scipy import stats


def generate_sample_data(n_samples=1000):
    """
    Generate synthetic data for quality checks
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


def check_missing_values(df, threshold=0.05):
    """
    Check for missing values
    Threshold: Maximum allowed percentage of missing values
    """
    print("🔍 Checking for missing values...")

    missing_pct = df.isnull().sum() / len(df)
    missing_cols = missing_pct[missing_pct > 0]

    if len(missing_cols) == 0:
        print("✅ No missing values found")
        return True

    print(f"⚠️  Found missing values in {len(missing_cols)} columns:")
    for col, pct in missing_cols.items():
        status = "❌" if pct > threshold else "⚠️ "
        print(f"   {status} {col}: {pct*100:.2f}% missing")

    if (missing_pct > threshold).any():
        print(f"❌ Some columns exceed threshold of {threshold*100}%")
        return False

    print(f"✅ All missing value percentages below threshold")
    return True


def check_outliers(df, z_threshold=3.0):
    """
    Check for outliers using Z-score method
    """
    print("
🔍 Checking for outliers...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    has_severe_outliers = False

    for col in numeric_cols:
        if col == 'churn':  # Skip target variable
            continue

        z_scores = np.abs(stats.zscore(df[col]))
        outliers = (z_scores > z_threshold).sum()
        outlier_pct = outliers / len(df) * 100

        if outlier_pct > 5:  # More than 5% outliers is concerning
            print(f"   ⚠️  {col}: {outliers} outliers ({outlier_pct:.2f}%)")
            if outlier_pct > 10:
                has_severe_outliers = True
        else:
            print(f"   ✅ {col}: {outliers} outliers ({outlier_pct:.2f}%)")

    if has_severe_outliers:
        print("⚠️  Some columns have >10% outliers - review recommended")
    else:
        print("✅ Outlier levels acceptable")

    return not has_severe_outliers


def check_class_balance(df, target_col='churn', imbalance_threshold=0.1):
    """
    Check target class balance
    """
    print("
🔍 Checking class balance...")

    if target_col not in df.columns:
        print(f"⚠️  Target column '{target_col}' not found")
        return True

    class_counts = df[target_col].value_counts()
    class_pcts = df[target_col].value_counts(normalize=True)

    print(f"   Class distribution:")
    for class_val, count in class_counts.items():
        pct = class_pcts[class_val]
        print(f"      Class {class_val}: {count} samples ({pct*100:.2f}%)")

    min_class_pct = class_pcts.min()

    if min_class_pct < imbalance_threshold:
        print(f"❌ Severe class imbalance detected (minority class: {min_class_pct*100:.2f}%)")
        return False
    elif min_class_pct < 0.3:
        print(f"⚠️  Class imbalance detected (minority class: {min_class_pct*100:.2f}%)")
        print("   Consider using stratified sampling or class weights")
    else:
        print("✅ Class balance is acceptable")

    return True


def check_feature_variance(df, variance_threshold=0.01):
    """
    Check for low-variance features
    """
    print("
🔍 Checking feature variance...")

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    low_variance_features = []

    for col in numeric_cols:
        if col == 'churn':  # Skip target
            continue

        variance = df[col].var()
        if variance < variance_threshold:
            low_variance_features.append((col, variance))
            print(f"   ⚠️  {col}: low variance ({variance:.6f})")
        else:
            print(f"   ✅ {col}: variance OK ({variance:.2f})")

    if low_variance_features:
        print(f"⚠️  Found {len(low_variance_features)} low-variance features")
        print("   Consider removing or transforming these features")
    else:
        print("✅ All features have sufficient variance")

    return len(low_variance_features) == 0


def check_duplicate_rows(df, threshold=0.01):
    """
    Check for duplicate rows
    """
    print("
🔍 Checking for duplicate rows...")

    n_duplicates = df.duplicated().sum()
    duplicate_pct = n_duplicates / len(df)

    if n_duplicates == 0:
        print("✅ No duplicate rows found")
        return True

    print(f"   Found {n_duplicates} duplicate rows ({duplicate_pct*100:.2f}%)")

    if duplicate_pct > threshold:
        print(f"❌ Duplicate rows exceed threshold of {threshold*100}%")
        return False

    print("✅ Duplicate rows below threshold")
    return True


def check_feature_correlations(df, correlation_threshold=0.95):
    """
    Check for highly correlated features
    """
    print("
🔍 Checking feature correlations...")

    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns
                    if col != 'churn']

    if len(numeric_cols) < 2:
        print("⚠️  Not enough numeric features for correlation analysis")
        return True

    corr_matrix = df[numeric_cols].corr()
    high_corr_pairs = []

    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_value = abs(corr_matrix.iloc[i, j])
            if corr_value > correlation_threshold:
                col1 = corr_matrix.columns[i]
                col2 = corr_matrix.columns[j]
                high_corr_pairs.append((col1, col2, corr_value))
                print(f"   ⚠️  High correlation: {col1} <-> {col2} ({corr_value:.3f})")

    if not high_corr_pairs:
        print("✅ No highly correlated feature pairs found")
        return True

    print(f"⚠️  Found {len(high_corr_pairs)} highly correlated pairs")
    print("   Consider removing redundant features")
    return True  # Warning, not failure


def check_data_freshness(df, date_col=None, max_age_days=30):
    """
    Check data freshness (if date column available)
    """
    print("
🔍 Checking data freshness...")

    if date_col is None or date_col not in df.columns:
        print("⚠️  No date column available - skipping freshness check")
        return True

    # This would check actual dates in production
    print("✅ Data freshness check passed (demo mode)")
    return True


def generate_quality_report(df):
    """
    Generate summary quality report
    """
    print("
" + "=" * 60)
    print("DATA QUALITY SUMMARY REPORT")
    print("=" * 60)

    print(f"
📊 Dataset Overview:")
    print(f"   • Rows: {len(df):,}")
    print(f"   • Columns: {len(df.columns)}")
    print(f"   • Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    print(f"
📈 Feature Statistics:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'churn':
            print(f"   • {col}:")
            print(f"      - Mean: {df[col].mean():.2f}")
            print(f"      - Std: {df[col].std():.2f}")
            print(f"      - Min: {df[col].min():.2f}")
            print(f"      - Max: {df[col].max():.2f}")

    print()


def main():
    """
    Main quality check function
    """
    print("=" * 60)
    print("DATA QUALITY CHECKS")
    print("=" * 60)
    print()

    # Load data
    print("📊 Loading data...")
    try:
        df = generate_sample_data()
        print(f"✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
        print()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        sys.exit(1)

    # Run all quality checks
    all_passed = True

    checks = [
        check_missing_values(df),
        check_outliers(df),
        check_class_balance(df),
        check_feature_variance(df),
        check_duplicate_rows(df),
        check_feature_correlations(df),
        check_data_freshness(df)
    ]

    all_passed = all(checks)

    # Generate report
    generate_quality_report(df)

    # Final result
    print("=" * 60)
    if all_passed:
        print("✅ ALL DATA QUALITY CHECKS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print("❌ SOME DATA QUALITY CHECKS FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()