#!/usr/bin/env python3
"""
Complete Example: Groupby + Apply in Pandas
This script demonstrates various use cases of groupby().apply() with sample data.
All generated files are saved in the same directory as this script.
FIXED VERSION - handles MultiIndex and Windows encoding issues properly.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# ============================================================================
# CONFIGURATION: Get script directory for saving files
# ============================================================================

def get_script_directory():
    """Get the directory where this script is located"""
    # Get the absolute path of the script
    script_path = os.path.abspath(__file__)
    # Return the directory containing the script
    return os.path.dirname(script_path)

# Set the output directory to the script's directory
SCRIPT_DIR = get_script_directory()
OUTPUT_DIR = SCRIPT_DIR

print("=" * 70)
print("GROUPBY + APPLY DEMONSTRATION SCRIPT (FIXED FOR WINDOWS)")
print("=" * 70)
print(f"Script directory: {SCRIPT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 70)

# ============================================================================
# 1. CREATE SAMPLE DATASET
# ============================================================================

def create_sample_data(n_rows=200):
    """Create a realistic sample sales dataset"""
    np.random.seed(42)  # For reproducible results
    
    # Generate sample data
    regions = ['North', 'South', 'East', 'West']
    products = ['Laptop', 'Phone', 'Tablet', 'Monitor', 'Keyboard', 'Mouse']
    categories = ['Electronics', 'Accessories', 'Software']
    
    # Create base data
    data = {
        'transaction_id': [f"TXN{1000+i}" for i in range(n_rows)],
        'date': pd.date_range('2024-01-01', periods=n_rows, freq='h'),
        'region': np.random.choice(regions, n_rows, p=[0.3, 0.3, 0.2, 0.2]),
        'product': np.random.choice(products, n_rows),
        'category': np.random.choice(categories, n_rows),
        'customer_id': [f"CUST{np.random.randint(100, 999)}" for _ in range(n_rows)],
        'quantity': np.random.randint(1, 6, n_rows),
        'unit_price': np.random.uniform(100, 2000, n_rows).round(2),
        'discount_pct': np.random.choice([0, 5, 10, 15, 20], n_rows, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    }
    
    df = pd.DataFrame(data)
    
    # Calculate derived columns
    df['sales_amount'] = df['quantity'] * df['unit_price'] * (1 - df['discount_pct']/100)
    df['sales_amount'] = df['sales_amount'].round(2)
    
    # Add profit (random between 15-35% of sales)
    df['profit'] = df['sales_amount'] * np.random.uniform(0.15, 0.35, n_rows)
    df['profit'] = df['profit'].round(2)
    
    # Add time-based features
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month
    
    return df

# Create the dataset
print("\n1. CREATING SAMPLE DATASET")
print("-" * 40)
df = create_sample_data(200)
print(f"[OK] Dataset created with {df.shape[0]} rows and {df.shape[1]} columns")
print(f"    Saved in memory as 'df' variable")
print(f"    First 5 rows preview:")
print(df.head().to_string())

# Also save the original sample data to CSV
sample_data_path = os.path.join(OUTPUT_DIR, "sample_sales_data.csv")
df.to_csv(sample_data_path, index=False)
print(f"[OK] Sample data saved to: {sample_data_path}")

# ============================================================================
# 2. EXAMPLE 1: COMPLEX AGGREGATIONS WITH APPLY (Returns Series)
# ============================================================================

print("\n\n2. EXAMPLE 1: Complex Business Metrics by Region")
print("-" * 40)

def calculate_region_performance(group):
    """
    Calculate comprehensive performance metrics for a region.
    This function returns a Series that will become one row in the result.
    """
    # Basic metrics
    total_sales = group['sales_amount'].sum()
    total_profit = group['profit'].sum()
    avg_transaction = group['sales_amount'].mean()
    
    # Advanced metrics
    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
    
    # Customer metrics
    unique_customers = group['customer_id'].nunique()
    avg_sale_per_customer = total_sales / unique_customers if unique_customers > 0 else 0
    
    # Discount analysis
    high_value_transactions = len(group[group['sales_amount'] > group['sales_amount'].median()])
    discount_impact = group['discount_pct'].mean()
    
    # Product diversity
    product_variety = group['product'].nunique()
    
    # Time-based metrics
    peak_hour = group.groupby('hour')['sales_amount'].sum().idxmax() if not group.empty else None
    
    # Performance tier (business logic)
    if profit_margin > 25:
        tier = 'Excellent'
    elif profit_margin > 20:
        tier = 'Good'
    elif profit_margin > 15:
        tier = 'Average'
    else:
        tier = 'Needs Improvement'
    
    # Return as a Series - each key becomes a column in the result
    return pd.Series({
        'total_sales': total_sales,
        'total_profit': total_profit,
        'profit_margin_pct': round(profit_margin, 2),
        'avg_transaction_value': round(avg_transaction, 2),
        'unique_customers': unique_customers,
        'avg_sale_per_customer': round(avg_sale_per_customer, 2),
        'transaction_count': len(group),
        'high_value_transactions': high_value_transactions,
        'avg_discount_pct': round(discount_impact, 2),
        'product_variety': product_variety,
        'peak_hour': peak_hour,
        'performance_tier': tier
    })

# Apply the function to each region group
region_analysis = df.groupby('region').apply(calculate_region_performance)

print(f"[OK] Region analysis completed")
print(f"    Generated {region_analysis.shape[0]} regions with {region_analysis.shape[1]} metrics each")
print("\n    Results preview:")
print(region_analysis.to_string())

# Save to CSV
region_analysis_path = os.path.join(OUTPUT_DIR, "region_performance_analysis.csv")
region_analysis.to_csv(region_analysis_path)
print(f"[OK] Saved to: {region_analysis_path}")

# ============================================================================
# 3. EXAMPLE 2: DATA TRANSFORMATION WITHIN GROUPS (Returns DataFrame)
# ============================================================================

print("\n\n3. EXAMPLE 2: Normalize Sales Data Within Each Product Category")
print("-" * 40)

def normalize_within_category(group):
    """
    Normalize sales data within each category and add summary statistics.
    This returns a DataFrame with the same number of rows as the input group.
    """
    # Create a copy to avoid modifying the original
    result = group.copy()
    
    # Calculate category-level statistics
    cat_sales_mean = group['sales_amount'].mean()
    cat_sales_std = group['sales_amount'].std()
    cat_total_sales = group['sales_amount'].sum()
    
    # Normalize sales within category (z-score)
    if cat_sales_std > 0:
        result['sales_zscore'] = (group['sales_amount'] - cat_sales_mean) / cat_sales_std
    else:
        result['sales_zscore'] = 0
    
    # Calculate percentile rank within category
    result['sales_percentile'] = group['sales_amount'].rank(pct=True) * 100
    
    # Flag outliers within category (more than 2 std from mean)
    result['is_outlier'] = abs(result['sales_zscore']) > 2
    
    # Add category summary as new columns (repeated for each row)
    result['category_avg_sale'] = cat_sales_mean
    result['category_total_sales'] = cat_total_sales
    result['category_transaction_count'] = len(group)
    
    # Calculate contribution percentage
    result['contribution_pct'] = (result['sales_amount'] / cat_total_sales * 100).round(2)
    
    return result

# Apply normalization within each category - using group_keys=False avoids MultiIndex issues
normalized_df = df.groupby('category', group_keys=False).apply(normalize_within_category)

print(f"[OK] Data normalization completed")
print(f"    Original shape: {df.shape}")
print(f"    Normalized shape: {normalized_df.shape}")
print(f"    Added new columns: sales_zscore, sales_percentile, is_outlier, etc.")

# Save to CSV
normalized_path = os.path.join(OUTPUT_DIR, "normalized_sales_by_category.csv")
normalized_df.to_csv(normalized_path, index=False)
print(f"[OK] Saved to: {normalized_path}")

# Show a quick summary of outliers
outlier_summary = normalized_df.groupby('category')['is_outlier'].sum()
print(f"\n    Outlier summary by category:")
for category, count in outlier_summary.items():
    print(f"        {category}: {count} outliers detected")

# ============================================================================
# 4. EXAMPLE 3: TIME-SERIES ANALYSIS PER CUSTOMER (Returns Series)
# ============================================================================

print("\n\n4. EXAMPLE 3: Customer Behavior Analysis")
print("-" * 40)

def analyze_customer_behavior(customer_group):
    """
    Analyze purchasing behavior for a single customer.
    Returns a Series with customer metrics.
    """
    # Sort by date for time-based analysis
    customer_group = customer_group.sort_values('date')
    
    # Basic metrics
    total_spent = customer_group['sales_amount'].sum()
    total_profit = customer_group['profit'].sum()
    transaction_count = len(customer_group)
    
    # Time-based metrics
    if transaction_count > 1:
        first_purchase = customer_group['date'].min()
        last_purchase = customer_group['date'].max()
        days_between = (last_purchase - first_purchase).days
        avg_days_between = days_between / (transaction_count - 1) if transaction_count > 1 else 0
        
        # Calculate purchase frequency (transactions per week)
        weeks_active = max(1, days_between / 7)
        purchase_frequency = transaction_count / weeks_active
    else:
        first_purchase = customer_group['date'].iloc[0] if not customer_group.empty else None
        last_purchase = first_purchase
        avg_days_between = 0
        purchase_frequency = 0
    
    # Recency (days since last purchase)
    latest_date_in_data = df['date'].max()
    recency_days = (latest_date_in_data - last_purchase).days
    
    # Product preferences
    favorite_category = customer_group['category'].mode()
    favorite_category = favorite_category.iloc[0] if not favorite_category.empty else 'None'
    
    favorite_product = customer_group['product'].mode()
    favorite_product = favorite_product.iloc[0] if not favorite_product.empty else 'None'
    
    # Discount sensitivity
    avg_discount = customer_group['discount_pct'].mean()
    
    # Value segmentation
    avg_transaction_value = total_spent / transaction_count if transaction_count > 0 else 0
    
    # Customer segmentation logic
    if transaction_count >= 5 and avg_transaction_value > 500:
        segment = 'VIP'
    elif transaction_count >= 3:
        segment = 'Regular'
    elif total_spent > 1000:
        segment = 'High Value New'
    else:
        segment = 'New/Infrequent'
    
    # Return customer profile
    return pd.Series({
        'customer_id': customer_group['customer_id'].iloc[0],
        'total_spent': round(total_spent, 2),
        'total_profit': round(total_profit, 2),
        'transaction_count': transaction_count,
        'avg_transaction_value': round(avg_transaction_value, 2),
        'first_purchase': first_purchase,
        'last_purchase': last_purchase,
        'recency_days': recency_days,
        'avg_days_between_purchases': round(avg_days_between, 2),
        'purchase_frequency_per_week': round(purchase_frequency, 2),
        'favorite_category': favorite_category,
        'favorite_product': favorite_product,
        'avg_discount_used': round(avg_discount, 2),
        'customer_segment': segment,
        'profit_margin_pct': round((total_profit / total_spent * 100), 2) if total_spent > 0 else 0
    })

# Apply to top 15 customers by transaction count
top_customers = df['customer_id'].value_counts().head(15).index
customer_data = df[df['customer_id'].isin(top_customers)]

customer_profiles = customer_data.groupby('customer_id').apply(analyze_customer_behavior)

print(f"[OK] Customer analysis completed")
print(f"    Analyzed {len(customer_profiles)} customers")
print(f"\n    Top 5 customers by total spent:")
top_customers_table = customer_profiles.sort_values('total_spent', ascending=False).head()
print(top_customers_table[['customer_id', 'total_spent', 'transaction_count', 'customer_segment']].to_string(index=False))

# Save to CSV
customer_profiles_path = os.path.join(OUTPUT_DIR, "customer_behavior_profiles.csv")
customer_profiles.to_csv(customer_profiles_path)
print(f"\n[OK] Saved to: {customer_profiles_path}")

# ============================================================================
# 5. EXAMPLE 4: FIXED - MULTI-LEVEL ANALYSIS RETURNING DATAFRAMES
# ============================================================================

print("\n\n5. EXAMPLE 4: Detailed Product Analysis by Region (FIXED)")
print("-" * 40)

def detailed_product_analysis(region_group):
    """
    Perform detailed analysis for a region and return a DataFrame
    with multiple rows (one per product) per region.
    """
    # Get region name from the group
    region_name = region_group['region'].iloc[0]
    
    # Initialize list to store results
    product_analyses = []
    
    # Analyze each product in the region
    for product, product_group in region_group.groupby('product'):
        # Basic metrics
        total_sales = product_group['sales_amount'].sum()
        total_quantity = product_group['quantity'].sum()
        avg_price = product_group['unit_price'].mean()
        
        # Time analysis
        sales_by_hour = product_group.groupby('hour')['sales_amount'].sum()
        peak_hour = sales_by_hour.idxmax() if not sales_by_hour.empty else None
        
        # Customer analysis
        unique_customers = product_group['customer_id'].nunique()
        
        # Discount impact
        avg_discount = product_group['discount_pct'].mean()
        discounted_sales = product_group[product_group['discount_pct'] > 0]['sales_amount'].sum()
        discount_effectiveness = (discounted_sales / total_sales * 100) if total_sales > 0 else 0
        
        # Create product analysis row
        analysis_row = {
            'region': region_name,  # Use the region name directly
            'product': product,
            'category': product_group['category'].iloc[0],
            'total_sales': round(total_sales, 2),
            'total_quantity': total_quantity,
            'avg_unit_price': round(avg_price, 2),
            'transaction_count': len(product_group),
            'unique_customers': unique_customers,
            'peak_hour': peak_hour,
            'avg_discount_pct': round(avg_discount, 2),
            'discounted_sales_pct': round(discount_effectiveness, 2),
            'sales_per_transaction': round(total_sales / len(product_group), 2),
            'quantity_per_transaction': round(total_quantity / len(product_group), 2)
        }
        
        product_analyses.append(analysis_row)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(product_analyses)
    
    # Add region-level summary columns
    region_total_sales = region_group['sales_amount'].sum()
    result_df['product_contribution_pct'] = (result_df['total_sales'] / region_total_sales * 100).round(2)
    
    # Calculate rank within region
    result_df['sales_rank_in_region'] = result_df['total_sales'].rank(ascending=False).astype(int)
    
    return result_df

# Apply the analysis to each region - RESET INDEX to avoid MultiIndex issues
detailed_analysis = df.groupby('region', group_keys=False).apply(detailed_product_analysis)

print(f"[OK] Detailed product analysis completed")
print(f"    Generated {detailed_analysis.shape[0]} product-region combinations")

# Show summary by region (using the 'region' column, not index)
region_counts = detailed_analysis['region'].value_counts()
print(f"    Products per region:")
for region, count in region_counts.items():
    print(f"        {region}: {count} products")

# Save to CSV
detailed_analysis_path = os.path.join(OUTPUT_DIR, "detailed_product_analysis_by_region.csv")
detailed_analysis.to_csv(detailed_analysis_path, index=False)
print(f"[OK] Saved to: {detailed_analysis_path}")

# Show top products by region
print("\n    Top products by region:")
for region in df['region'].unique():
    region_products = detailed_analysis[detailed_analysis['region'] == region]
    if not region_products.empty:
        top_product_row = region_products.loc[region_products['total_sales'].idxmax()]
        print(f"        {region}: {top_product_row['product']} (${top_product_row['total_sales']:.2f})")

# ============================================================================
# 6. EXAMPLE 5: REAL-WORLD USE CASE - ANOMALY DETECTION
# ============================================================================

print("\n\n6. EXAMPLE 5: Anomaly Detection in Sales Transactions")
print("-" * 40)

def detect_anomalies(category_group):
    """
    Detect anomalous transactions within each product category.
    Returns a DataFrame with anomaly flags and scores.
    """
    # Calculate category statistics
    sales_mean = category_group['sales_amount'].mean()
    sales_std = category_group['sales_amount'].std()
    quantity_mean = category_group['quantity'].mean()
    quantity_std = category_group['quantity'].std()
    
    # Initialize result
    result = category_group.copy()
    
    # Statistical anomaly detection
    if sales_std > 0:
        # Z-score for sales amount
        result['sales_zscore'] = (category_group['sales_amount'] - sales_mean) / sales_std
        
        # Flag anomalies (more than 3 standard deviations)
        result['sales_anomaly'] = abs(result['sales_zscore']) > 3
        
        # Anomaly score (0-100)
        result['anomaly_score'] = (abs(result['sales_zscore']) / 3 * 100).clip(0, 100)
    else:
        result['sales_zscore'] = 0
        result['sales_anomaly'] = False
        result['anomaly_score'] = 0
    
    # Quantity anomalies
    if quantity_std > 0:
        quantity_zscore = (category_group['quantity'] - quantity_mean) / quantity_std
        result['quantity_anomaly'] = abs(quantity_zscore) > 3
    else:
        result['quantity_anomaly'] = False
    
    # Business rule anomalies
    result['high_discount_anomaly'] = category_group['discount_pct'] > 20  # Unusually high discount
    result['low_price_anomaly'] = category_group['unit_price'] < 50  # Suspiciously low price
    
    # Combined anomaly flag
    result['is_anomaly'] = (
        result['sales_anomaly'] | 
        result['quantity_anomaly'] | 
        result['high_discount_anomaly'] | 
        result['low_price_anomaly']
    )
    
    # Add explanation for anomalies
    def create_anomaly_explanation(row):
        reasons = []
        if row['sales_anomaly']:
            reasons.append('Extreme sales amount')
        if row['quantity_anomaly']:
            reasons.append('Unusual quantity')
        if row['high_discount_anomaly']:
            reasons.append('High discount')
        if row['low_price_anomaly']:
            reasons.append('Low unit price')
        return ', '.join(reasons) if reasons else 'Normal'
    
    result['anomaly_reason'] = result.apply(create_anomaly_explanation, axis=1)
    
    return result

# Apply anomaly detection per category
anomaly_results = df.groupby('category', group_keys=False).apply(detect_anomalies)

print(f"[OK] Anomaly detection completed")
print(f"    Processed {anomaly_results.shape[0]} transactions")

# Summary of anomalies
anomaly_summary = anomaly_results.groupby('category').agg({
    'transaction_id': 'count',
    'is_anomaly': 'sum',
    'anomaly_score': 'mean'
}).rename(columns={
    'transaction_id': 'total_transactions',
    'is_anomaly': 'anomaly_count'
})

anomaly_summary['anomaly_pct'] = (anomaly_summary['anomaly_count'] / 
                                  anomaly_summary['total_transactions'] * 100).round(2)

print(f"\n    Anomaly Summary:")
print(anomaly_summary.to_string())

# Save to CSV
anomaly_results_path = os.path.join(OUTPUT_DIR, "anomaly_detection_results.csv")
anomaly_results.to_csv(anomaly_results_path, index=False)
print(f"\n[OK] Saved detailed results to: {anomaly_results_path}")

# Save anomaly summary separately
anomaly_summary_path = os.path.join(OUTPUT_DIR, "anomaly_summary.csv")
anomaly_summary.to_csv(anomaly_summary_path)
print(f"[OK] Saved summary to: {anomaly_summary_path}")

# ============================================================================
# 7. BONUS: VISUALIZATION EXAMPLE (Optional)
# ============================================================================

try:
    import matplotlib.pyplot as plt
    
    print("\n\n7. BONUS: Creating Visualizations")
    print("-" * 40)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Region sales comparison
    region_analysis['total_sales'].plot(kind='bar', ax=axes[0, 0], color='skyblue')
    axes[0, 0].set_title('Total Sales by Region')
    axes[0, 0].set_ylabel('Sales ($)')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Profit margins
    region_analysis['profit_margin_pct'].plot(kind='bar', ax=axes[0, 1], color='lightgreen')
    axes[0, 1].set_title('Profit Margin by Region')
    axes[0, 1].set_ylabel('Profit Margin (%)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Customer segments
    segment_counts = customer_profiles['customer_segment'].value_counts()
    segment_counts.plot(kind='pie', ax=axes[1, 0], autopct='%1.1f%%')
    axes[1, 0].set_title('Customer Segmentation Distribution')
    
    # 4. Anomaly detection by category
    anomaly_summary['anomaly_pct'].plot(kind='bar', ax=axes[1, 1], color='salmon')
    axes[1, 1].set_title('Anomaly Percentage by Category')
    axes[1, 1].set_ylabel('Anomaly %')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save visualization to script directory
    visualization_path = os.path.join(OUTPUT_DIR, "groupby_analysis_visualizations.png")
    plt.savefig(visualization_path, dpi=100, bbox_inches='tight')
    print(f"[OK] Visualizations saved to: {visualization_path}")
    
    # Also create a simple bar chart of region performance
    plt.figure(figsize=(10, 6))
    
    # Reset index for plotting
    plot_data = region_analysis.reset_index()
    
    # Create grouped bar chart
    x = np.arange(len(plot_data))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot total sales on primary axis
    ax1.bar(x - width/2, plot_data['total_sales'], width, label='Total Sales ($)', color='skyblue')
    ax1.set_xlabel('Region')
    ax1.set_ylabel('Total Sales ($)', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_data['region'])
    
    # Create secondary axis for profit margin
    ax2 = ax1.twinx()
    ax2.plot(x, plot_data['profit_margin_pct'], color='red', marker='o', linewidth=2, label='Profit Margin (%)')
    ax2.set_ylabel('Profit Margin (%)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add title and legend
    plt.title('Region Performance: Sales vs Profit Margin')
    fig.tight_layout()
    
    combined_chart_path = os.path.join(OUTPUT_DIR, "region_sales_profit_chart.png")
    plt.savefig(combined_chart_path, dpi=100, bbox_inches='tight')
    print(f"[OK] Combined chart saved to: {combined_chart_path}")
    
except ImportError:
    print("\n7. BONUS: Visualization (Skipped)")
    print("-" * 40)
    print("Note: Install matplotlib to generate visualizations:")
    print("  pip install matplotlib")

# ============================================================================
# 8. CREATE A SUMMARY REPORT (FIXED for Windows encoding)
# ============================================================================

print("\n\n8. CREATING SUMMARY REPORT")
print("-" * 40)

# Create a summary text file with explicit UTF-8 encoding
summary_path = os.path.join(OUTPUT_DIR, "groupby_apply_summary_report.txt")

with open(summary_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("GROUPBY + APPLY ANALYSIS SUMMARY REPORT\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("EXECUTION DETAILS\n")
    f.write("-" * 40 + "\n")
    f.write(f"Script executed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Script directory: {SCRIPT_DIR}\n")
    f.write(f"Sample data rows: {df.shape[0]}\n")
    f.write(f"Sample data columns: {df.shape[1]}\n")
    f.write(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}\n\n")
    
    f.write("ANALYSIS PERFORMED\n")
    f.write("-" * 40 + "\n")
    f.write("1. Region Performance Analysis\n")
    f.write(f"   - Regions analyzed: {len(region_analysis)}\n")
    f.write(f"   - Metrics calculated: {len(region_analysis.columns)}\n")
    f.write(f"   - Top region by sales: {region_analysis['total_sales'].idxmax()} (${region_analysis['total_sales'].max():.2f})\n\n")
    
    f.write("2. Category Normalization\n")
    f.write(f"   - Categories processed: {df['category'].nunique()}\n")
    f.write(f"   - Outliers detected: {normalized_df['is_outlier'].sum()}\n")
    f.write(f"   - New columns added: 6\n\n")
    
    f.write("3. Customer Behavior Analysis\n")
    f.write(f"   - Customers analyzed: {len(customer_profiles)}\n")
    f.write(f"   - Customer segments identified: {customer_profiles['customer_segment'].nunique()}\n")
    f.write(f"   - VIP customers: {(customer_profiles['customer_segment'] == 'VIP').sum()}\n\n")
    
    f.write("4. Product Analysis by Region (FIXED)\n")
    f.write(f"   - Product-region combinations: {detailed_analysis.shape[0]}\n")
    f.write(f"   - Products analyzed: {detailed_analysis['product'].nunique()}\n\n")
    
    f.write("5. Anomaly Detection\n")
    f.write(f"   - Total anomalies detected: {anomaly_results['is_anomaly'].sum()}\n")
    f.write(f"   - Anomaly rate: {(anomaly_results['is_anomaly'].sum() / len(anomaly_results) * 100):.2f}%\n")
    f.write(f"   - Most anomalous category: {anomaly_summary['anomaly_pct'].idxmax()} ({anomaly_summary['anomaly_pct'].max():.2f}%)\n\n")
    
    f.write("FILES GENERATED\n")
    f.write("-" * 40 + "\n")
    files_generated = [
        ("sample_sales_data.csv", "Original sample dataset"),
        ("region_performance_analysis.csv", "Region performance metrics"),
        ("normalized_sales_by_category.csv", "Normalized sales data with outliers"),
        ("customer_behavior_profiles.csv", "Customer behavior profiles"),
        ("detailed_product_analysis_by_region.csv", "Product analysis by region"),
        ("anomaly_detection_results.csv", "Detailed anomaly detection results"),
        ("anomaly_summary.csv", "Anomaly detection summary"),
        ("groupby_analysis_visualizations.png", "Visualizations (if matplotlib installed)"),
        ("region_sales_profit_chart.png", "Region sales vs profit chart"),
        ("groupby_apply_summary_report.txt", "This summary report")
    ]
    
    for filename, description in files_generated:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            size_kb = os.path.getsize(filepath) / 1024
            f.write(f"[OK] {filename:<45} {description:<40} ({size_kb:.1f} KB)\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("SCRIPT COMPLETED SUCCESSFULLY\n")
    f.write("=" * 70 + "\n")

print(f"[OK] Summary report saved to: {summary_path}")

# ============================================================================
# FINAL SUMMARY (ASCII-only for Windows compatibility)
# ============================================================================

print("\n" + "=" * 70)
print("SCRIPT EXECUTION COMPLETE")
print("=" * 70)

print(f"\nALL FILES SAVED TO: {OUTPUT_DIR}")
print(f"TOTAL ANALYSES PERFORMED: 5")
print(f"FILES GENERATED: 10+")

print("\nGENERATED FILES:")
print("-" * 40)

# List all generated files
generated_files = [
    "sample_sales_data.csv",
    "region_performance_analysis.csv", 
    "normalized_sales_by_category.csv",
    "customer_behavior_profiles.csv",
    "detailed_product_analysis_by_region.csv",
    "anomaly_detection_results.csv",
    "anomaly_summary.csv",
    "groupby_apply_summary_report.txt"
]

# Add visualization files if they exist
vis_files = ["groupby_analysis_visualizations.png", "region_sales_profit_chart.png"]
for vis_file in vis_files:
    if os.path.exists(os.path.join(OUTPUT_DIR, vis_file)):
        generated_files.append(vis_file)

for i, filename in enumerate(generated_files, 1):
    filepath = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        size_str = f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
        print(f"{i:2d}. {filename:<45} [{size_str:>10}]")

print("\nKEY VARIABLES IN MEMORY:")
print("-" * 40)
print("df                - Original sample dataset (200 rows)")
print("region_analysis   - Region performance metrics")
print("normalized_df     - Normalized sales data by category")
print("customer_profiles - Customer behavior analysis")
print("detailed_analysis - Product analysis by region (FIXED)")
print("anomaly_results   - Anomaly detection results")

print("\nFIXED ISSUES:")
print("-" * 40)
print("1. MultiIndex ambiguity in Example 4")
print("2. Windows encoding issues with Unicode characters")
print("3. All groupby().apply() calls use group_keys=False")
print("4. Summary report uses UTF-8 encoding explicitly")

print("\nKEY TAKEAWAYS:")
print("-" * 40)
print("1. Use group_keys=False when returning DataFrames from apply()")
print("2. On Windows, avoid Unicode characters or use explicit encoding")
print("3. When functions return DataFrames, store grouping column as a regular column")
print("4. Always check DataFrame structure after groupby().apply()")

print("\nTO SEE ALL GENERATED FILES, RUN:")
print(f"  dir /b \"{OUTPUT_DIR}\"\\*.csv")
print(f"  dir /b \"{OUTPUT_DIR}\"\\*.png")
print(f"  dir /b \"{OUTPUT_DIR}\"\\*.txt")
print("=" * 70)

# ============================================================================
# BONUS: Show how to inspect DataFrame structures
# ============================================================================

print("\nBONUS: DATAFRAME STRUCTURE INSPECTION")
print("-" * 40)
print("To avoid 'ambiguous column/index' errors, always check structure:")

print("\n1. Check index type:")
print(f"   detailed_analysis.index: {type(detailed_analysis.index)}")

print("\n2. Check for MultiIndex:")
if isinstance(detailed_analysis.index, pd.MultiIndex):
    print("   WARNING: DataFrame has MultiIndex")
    print(f"   Index levels: {detailed_analysis.index.nlevels}")
    print(f"   Index names: {detailed_analysis.index.names}")
else:
    print("   OK: DataFrame has simple index")

print("\n3. Check for duplicate column/index names:")
if hasattr(detailed_analysis.index, 'names'):
    index_names = [name for name in detailed_analysis.index.names if name]
    common_columns = set(detailed_analysis.columns) & set(index_names)
    if common_columns:
        print(f"   WARNING: These appear as both columns and index levels: {common_columns}")
        print("   Solution: Use .reset_index() or rename columns")
    else:
        print("   OK: No duplicate column/index names")
else:
    print("   OK: Simple index, no names to check")

print("\n4. Safe way to groupby after apply():")
print("   # Reset index first to be safe")
print("   df_clean = detailed_analysis.reset_index(drop=True)")
print("   # Now groupby works without ambiguity")
print("   df_clean.groupby('region').size()")

print("\n" + "=" * 70)
print("END OF SCRIPT")
print("=" * 70)