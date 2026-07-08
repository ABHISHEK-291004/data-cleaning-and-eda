import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# 0. Raw Data Generation (Client Raw Data)
# We generate a dataset that mimics a raw, messy customer database to satisfy all tasks.
def generate_messy_data(filename="customer_raw_data.csv"):
    n_rows = 1000
    
    # 1. CustomerID (1001 to 2000)
    customer_ids = np.arange(1001, 1001 + n_rows)
    
    # 2. Name
    first_names = ['Alex', 'Emma', 'Liam', 'Sophia', 'Mason', 'Olivia', 'Ethan', 'Ava', 'Lucas', 'Isabella']
    last_names = ['Smith', 'Johnson', 'Brown', 'Taylor', 'Miller', 'Wilson', 'Davis', 'Anderson', 'Thomas', 'White']
    names = [f"{np.random.choice(first_names)} {np.random.choice(last_names)}" for _ in range(n_rows)]
    
    # 3. Age (Normally distributed, but we inject some outliers and anomalies)
    age = np.random.normal(37, 12, n_rows).astype(int)
    age = np.clip(age, 18, 80)
    # Inject obvious outliers / anomalies
    age[np.random.choice(n_rows, 8, replace=False)] = [99, 108, 115, 120, -10, -5, 130, -15]
    
    # 4. Annual_Income (Log-normal distribution for positive skew, stored as string with '$' and ',')
    raw_income = np.random.lognormal(mean=10.9, sigma=0.6, size=n_rows)
    annual_income = [f"${int(val):,}" for val in raw_income]
    
    # 5. Spending_Score (Beta distribution for negative skew, scaled to 0-100)
    spending_score = np.random.beta(a=5, b=2, size=n_rows) * 100
    
    # 6. Purchase_Frequency (Consistently increases with spending score, monotonic but non-linear relationship)
    purchase_frequency = np.exp(spending_score / 30) + np.random.normal(3, 1.5, n_rows)
    purchase_frequency = np.clip(purchase_frequency, 1, 100)
    
    # 7. Membership_Tier (Repetitive string column)
    tiers = ['Bronze', 'Silver', 'Gold', 'Platinum']
    membership_tier = np.random.choice(tiers, n_rows, p=[0.50, 0.30, 0.14, 0.06])
    
    # 8. Signup_Date
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', periods=n_rows)
    signup_date = np.random.choice(dates, n_rows).astype(str)
    
    # 9. Satisfaction_Score (Numeric, has nulls)
    satisfaction_score = np.random.choice([1.0, 2.0, 3.0, 4.0, 5.0], n_rows, p=[0.08, 0.12, 0.20, 0.40, 0.20])
    
    # 10. Unrelated_Empty_Col (100% null)
    unrelated_empty_col = [np.nan] * n_rows
    
    # 11. Extra_Null_Col (35% null)
    extra_null_col = np.random.choice([np.nan, 'Active', 'Inactive'], n_rows, p=[0.35, 0.50, 0.15])
    
    df = pd.DataFrame({
        'CustomerID': customer_ids,
        'Name': names,
        'Age': age,
        'Annual_Income': annual_income,
        'Spending_Score': spending_score,
        'Purchase_Frequency': purchase_frequency,
        'Membership_Tier': membership_tier,
        'Signup_Date': signup_date,
        'Satisfaction_Score': satisfaction_score,
        'Unrelated_Empty_Col': unrelated_empty_col,
        'Extra_Null_Col': extra_null_col
    })
    
    # Inject nulls under 20% for numeric columns
    # Annual_Income (5% nulls)
    null_inc = np.random.choice(n_rows, int(n_rows * 0.05), replace=False)
    for idx in null_inc:
        df.loc[idx, 'Annual_Income'] = np.nan
        
    # Spending_Score (7% nulls)
    null_spd = np.random.choice(n_rows, int(n_rows * 0.07), replace=False)
    df.loc[null_spd, 'Spending_Score'] = np.nan
    
    # Satisfaction_Score (10% nulls)
    null_sat = np.random.choice(n_rows, int(n_rows * 0.10), replace=False)
    df.loc[null_sat, 'Satisfaction_Score'] = np.nan
    
    # Inject 25 exact duplicate rows
    dup_indices = np.random.choice(n_rows, 25, replace=False)
    df_dups = df.iloc[dup_indices].copy()
    df = pd.concat([df, df_dups], ignore_index=True)
    
    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(filename, index=False)
    print(f"Created messy raw data file: '{filename}'")

# Generate the data
generate_messy_data()

# Create plots folder if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Task 1: Load dataset & print metadata

print("\n" + "="*50)
print("TASK 1: Loading Dataset")
print("="*50)
df = pd.read_csv('customer_raw_data.csv')
print(f"DataFrame Shape: {df.shape}")
print("\nFirst 5 Rows:")
print(df.head())
print("\nColumn Data Types (dtypes):")
print(df.dtypes)

# Save initial state for comparison in later tasks
df_initial = df.copy()


# Task 2: Null value analysis

print("\n" + "="*50)
print("TASK 2: Null Value Analysis")
print("="*50)
null_counts = df.isnull().sum()
null_percentages = (null_counts / df.shape[0]) * 100

null_report = pd.DataFrame({
    'Null Count': null_counts,
    'Null Percentage (%)': null_percentages
})
print("Null count and percentage for each column:")
print(null_report)

# Identify columns exceeding 20% null rate
exceed_20 = null_report[null_report['Null Percentage (%)'] > 20.0].index.tolist()
print(f"\nColumns exceeding 20% null rate: {exceed_20}")

# For numeric columns below 20% nulls, fill with the median
# Currently, the numeric columns with nulls are Spending_Score and Satisfaction_Score.
# It will be converted in Task 4 and imputed in Task 9a).
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
imputed_cols = []
for col in numeric_cols:
    pct = null_percentages[col]
    if pct > 0 and pct < 20:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        imputed_cols.append(col)
        print(f"Filled numeric column '{col}' nulls with its median: {median_val:.2f}")


# Task 3: Duplicate detection and removal

print("\n" + "="*50)
print("TASK 3: Duplicate Detection & Removal")
print("="*50)
num_dups = df.duplicated().sum()
print(f"Total duplicate rows detected: {num_dups}")

# Drop duplicates
df_no_dups = df.drop_duplicates()
rows_removed = len(df) - len(df_no_dups)
df = df_no_dups.copy()
print(f"Number of rows removed: {rows_removed}")
print(f"New shape after dropping duplicates: {df.shape}")

# Report whether the removal changes any column's null percentage
null_counts_after = df.isnull().sum()
null_pct_after = (null_counts_after / df.shape[0]) * 100

null_comparison = pd.DataFrame({
    'Null Pct Before': null_percentages,
    'Null Pct After': null_pct_after,
    'Difference': null_pct_after - null_percentages
})
print("\nComparison of null percentages before vs. after duplicate removal:")
print(null_comparison)


# Task 4: Data type correction

print("\n" + "="*50)
print("TASK 4: Data Type Correction")
print("="*50)
# Deep memory check before conversion
mem_before = df.memory_usage(deep=True).sum()
print(f"Memory usage before conversion: {mem_before:,} bytes")

# 1. Convert Annual_Income to numeric (strip '$' and ',')
df['Annual_Income'] = df['Annual_Income'].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
df['Annual_Income'] = pd.to_numeric(df['Annual_Income'], errors='coerce')
print("Converted 'Annual_Income' to numeric float64.")

# 2. Convert repetitive string column (Membership_Tier) to category
df['Membership_Tier'] = df['Membership_Tier'].astype('category')
print("Converted 'Membership_Tier' to category type.")

# Deep memory check after conversion
mem_after = df.memory_usage(deep=True).sum()
print(f"Memory usage after conversion: {mem_after:,} bytes")
print(f"Net memory savings: {mem_before - mem_after:,} bytes ({(1 - mem_after/mem_before)*100:.2f}% reduction)")


# Task 5: Descriptive statistics & skewness

print("\n" + "="*50)
print("TASK 5: Descriptive Statistics & Skewness")
print("="*50)
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
print("Descriptive statistics for numeric columns:")
print(df[num_cols].describe())

print("\nSkewness for numeric columns:")
skew_vals = {}
for col in num_cols:
    skew_val = df[col].skew()
    skew_vals[col] = skew_val
    print(f"  {col}: {skew_val:.4f}")

# Find the column with the highest absolute skewness
highest_skew_col = max(skew_vals, key=lambda k: abs(skew_vals[k]))
print(f"\nColumn with the highest absolute skewness: '{highest_skew_col}' (Skewness: {skew_vals[highest_skew_col]:.4f})")

# Task 6: Outlier detection with IQR

print("\n" + "="*50)
print("TASK 6: Outlier Detection with IQR")
print("="*50)
outlier_cols = ['Age', 'Annual_Income']
for col in outlier_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count rows falling outside the bounds
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    print(f"Column '{col}':")
    print(f"  Q1 (25th percentile): {Q1:.2f}")
    print(f"  Q3 (75th percentile): {Q3:.2f}")
    print(f"  IQR: {IQR:.2f}")
    print(f"  Lower Bound: {lower_bound:.2f}")
    print(f"  Upper Bound: {upper_bound:.2f}")
    print(f"  Number of Outliers: {len(outliers)} ({(len(outliers)/len(df))*100:.2f}%)")
    print("-" * 30)


# Task 7: Visualizations (5 required plots)

print("\n" + "="*50)
print("TASK 7: Generating Visualizations")
print("="*50)

sns.set_theme(style="whitegrid")

# 1. Line plot: numeric variable sorted by index
df_sorted = df.sort_values(by='CustomerID').reset_index(drop=True)
plt.figure(figsize=(10, 5))
plt.plot(df_sorted.index, df_sorted['Purchase_Frequency'], color='royalblue', alpha=0.7, linewidth=1.5)
plt.title('Purchase Frequency Over Customer Registration Index', fontsize=14, pad=15)
plt.xlabel('Customer Registration Sequence (Index)', fontsize=12)
plt.ylabel('Purchase Frequency (Purchases/Year)', fontsize=12)
plt.tight_layout()
plt.savefig('plots/line_plot.png', dpi=150)
plt.close()
print("Saved 1/5: line_plot.png")

# 2. Bar chart: Comparing the mean of one numeric column across categories
tier_income = df.groupby('Membership_Tier', observed=False)['Annual_Income'].mean().reset_index()
plt.figure(figsize=(8, 5))
plt.bar(tier_income['Membership_Tier'].astype(str), tier_income['Annual_Income'], color=['#cd7f32', '#c0c0c0', '#ffd700', '#e5e4e2'], edgecolor='black', width=0.6)
plt.title('Mean Annual Income by Membership Tier', fontsize=14, pad=15)
plt.xlabel('Membership Tier', fontsize=12)
plt.ylabel('Mean Annual Income ($)', fontsize=12)
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels([f"${int(x):,}" for x in current_values])
plt.tight_layout()
plt.savefig('plots/bar_chart.png', dpi=150)
plt.close()
print("Saved 2/5: bar_chart.png")

# 3. Histogram of the most skewed numeric column (bins=20)
plt.figure(figsize=(8, 5))
sns.histplot(df['Annual_Income'].dropna(), bins=20, kde=True, color='crimson', edgecolor='black')
plt.title(f'Distribution of Annual Income (Most Skewed Variable: {skew_vals[highest_skew_col]:.2f})', fontsize=14, pad=15)
plt.xlabel('Annual Income ($)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('plots/histogram.png', dpi=150)
plt.close()
print("Saved 3/5: histogram.png")

# 4. Scatter plot between two numeric columns expected to be correlated
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Spending_Score', y='Purchase_Frequency', alpha=0.7, color='teal', edgecolor='w', s=50)
plt.title('Customer Spending Score vs. Purchase Frequency', fontsize=14, pad=15)
plt.xlabel('Spending Score (0-100)', fontsize=12)
plt.ylabel('Purchase Frequency (Purchases/Year)', fontsize=12)
plt.tight_layout()
plt.savefig('plots/scatterplot.png', dpi=150)
plt.close()
print("Saved 4/5: scatterplot.png")

# 5. Box plot of a numeric column split by a categorical column
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Membership_Tier', y='Spending_Score', palette='Set2')
plt.title('Distribution of Spending Score by Membership Tier', fontsize=14, pad=15)
plt.xlabel('Membership Tier', fontsize=12)
plt.ylabel('Spending Score (0-100)', fontsize=12)
plt.tight_layout()
plt.savefig('plots/boxplot.png', dpi=150)
plt.close()
print("Saved 5/5: boxplot.png")


# Task 8: Correlation heat map

print("\n" + "="*50)
print("TASK 8: Correlation Heat Map")
print("="*50)
num_cols_only = df.select_dtypes(include=['int64', 'float64'])
corr_matrix = num_cols_only.corr()
print("Pearson Correlation Matrix:")
print(corr_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5, vmin=-1, vmax=1)
plt.title('Pearson Correlation Heatmap of Numeric Variables', fontsize=14, pad=15)
plt.tight_layout()
plt.savefig('plots/heatmap.png', dpi=150)
plt.close()
print("Saved heatmap: heatmap.png")

corr_unstacked = corr_matrix.abs().unstack()
corr_unstacked = corr_unstacked[corr_unstacked.index.get_level_values(0) != corr_unstacked.index.get_level_values(1)]
highest_pair = corr_unstacked.idxmax()
highest_corr_val = corr_matrix.loc[highest_pair[0], highest_pair[1]]
print(f"Highest absolute correlation: {highest_pair[0]} & {highest_pair[1]} (Pearson Correlation: {highest_corr_val:.4f})")

# Task 9a: Imputation strategy comparison

print("\n" + "="*50)
print("TASK 9a: Imputation Strategy Comparison")
print("="*50)
# Re-compute skewness to find top 2 skewed numeric columns (which will be Annual_Income and Age)
skew_vals_updated = {}
for col in num_cols_only.columns:
    skew_vals_updated[col] = df[col].skew()
skew_sorted = sorted(skew_vals_updated.items(), key=lambda x: abs(x[1]), reverse=True)
top_2_skewed = [skew_sorted[0][0], skew_sorted[1][0]]
print(f"Two numeric columns with highest absolute skewness: {top_2_skewed}")

for col in top_2_skewed:
    raw_mean = df[col].mean()
    raw_median = df[col].median()
    print(f"Column '{col}' (Pre-Imputation):")
    print(f"  Mean:   {raw_mean:.4f}")
    print(f"  Median: {raw_median:.4f}")

for col in top_2_skewed:
    nulls_before = df[col].isnull().sum()
    if nulls_before > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"Imputed {nulls_before} null values in '{col}' with median: {median_val:.2f}")
    else:
        print(f"Column '{col}' already contains 0 null values.")

print("\nVerification of null values in the two columns after imputation:")
print(df[top_2_skewed].isnull().sum())

# Task 9b: Spearman rank correlation

print("\n" + "="*50)
print("TASK 9b: Spearman Rank Correlation")
print("="*50)
spearman_corr = num_cols_only.corr(method='spearman')

print("Spearman Correlation Matrix:")
print(spearman_corr)

diff_pairs = []
cols_list = corr_matrix.columns.tolist()
for i in range(len(cols_list)):
    for j in range(i + 1, len(cols_list)):
        c1, c2 = cols_list[i], cols_list[j]
        p_val = corr_matrix.loc[c1, c2]
        s_val = spearman_corr.loc[c1, c2]
        
        # Skip if either correlation is NaN (e.g. for completely null columns)
        if pd.isna(p_val) or pd.isna(s_val):
            continue
            
        diff_val = s_val - p_val
        abs_diff = abs(diff_val)
        diff_pairs.append(((c1, c2), p_val, s_val, diff_val, abs_diff))

diff_pairs_sorted = sorted(diff_pairs, key=lambda x: x[4], reverse=True)
top_3_diffs = diff_pairs_sorted[:3]

print("\nTop 3 Column Pairs with Largest Absolute Difference (Spearman - Pearson):")
print(f"{'Column Pair':<45} | {'Spearman':<10} | {'Pearson':<10} | {'Difference':<10}")
print("-" * 85)
for (c1, c2), p_val, s_val, diff_val, abs_diff in top_3_diffs:
    pair_name = f"{c1} & {c2}"
    print(f"{pair_name:<45} | {s_val:<10.4f} | {p_val:<10.4f} | {diff_val:<10.4f}")

# ==========================================
# Task 9c: Grouped aggregation
# ==========================================
print("\n" + "="*50)
print("TASK 9c: Grouped Aggregation")
print("="*50)
grouped_agg = df.groupby('Membership_Tier', observed=False)['Spending_Score'].agg(['mean', 'std', 'count'])
print("Grouped Aggregation of Spending Score by Membership Tier:")
print(grouped_agg)


# Task 10: Save clean dataset

print("\n" + "="*50)
print("TASK 10: Saving Cleaned Dataset")
print("="*50)
df_clean = df.drop(columns=exceed_20)
print(f"Dropped columns exceeding 20% null rate: {exceed_20}")
print(f"Final clean dataset shape: {df_clean.shape}")

df_clean.to_csv('cleaned_data.csv', index=False)
print("Saved final cleaned data to 'cleaned_data.csv'")
print("="*50)
