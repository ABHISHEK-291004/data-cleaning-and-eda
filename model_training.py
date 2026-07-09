import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve
)

# Set random seed for reproducibility
np.random.seed(42)

# Create plots folder if it doesn't exist
os.makedirs('plots', exist_ok=True)

# ==========================================
# 1. Load Cleaned Dataset
# ==========================================
print("\n" + "="*50)
print("1. LOADING CLEANED DATASET")
print("="*50)
if not os.path.exists('cleaned_data.csv'):
    raise FileNotFoundError("cleaned_data.csv not found. Please run eda_assignment.py first.")

df = pd.read_csv('cleaned_data.csv')
print(f"Loaded cleaned_data.csv with shape: {df.shape}")

# ==========================================
# 2. Define Binary Target Variable
# ==========================================
# We define "High_Value_Customer" as our target classification label:
# 1 if Spending_Score > 70 AND Annual_Income > 50,000, else 0.
df['High_Value_Customer'] = ((df['Spending_Score'] > 70) & (df['Annual_Income'] > 50000)).astype(int)

target_counts = df['High_Value_Customer'].value_counts()
target_pct = df['High_Value_Customer'].value_counts(normalize=True) * 100
print("\nTarget Variable Distribution ('High_Value_Customer'):")
print(f"  Class 0 (Standard): {target_counts[0]} ({target_pct[0]:.2f}%)")
print(f"  Class 1 (High Value): {target_counts[1]} ({target_pct[1]:.2f}%)")

# ==========================================
# 3. Preprocessing and Feature Engineering
# ==========================================
print("\n" + "="*50)
print("3. PREPROCESSING AND TRAIN-TEST SPLIT")
print("="*50)

# Define predictor features (we drop identifiers, leaks, and target)
# Spending_Score and Annual_Income are dropped to avoid data leakage since they define the target.
features = ['Age', 'Purchase_Frequency', 'Satisfaction_Score', 'Membership_Tier']
X = df[features]
y = df['High_Value_Customer']

# Split the data (80% train, 20% test)
# Stratify=y keeps class distribution identical in train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training Set Shape: X={X_train.shape}, y={y_train.shape}")
print(f"Testing Set Shape:  X={X_test.shape}, y={y_test.shape}")

# Create ColumnTransformer for scaling and encoding
numeric_features = ['Age', 'Purchase_Frequency', 'Satisfaction_Score']
categorical_features = ['Membership_Tier']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
    ]
)

# ==========================================
# 4. Model Training (Baseline vs. Challenger)
# ==========================================
print("\n" + "="*50)
print("4. TRAINING BASELINE AND CHALLENGER MODELS")
print("="*50)

# Baseline: Logistic Regression
lr_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Challenger: Random Forest Classifier
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42, n_estimators=100, max_depth=6))
])

# Fit both models on training data
print("Training Logistic Regression baseline...")
lr_pipeline.fit(X_train, y_train)

print("Training Random Forest Classifier challenger...")
rf_pipeline.fit(X_train, y_train)

# ==========================================
# 5. Model Evaluation
# ==========================================
print("\n" + "="*50)
print("5. MODEL EVALUATION ON TEST SET")
print("="*50)

# Evaluate Baseline (Logistic Regression)
y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

lr_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_lr),
    'Precision': precision_score(y_test, y_pred_lr),
    'Recall': recall_score(y_test, y_pred_lr),
    'F1-Score': f1_score(y_test, y_pred_lr),
    'AUC-ROC': roc_auc_score(y_test, y_prob_lr)
}

# Evaluate Challenger (Random Forest)
y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

rf_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred_rf),
    'Precision': precision_score(y_test, y_pred_rf),
    'Recall': recall_score(y_test, y_pred_rf),
    'F1-Score': f1_score(y_test, y_pred_rf),
    'AUC-ROC': roc_auc_score(y_test, y_prob_rf)
}

# Print evaluation report
print(f"{'Metric':<15} | {'Logistic Regression (Baseline)':<30} | {'Random Forest (Challenger)':<30}")
print("-" * 85)
for metric in lr_metrics:
    print(f"{metric:<15} | {lr_metrics[metric]:<30.4f} | {rf_metrics[metric]:<30.4f}")

# Plot ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_metrics["AUC-ROC"]:.4f})', color='royalblue', lw=2)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {rf_metrics["AUC-ROC"]:.4f})', color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing', alpha=0.5)
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('ROC Curve Comparison (Baseline vs. Challenger)', fontsize=14, pad=15)
plt.legend(loc='lower right', fontsize=11)
plt.tight_layout()
plt.savefig('plots/roc_curves.png', dpi=150)
plt.close()
print("\nSaved ROC curves plot to 'plots/roc_curves.png'")

# ==========================================
# 6. Bootstrap Resampling (AUC Difference)
# ==========================================
print("\n" + "="*50)
print("6. BOOTSTRAP RESAMPLING FOR AUC COMPARISON")
print("="*50)
n_iterations = 1000
boot_auc_diffs = []

# Convert X_test and y_test to numpy arrays/pandas indexes for clean sampling
test_indices = np.arange(len(y_test))

print(f"Running {n_iterations} bootstrap iterations on the test set...")
for i in range(n_iterations):
    # Sample test set indices with replacement
    boot_idx = np.random.choice(test_indices, size=len(test_indices), replace=True)
    
    X_boot = X_test.iloc[boot_idx]
    y_boot = y_test.iloc[boot_idx]
    
    # Ensure both classes are present in the bootstrap sample
    if len(np.unique(y_boot)) < 2:
        continue
    
    # Calculate AUCs on bootstrap sample
    boot_prob_lr = lr_pipeline.predict_proba(X_boot)[:, 1]
    boot_prob_rf = rf_pipeline.predict_proba(X_boot)[:, 1]
    
    boot_auc_lr = roc_auc_score(y_boot, boot_prob_lr)
    boot_auc_rf = roc_auc_score(y_boot, boot_prob_rf)
    
    # Compute difference: Challenger - Baseline
    auc_diff = boot_auc_rf - boot_auc_lr
    boot_auc_diffs.append(auc_diff)

# Calculate statistical metrics
mean_auc_diff = np.mean(boot_auc_diffs)
ci_lower = np.percentile(boot_auc_diffs, 2.5)
ci_upper = np.percentile(boot_auc_diffs, 97.5)
excludes_zero = (ci_lower > 0) or (ci_upper < 0)

print("\nBootstrap Results (Random Forest AUC - Logistic Regression AUC):")
print(f"  Mean AUC Difference: {mean_auc_diff:.4f}")
print(f"  95% Confidence Interval: [{ci_lower:.4f}, {ci_upper:.4f}] (2.5th to 97.5th percentiles)")
print(f"  Interval Excludes Zero:  {excludes_zero}")

if excludes_zero:
    if mean_auc_diff > 0:
        print("  Conclusion: The Random Forest model is statistically significantly better than Logistic Regression (p < 0.05).")
    else:
        print("  Conclusion: The Logistic Regression model is statistically significantly better than Random Forest (p < 0.05).")
else:
    print("  Conclusion: The performance difference between the models is not statistically significant (p >= 0.05).")

# Plot Bootstrap AUC Distribution
plt.figure(figsize=(8, 5))
sns.histplot(boot_auc_diffs, kde=True, color='purple', edgecolor='black', alpha=0.7)
plt.axvline(mean_auc_diff, color='red', linestyle='-', linewidth=2, label=f'Mean Difference ({mean_auc_diff:.4f})')
plt.axvline(ci_lower, color='darkgreen', linestyle='--', linewidth=2, label=f'2.5th Percentile ({ci_lower:.4f})')
plt.axvline(ci_upper, color='darkgreen', linestyle='--', linewidth=2, label=f'97.5th Percentile ({ci_upper:.4f})')
plt.axvline(0, color='black', linestyle=':', linewidth=1.5, label='Zero Line (No Difference)')
plt.xlabel('AUC Difference (Random Forest - Logistic Regression)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Bootstrap Distribution of AUC Difference', fontsize=14, pad=15)
plt.legend(loc='upper right', fontsize=10)
plt.tight_layout()
plt.savefig('plots/bootstrap_auc_distribution.png', dpi=150)
plt.close()
print("Saved bootstrap AUC distribution plot to 'plots/bootstrap_auc_distribution.png'")
print("="*50)
