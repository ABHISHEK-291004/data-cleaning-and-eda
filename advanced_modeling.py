import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

print("==================================================")
print("PART 3: ADVANCED MODELING AND ML PIPELINES")
print("==================================================\n")

# 1. LOAD AND PREPARE DATA
df = pd.read_csv('cleaned_data.csv')
df['High_Value_Customer'] = np.where((df['Spending_Score'] > 70) & (df['Annual_Income'] > 50000), 1, 0)
X = df.drop(columns=['CustomerID', 'Name', 'Signup_Date', 'Spending_Score', 'Annual_Income', 'High_Value_Customer'])
y = df['High_Value_Customer']

# Split data
X_train, X_test, y_clf_train, y_clf_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define Preprocessor for models that need pre-scaled data
numeric_features = ['Age', 'Purchase_Frequency', 'Satisfaction_Score']
categorical_features = ['Membership_Tier']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

# Fit and transform for Task 1-5
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)
feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

print("==================================================")
print("TASK 1: Baseline Decision Tree (Unconstrained)")
print("==================================================")
dt_base = DecisionTreeClassifier(max_depth=None, random_state=42)
dt_base.fit(X_train_scaled, y_clf_train)
base_train_acc = accuracy_score(y_clf_train, dt_base.predict(X_train_scaled))
base_test_acc = accuracy_score(y_clf_test, dt_base.predict(X_test_scaled))
print(f"Train Accuracy: {base_train_acc:.4f}")
print(f"Test Accuracy:  {base_test_acc:.4f}\n")

print("==================================================")
print("TASK 2: Controlled Decision Tree")
print("==================================================")
dt_controlled = DecisionTreeClassifier(max_depth=5, min_samples_split=20, random_state=42)
dt_controlled.fit(X_train_scaled, y_clf_train)
ctrl_train_acc = accuracy_score(y_clf_train, dt_controlled.predict(X_train_scaled))
ctrl_test_acc = accuracy_score(y_clf_test, dt_controlled.predict(X_test_scaled))
print(f"Train Accuracy: {ctrl_train_acc:.4f}")
print(f"Test Accuracy:  {ctrl_test_acc:.4f}\n")

print("==================================================")
print("TASK 3: Gini vs Entropy Comparison")
print("==================================================")
dt_gini = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=42)
dt_entropy = DecisionTreeClassifier(max_depth=5, criterion='entropy', random_state=42)
dt_gini.fit(X_train_scaled, y_clf_train)
dt_entropy.fit(X_train_scaled, y_clf_train)
print(f"Gini Test Accuracy:    {accuracy_score(y_clf_test, dt_gini.predict(X_test_scaled)):.4f}")
print(f"Entropy Test Accuracy: {accuracy_score(y_clf_test, dt_entropy.predict(X_test_scaled)):.4f}\n")

print("==================================================")
print("TASK 4: Random Forest")
print("==================================================")
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train_scaled, y_clf_train)
rf_train_acc = accuracy_score(y_clf_train, rf.predict(X_train_scaled))
rf_test_acc = accuracy_score(y_clf_test, rf.predict(X_test_scaled))
rf_test_auc = roc_auc_score(y_clf_test, rf.predict_proba(X_test_scaled)[:, 1])
print(f"Train Accuracy: {rf_train_acc:.4f}")
print(f"Test Accuracy:  {rf_test_acc:.4f}")
print(f"Test ROC-AUC:   {rf_test_auc:.4f}")

importances = rf.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
top_5 = importance_df.sort_values(by='Importance', ascending=False).head(5)
print("\nTop 5 Features by Importance:")
for idx, row in top_5.iterrows():
    print(f" - {row['Feature']}: {row['Importance']:.4f}")
print()

print("==================================================")
print("TASK 4a: Gradient Boosting")
print("==================================================")
gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbc.fit(X_train_scaled, y_clf_train)
gbc_train_acc = accuracy_score(y_clf_train, gbc.predict(X_train_scaled))
gbc_test_acc = accuracy_score(y_clf_test, gbc.predict(X_test_scaled))
gbc_test_auc = roc_auc_score(y_clf_test, gbc.predict_proba(X_test_scaled)[:, 1])
print(f"Train Accuracy: {gbc_train_acc:.4f}")
print(f"Test Accuracy:  {gbc_test_acc:.4f}")
print(f"Test ROC-AUC:   {gbc_test_auc:.4f}\n")

print("==================================================")
print("TASK 4b: Feature Ablation Study")
print("==================================================")
lowest_5_features = importance_df.sort_values(by='Importance', ascending=True).head(5)['Feature'].tolist()
print(f"Lowest 5 Features to Remove: {lowest_5_features}")

# Create reduced datasets
X_train_df = pd.DataFrame(X_train_scaled, columns=feature_names)
X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
X_train_reduced = X_train_df.drop(columns=lowest_5_features).values
X_test_reduced = X_test_df.drop(columns=lowest_5_features).values

rf_reduced = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_reduced.fit(X_train_reduced, y_clf_train)
reduced_rf_test_auc = roc_auc_score(y_clf_test, rf_reduced.predict_proba(X_test_reduced)[:, 1])
print(f"Full Model Test ROC-AUC:    {rf_test_auc:.4f}")
print(f"Reduced Model Test ROC-AUC: {reduced_rf_test_auc:.4f}\n")

print("==================================================")
print("TASK 5: Cross-Validated Comparison")
print("==================================================")
lr = LogisticRegression(random_state=42, max_iter=1000)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Logistic Regression': lr,
    'Controlled DT': dt_controlled,
    'Random Forest': rf,
    'Gradient Boosting': gbc
}

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X_train_scaled, y_clf_train, cv=cv, scoring='roc_auc')
    cv_results[name] = scores
    print(f"{name} 5-Fold AUC: Mean={np.mean(scores):.4f}, Std={np.std(scores):.4f}")
print()

print("==================================================")
print("TASK 6: Hyperparameter Tuning with GridSearchCV")
print("==================================================")
# Incorporating ColumnTransformer directly into the pipeline to handle categorical data correctly
tuning_preprocessor = ColumnTransformer(
    transformers=[
        ('num', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), numeric_features),
        ('cat', make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(drop='first')), categorical_features)
    ])

pipeline = Pipeline([
    ('preprocessor', tuning_preprocessor),
    ('randomforestclassifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'randomforestclassifier__n_estimators': [50, 100, 200],
    'randomforestclassifier__max_depth': [5, 10, None],
    'randomforestclassifier__min_samples_leaf': [1, 5]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_clf_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best 5-Fold ROC-AUC Score: {grid_search.best_score_:.4f}\n")

best_pipeline = grid_search.best_estimator_

print("==================================================")
print("TASK 7: Manual Learning Curve")
print("==================================================")
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
print(f"{'Training Fraction':<20} | {'Training AUC':<15} | {'Test AUC':<15}")
print("-" * 55)

for f in fractions:
    subset_size = int(f * len(X_train))
    X_train_sub = X_train.iloc[:subset_size]
    y_train_sub = y_clf_train.iloc[:subset_size]
    
    # Re-fit the best pipeline on the subset
    best_pipeline.fit(X_train_sub, y_train_sub)
    
    train_auc = roc_auc_score(y_train_sub, best_pipeline.predict_proba(X_train_sub)[:, 1])
    test_auc = roc_auc_score(y_clf_test, best_pipeline.predict_proba(X_test)[:, 1])
    
    print(f"{f:<20} | {train_auc:<15.4f} | {test_auc:<15.4f}")
print()

print("==================================================")
print("TASK 8: Serialize the Best Model")
print("==================================================")
# Save model
best_pipeline.fit(X_train, y_clf_train) # Fit on full train
joblib.dump(best_pipeline, 'best_model.pkl')
print("Model saved to 'best_model.pkl'.")

# Reload and predict block
loaded_model = joblib.load('best_model.pkl')

# Create two hand-crafted test rows ensuring data types match
test_rows = pd.DataFrame([
    {'Age': 40, 'Purchase_Frequency': 30, 'Membership_Tier': 'Platinum', 'Satisfaction_Score': 5.0},
    {'Age': 25, 'Purchase_Frequency': 5, 'Membership_Tier': 'Bronze', 'Satisfaction_Score': 2.0}
])

predictions = loaded_model.predict(test_rows)
print(f"Predictions for hand-crafted rows: {predictions}")
print("Reload and predict block executed successfully.\n")

