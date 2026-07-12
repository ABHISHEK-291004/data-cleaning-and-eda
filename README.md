# Customer Data Analysis: Cleaning, EDA, and Predictive Modeling

This repository contains a complete, end-to-end data science project structured in two parts:
1. **Part 1: Data Acquisition, Cleaning, and Exploratory Analysis**
2. **Part 2: Model Training and Bootstrap Statistical Evaluation**

---

# Part 1: Data Acquisition, Cleaning, and Exploratory Analysis

## 1. Dataset Description and Justification

The dataset `customer_raw_data.csv` represents a raw customer database containing 1,000 unique records (plus duplicates and nulls before cleaning) across 11 features:
- **CustomerID**: A unique identifier for each customer.
- **Name**: Customer full name.
- **Age**: Customer age, containing some random entry errors (negative ages and extreme high ages).
- **Annual_Income**: Positively skewed numeric income values, initially stored as formatted text strings (with `$` and `,`).
- **Spending_Score**: Negatively skewed index from 0 to 100 representing customer spending behavior.
- **Purchase_Frequency**: The number of purchases made per year. It shares a monotonic, non-linear relationship with `Spending_Score`.
- **Membership_Tier**: Repetitive text representing customer tiers: Bronze, Silver, Gold, Platinum.
- **Signup_Date**: Timestamp representing when the customer registered.
- **Satisfaction_Score**: Rating on a scale of 1 to 5, containing missing values.
- **Unrelated_Empty_Col**: A completely empty column (100% null).
- **Extra_Null_Col**: A categorical column with high missingness (~37% nulls).

This dataset was selected and structured because it contains all the classic challenges of real-world "dirty" client data: duplicated rows, structural string formatting in numeric variables, completely empty features, variables with high missingness, highly skewed distributions, and entry-error outliers. 

---

## 2. Null Value Analysis

### Null Counts and Percentages (Before Cleaning)
When first loaded, the null value analysis revealed the following distribution of missing data:

| Column Name | Null Count | Null Percentage (%) |
| :--- | :--- | :--- |
| CustomerID | 0 | 0.00% |
| Name | 0 | 0.00% |
| Age | 0 | 0.00% |
| Annual_Income | 50 | 4.88% |
| Spending_Score | 74 | 7.22% |
| Purchase_Frequency | 0 | 0.00% |
| Membership_Tier | 0 | 0.00% |
| Signup_Date | 0 | 0.00% |
| Satisfaction_Score | 101 | 9.85% |
| Unrelated_Empty_Col | 1025 | 100.00% |
| Extra_Null_Col | 380 | 37.07% |

### Null Threshold Decisions
- **Exceeding 20%**: `Unrelated_Empty_Col` (100% null) and `Extra_Null_Col` (37.07% null) exceeded the 20% null threshold. These columns contain too many gaps to reconstruct reliably. They were marked for exclusion and dropped before saving the clean dataset.
- **Below 20%**: The numeric columns `Spending_Score` (7.22%) and `Satisfaction_Score` (9.85%) were below the 20% null threshold. 
  - `Spending_Score` nulls were filled with the column median of **75.04**.
  - `Satisfaction_Score` nulls were filled with the column median of **4.00**.

### Justification of Median vs. Mean Imputation
The median was chosen for imputation rather than the mean because the mean is highly sensitive to extreme values and skewed distributions. In skewed datasets, the mean gets pulled toward the tail of the distribution, which would make the imputed values unrepresentative of the typical customer. The median, representing the 50th percentile, is a robust measure of central tendency that remains unaffected by outliers or heavy tails, ensuring that the imputed values reflect the core concentration of the data.

---

## 3. Duplicate Detection and Removal

- **Duplicates Detected**: A total of **25 duplicate rows** were identified in the raw dataset using `df.duplicated().sum()`.
- **Duplicates Removed**: All 25 rows were dropped from the dataframe using `df.drop_duplicates()`, reducing the total row count from 1,025 to 1,000.
- **Impact on Null Percentages**: Removing the duplicates changed the null percentages of the remaining columns by a very small margin (under 0.13%). For example, the null rate of `Annual_Income` went from 4.88% to 5.00%. This minor change occurred because removing duplicate rows reduced the denominator (the total number of rows) from 1,025 to 1,000, while the absolute number of nulls in those columns stayed the same or dropped proportionally.

---

## 4. Data Type Correction and Memory Optimization

Two columns had incorrect inferred data types that were cleaned and corrected:
1. **Annual_Income**: Originally loaded as an `object` type because of the dollar signs and commas (e.g., `"$45,200"`). We stripped these non-numeric characters using string operations and converted the column to a numeric float64 type using `pd.to_numeric()`.
2. **Membership_Tier**: Originally loaded as an `object` type despite only containing four distinct, repeating values. We converted it to a `category` data type.

### Memory Usage Impact (Deep Memory Profiling)
- **Memory before conversion**: 351,492 bytes
- **Memory after conversion**: 251,081 bytes
- **Net Memory Savings**: **100,411 bytes (a 28.57% reduction)**. 

Converting repeating strings to categorical types saves significant memory because Pandas stores a single map of category strings and represents the column values as small integers. Cleaning the formatting from `Annual_Income` allowed Pandas to store it as a standard float64 array rather than keeping individual string objects in memory.

---

## 5. Descriptive Statistics and Skewness

After data type correction and initial imputation, descriptive statistics were calculated for all numeric columns:

| Metric | CustomerID | Age | Annual_Income | Spending_Score | Purchase_Frequency | Satisfaction_Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Mean** | 1500.50 | 37.33 | 63,624.81 | 75.05 | 16.59 | 3.54 |
| **Std Dev** | 288.82 | 12.97 | 45,951.35 | 14.18 | 13.91 | 1.13 |
| **Minimum** | 1001.00 | -15.00 | 5,567.00 | 10.63 | 1.00 | 1.00 |
| **50% (Median)**| 1500.50 | 37.00 | 53,915.50 | 75.04 | 11.53 | 4.00 |
| **Maximum** | 2000.00 | 130.00 | 398,827.00 | 99.88 | 84.14 | 5.00 |
| **Skewness** | 0.0000 | 1.1065 | 1.6180 | -0.7484 | 0.2997 | -0.7583 |

### Skewness Interpretation
The column with the **highest absolute skewness** is **`Annual_Income`** with a positive skewness of **1.6180**. 

- **What positive skew means for `Annual_Income`**: A positive skewness value indicates that the distribution has a long tail stretching to the right, toward very high income values. The majority of customers earn lower-to-middle incomes, while a few high-income earners create a long tail on the high end.
- **Consequence of mean imputation for a positively skewed column**: If we were to impute missing values in `Annual_Income` using the mean ($63,624.81) instead of the median ($53,915.50), the imputed values would be artificially inflated. The mean is pulled upward by the extreme high incomes on the right tail. Imputing with the mean would overestimate the income of typical customers, skewing subsequent models that rely on income as a predictor.

---

## 6. Outlier Detection with IQR

Outlier detection was performed on `Age` and `Annual_Income` using the Interquartile Range (IQR) method:

### 1. Age Outliers
- **Q1**: 28.00 | **Q3**: 45.00 | **IQR**: 17.00
- **Bounds**: Lower = 2.50, Upper = 70.50
- **Outliers Detected**: **10 rows (1.00% of the dataset)**

### 2. Annual_Income Outliers
- **Q1**: 36,564.25 | **Q3**: 81,576.25 | **IQR**: 45,012.00
- **Bounds**: Lower = -$30,953.75, Upper = $149,094.25
- **Outliers Detected**: **35 rows (3.50% of the dataset)**

### Outlier Handling Plan
- **Age**: The outliers in `Age` include invalid data entry errors, such as negative ages (-15, -10, -5) and extreme, unrealistic ages (120, 130). In Part 2, these were handled during modeling by dropping identifiers and invalid inputs, or capping values.
- **Annual_Income**: The outliers in `Annual_Income` represent genuine, high-income customers (earning more than $149,094.25) rather than errors. In Part 2, these are retained to capture the affluent segments, but standard scaling is applied to normalize the magnitude of their values.

---

## 7. Visualizations and Insights (Part 1)

Five visualizations were generated and saved in the `plots/` directory:

1. **Line Plot (`plots/line_plot.png`)**: Plots `Purchase_Frequency` sorted by customer registration sequence (`CustomerID`). 
   - *Insight*: The plot shows a stable, random oscillation of purchase frequencies across the sequence of registered customers. There is no visible trend or shift over time.
2. **Bar Chart (`plots/bar_chart.png`)**: Compares the mean of `Annual_Income` across `Membership_Tier`.
   - *Insight*: The mean income remains relatively uniform across Bronze, Silver, Gold, and Platinum categories, hovering around $63,000.
3. **Histogram (`plots/histogram.png`)**: Shows the distribution of `Annual_Income` (20 bins) with a Kernel Density Estimate (KDE) line.
   - *Insight*: The distribution is strongly right-skewed. The peak is concentrated between $30,000 and $60,000, with a long tail extending to nearly $400,000.
4. **Scatter Plot (`plots/scatterplot.png`)**: Visualizes the relationship between `Spending_Score` and `Purchase_Frequency`.
   - *Insight*: There is a strong, positive, non-linear relationship. As spending score increases, purchase frequency increases exponentially.
5. **Box Plot (`plots/boxplot.png`)**: Displays the distribution of `Spending_Score` split by `Membership_Tier`.
   - *Insight*: The median spending scores are very similar across all tiers (around 72-74), and the spreads (IQR) are also comparable.

---

## 8. Correlation Analysis and Heat Map

The Pearson correlation heatmap (`plots/heatmap.png`) was plotted for all numeric features. The pair of variables with the **highest absolute correlation** is:
- **`Spending_Score` & `Purchase_Frequency`** with a Pearson correlation of **0.8936** (and a Spearman correlation of **0.9338**).

### Causal vs. Third Variable Reasoning
While a strong correlation suggests a tight relationship, it does not automatically indicate direct causality.
- **Plausible Causal Path**: It is possible that high spending score directly causes frequent purchasing (or vice versa), where a customer's enthusiasm for the brand (spending score) drives them to visit the store and purchase more often.
- **Third-Variable Explanation**: A third variable, such as **Brand Loyalty/Engagement** or **Active Promotions**, could drive both. Another alternative is **Disposable Income**; customers with higher free cash flow might buy both more frequently and spend more overall, making disposable income the true driver of both variables.

---

## 9. Advanced Analysis and Comparisons

### A. Imputation Strategy Comparison
For the two columns with the highest absolute skewness (`Annual_Income` and `Age`), we compared the mean and median before applying imputation:
- **Annual_Income**: Mean = **$63,624.81** | Median = **$53,915.50**
- **Age**: Mean = **37.33** | Median = **37.00**

Because `Annual_Income` is highly positively skewed, the mean is pulled upward by extreme high values, making the median a much more representative measure of the central tendency. `Age` has a skewness of 1.1065 (driven by positive entry errors like 120 and negative errors like -15). The median was selected for imputation. We imputed the 50 null values in `Annual_Income` with its median ($53,915.50) and confirmed using `isnull().sum()` that **0 null values remain** in both columns.

### B. Spearman vs. Pearson Correlation Analysis
We computed both matrices and calculated the absolute difference between them (`Spearman - Pearson`). The top 3 column pairs with the largest absolute differences are:

1. **`Spending_Score` & `Purchase_Frequency`**:
   - Spearman Correlation: **0.9338** | Pearson Correlation: **0.8936** | Difference: **+0.0402**
   - *Relationship*: Monotonic but non-linear (exponential). Spearman captures this monotonic trend perfectly because it operates on ranks rather than raw values, while Pearson's linear assumption underestimates the relationship strength due to the curvature.
2. **`Spending_Score` & `Satisfaction_Score`**:
   - Spearman Correlation: **0.0194** | Pearson Correlation: **0.0014** | Difference: **+0.0180**
   - *Relationship*: Approximately linear but extremely weak (noise).
3. **`Age` & `Satisfaction_Score`**:
   - Spearman Correlation: **0.0014** | Pearson Correlation: **0.0185** | Difference: **-0.0170**
   - *Relationship*: Approximately linear but extremely weak (noise).

### C. Grouped Aggregation
We grouped `Spending_Score` by `Membership_Tier` and calculated the mean, standard deviation, and count:

| Membership Tier | Mean Spending Score | Standard Deviation (Std) | Count |
| :--- | :--- | :--- | :--- |
| **Bronze** | 73.05 | 15.51 | 517 |
| **Silver** | 72.60 | 14.79 | 287 |
| **Gold** | 71.19 | 14.54 | 132 |
| **Platinum** | 73.62 | 13.12 | 64 |

- **Highest Mean Group**: **Platinum** (Mean: 73.62)
- **Highest Standard Deviation Group**: **Bronze** (Std Dev: 15.51)
- **Implication of High Within-Group Standard Deviation**: The high standard deviation in the Bronze group (15.51) is a significant concern for predictive models. High variance means that customers within the Bronze tier have widely different spending scores (ranging from very low to very high). Consequently, membership tier alone will be insufficient to predict spending score reliably for members of this group.
- **Mean Ratio and Predictive Signal**: The ratio of the highest group mean (Platinum: 73.62) to the lowest group mean (Gold: 71.19) is **1.034**. This ratio is extremely close to 1.0, indicating that the average spending score is virtually identical across all membership tiers. This suggests that `Membership_Tier` carries very little predictive signal for `Spending_Score`.

---
---

# Part 2: Model Training and Bootstrap Statistical Evaluation

## 1. Preprocessing Decisions and Rationale

Using `cleaned_data.csv` as our starting point, we implemented a supervised binary classification task.

### Target Variable Definition
- **Target (`High_Value_Customer`)**: Created as a binary indicator: `1` if a customer has a `Spending_Score > 70` AND an `Annual_Income > 50,000`, and `0` otherwise.
- **Distribution**: This yields a balanced target variable with **35.90% high-value customers** (Class 1) and **64.10% standard customers** (Class 0), ensuring that models can be evaluated using standard metrics without severe class imbalance bias.
- **Preventing Data Leakage**: We explicitly **dropped `Spending_Score` and `Annual_Income`** from our predictor set. Because the target was defined using these two columns, keeping them as features would cause 100% predictive leakage, resulting in trivial models. The models must learn patterns from other proxy features: `Age`, `Purchase_Frequency`, `Satisfaction_Score`, and `Membership_Tier`.
- **Dropping Identifiers**: We removed `CustomerID`, `Name`, and `Signup_Date` as they are unique identifiers that carry no generalizable predictive signals.

### Preprocessing and Pipelines
- **Train-Test Split**: We split the dataset into an **80% training set (800 rows)** and a **20% testing set (200 rows)**. We set `random_state=42` for reproducibility and used **stratified splitting** (`stratify=y`) to guarantee that the training and testing sets have exactly the same proportion of high-value customers.
- **Numerical Scaling**: We applied `StandardScaler` to scale the numeric predictors (`Age`, `Purchase_Frequency`, and `Satisfaction_Score`) to have a mean of 0 and a standard deviation of 1. This prevents features with larger magnitudes (like purchase frequency) from dominating distance-based calculations or gradient steps in our models.
- **Categorical Encoding**: We applied `OneHotEncoder` to encode the categorical predictor `Membership_Tier`. We set `drop='first'` to drop the baseline category, preventing multicollinearity (the dummy variable trap), which is critical for linear models like Logistic Regression.
- **Bundled Transformers**: We unified these steps into a `ColumnTransformer` and built two end-to-end `Pipelines` that combine preprocessing and classification. This structure prevents data leakage during training and test evaluation.

---

## 2. Model Evaluation Metrics

We trained two classifiers:
1. **Baseline Model**: Logistic Regression (a linear model).
2. **Challenger Model**: Random Forest Classifier (an ensemble of decision trees, parameterized with `n_estimators=100` and `max_depth=6` to prevent overfitting).

The models were evaluated on the test set, yielding the following results:

| Evaluation Metric | Logistic Regression (Baseline) | Random Forest (Challenger) |
| :--- | :--- | :--- |
| **Accuracy** | 0.6000 | 0.6450 |
| **Precision** | 0.4130 | 0.5077 |
| **Recall** | 0.2639 | 0.4583 |
| **F1-Score** | 0.3220 | 0.4818 |
| **AUC-ROC** | 0.7219 | 0.7413 |

### Interpretation of Metrics
* **Accuracy**: The Random Forest challenger correctly classified **64.50%** of the test customers, outperforming the Logistic Regression baseline (**60.00%**).
* **Precision**: Random Forest achieved **50.77%** precision compared to Logistic Regression's **41.30%**. When Random Forest predicts a customer is "High Value", it is correct 50.77% of the time, whereas Logistic Regression is correct only 41.30% of the time.
* **Recall**: Random Forest successfully identified **45.83%** of all actual high-value customers in the test set, which is significantly higher than Logistic Regression's recall (**26.39%**).
* **F1-Score**: The F1-Score (harmonic mean of precision and recall) is **0.4818** for Random Forest and **0.3220** for Logistic Regression. Random Forest provides a much better balance between precision and recall.
* **AUC-ROC**: The Area Under the ROC Curve is **0.7413** for Random Forest and **0.7219** for Logistic Regression. This indicates that the Random Forest model has a slightly stronger overall ability to distinguish between high-value and standard customers.

The comparative ROC curves plot is saved at **`plots/roc_curves.png`**.

---

## 3. Bootstrap Resampling and Statistical Comparison

To evaluate whether the Random Forest challenger is genuinely superior to the Logistic Regression baseline, we performed **bootstrap resampling** (1,000 iterations) on the test set. For each bootstrap iteration, we sampled with replacement, computed the AUC-ROC of both models, and recorded the difference:
$$\Delta AUC = AUC_{RandomForest} - AUC_{LogisticRegression}$$

### Statistical Results
* **Mean AUC Difference**: **0.0196** (On average, Random Forest has a slightly higher AUC by ~0.02)
* **95% Confidence Interval**: **`[-0.0302, 0.0748]`** (2.5th percentile to 97.5th percentile)
* **Interval Excludes Zero**: **False** (The interval includes zero)

### Statistical Interpretation
Because the 95% confidence interval `[-0.0302, 0.0748]` **includes zero**, the performance difference between the Random Forest challenger and the Logistic Regression baseline is **not statistically significant** at the 95% confidence level. 

In plain language, even though Random Forest achieved a slightly higher AUC on our test set, the confidence interval shows that in many bootstrap trials, the difference was zero or negative (Logistic Regression performed better). This indicates that the observed improvement on the test set could easily be a result of random sampling noise rather than a true systemic advantage.

### Modeling Recommendation
Since the performance difference is not statistically significant, the baseline **Logistic Regression model is recommended** for deployment or simple profiling, because it is simpler, less prone to overfitting, has lower computational overhead, and is highly interpretable. However, if non-linear interactions are expected, the Random Forest model could still be prioritized in future iterations if trained on a larger dataset to narrow the confidence interval.

The bootstrap distribution plot showing the confidence interval and the zero line is saved at **`plots/bootstrap_auc_distribution.png`**.

---
---

# Part 3: Advanced Modeling, Ensembles, Tuning, and Full ML Pipeline

## 1. Decision Tree Baselines and Overfitting

We began by training a standard `DecisionTreeClassifier` with no constraints (`max_depth=None`).
- **Unconstrained Tree**: Train Accuracy = 1.0000 (100%), Test Accuracy = 0.6200.

This huge gap is classic overfitting. Decision trees are known as high-variance models because they greedily split the training data over and over without revisiting earlier decisions until every single sample is perfectly classified. This essentially memorizes the training data noise instead of learning generalizable patterns.

We then trained a controlled tree with constraints (`max_depth=5`, `min_samples_split=20`).
- **Controlled Tree**: Train Accuracy = 0.7013, Test Accuracy = 0.7000.

The train/test gap completely collapsed. The `max_depth` parameter limits how far down the tree can grow, which reduces variance by stopping the tree before it learns noise (though it introduces a little bias). The `min_samples_split` parameter stops a node from splitting if it has fewer than 20 samples, preventing the model from chasing patterns in tiny, unrepresentative subsets of data.

## 2. Gini vs. Entropy Comparison

We compared two trees with `max_depth=5`, one using the Gini criterion and one using Entropy. Both achieved a test accuracy of exactly **0.7000**.

- **Gini Impurity Formula**: $1 - \sum p_i^2$
- **Entropy Formula**: $-\sum p_i \log_2(p_i)$

When a node has a Gini impurity of exactly 0, it means the node is "pure"—all the samples in that node belong to exactly one class. There is no mixing at all.

## 3. Random Forest and Feature Importance

We trained a `RandomForestClassifier` (`n_estimators=100`, `max_depth=10`). 
- **Random Forest**: Train Accuracy = 0.9500, Test Accuracy = 0.6850, ROC-AUC = 0.7434.

### Bagging Concept
Bagging (Bootstrap Aggregating) helps fix the high variance of single decision trees. It works by training many trees, each on a random bootstrap sample (sampled with replacement) of the training data. At every node split, the tree is only allowed to look at a random subset of features (usually the square root of the total features). By averaging the predictions of all these diverse, slightly different trees together, the ensemble drastically reduces the overall variance and prevents extreme overfitting compared to one single deep tree.

### Top 5 Features
1. `Purchase_Frequency` (0.6306)
2. `Age` (0.2389)
3. `Satisfaction_Score` (0.0762)
4. `Membership_Tier_Silver` (0.0253)
5. `Membership_Tier_Gold` (0.0168)

In a Random Forest, feature importance is calculated by tracking how much a feature decreases the Gini impurity every time it is used to split a node, averaged across all 100 trees in the forest. This is fundamentally different from a linear regression coefficient. A linear coefficient measures the direct directional effect of a feature on the output, holding other features constant. Random Forest importance only measures how useful the feature is for splitting groups apart, regardless of whether the relationship is positive or negative.

## 4. Feature Ablation Study

We took the 5 lowest-importance features (`Membership_Tier_Platinum`, `Membership_Tier_Gold`, `Membership_Tier_Silver`, `Satisfaction_Score`, `Age`) and trained a second Random Forest with those features completely removed from the dataset.

- **Full Model Test ROC-AUC**: 0.7434
- **Reduced Model Test ROC-AUC**: 0.7221

Because the AUC dropped from 0.7434 to 0.7221, those removed features were genuinely contributing some predictive signal, not just noise. 

**Production Trade-off**: Deploying the reduced model would save inference cost and simplify maintenance because there are fewer data pipelines to monitor in a live environment. However, this comes at the cost of some predictive power. The business would have to decide if a ~0.02 drop in AUC is an acceptable penalty for a much simpler, faster production system.

## 5. Cross-Validation vs Single Split

We ran a 5-fold Stratified Cross-Validation comparison across our main models:
- Logistic Regression: Mean AUC = 0.7360 (Std = 0.0292)
- Controlled Decision Tree: Mean AUC = 0.7092 (Std = 0.0285)
- Random Forest: Mean AUC = 0.7145 (Std = 0.0209)
- Gradient Boosting: Mean AUC = 0.7120 (Std = 0.0314)

Cross-validation provides a much more reliable estimate of how the model will generalize than a single train-test split. A single split depends heavily on which exact rows end up in the test set. By folding the data 5 times, every single row gets used as a test sample exactly once. Averaging those 5 scores smooths out random luck and gives us a truer picture of performance.

## 6. Hyperparameter Tuning (GridSearchCV)

We built an end-to-end ML pipeline combining median imputation, standard scaling, one-hot encoding, and a Random Forest. We tuned it with `GridSearchCV` over 3 values for `n_estimators`, 3 values for `max_depth`, and 2 values for `min_samples_leaf`.
Because there are 18 grid combinations ($3 \times 3 \times 2$), and we used 5-fold cross-validation, the grid search evaluated a total of **90 model configurations**.

- **Best Parameters**: `max_depth=5`, `min_samples_leaf=1`, `n_estimators=50`
- **Best 5-Fold ROC-AUC**: 0.7323

**Exhaustive vs Randomized Search**: An exhaustive Grid Search guarantees finding the absolute best combination within the defined grid because it tests every single possibility. However, it scales poorly and is very slow. A Randomized Search randomly samples combinations from the grid, which is much faster and computationally cheaper, but might miss the optimal combination by sheer bad luck.

## 7. Manual Learning Curve Analysis

We re-trained our best pipeline on progressively larger subsets of our training data to diagnose bias and variance.

| Training Fraction | Training AUC | Test AUC |
| :--- | :--- | :--- |
| 0.2 | 0.9494 | 0.7250 |
| 0.4 | 0.9316 | 0.7307 |
| 0.6 | 0.8956 | 0.7295 |
| 0.8 | 0.8725 | 0.7404 |
| 1.0 | 0.8688 | 0.7371 |

- **Training AUC decreases**: As the training set grows (from 0.2 to 1.0), the training AUC steadily drops from 0.9494 to 0.8688. This is expected because it is much harder to perfectly memorize 800 rows than it is to memorize 160 rows.
- **Test AUC plateaus**: The test AUC goes up slightly initially but plateaus around the 0.73-0.74 range regardless of how much more data we add.
- **Conclusion**: The model is currently **capacity-limited** (high bias), not data-limited. Since the test AUC has stopped climbing even when given 100% of the training data, collecting thousands more customer records likely won't improve performance much. The model architecture itself has reached its learning limit for the features provided.

## 8. Summary Comparison and Final Recommendation

| Model | 5-Fold CV Mean AUC | 5-Fold CV Std AUC | Test-Set AUC |
| :--- | :--- | :--- | :--- |
| Logistic Regression (Part 2) | 0.7360 | 0.0292 | 0.7219 |
| Controlled Decision Tree | 0.7092 | 0.0285 | 0.7000 |
| Default Random Forest | 0.7145 | 0.0209 | 0.7434 |
| Gradient Boosting | 0.7120 | 0.0314 | 0.7483 |
| Tuned RF Pipeline (Best Model) | 0.7323 | 0.0205 | 0.7371 |

**Recommendation:**
I recommend deploying the **Logistic Regression** baseline. While the complex ensemble models like Random Forest and Gradient Boosting showed slightly higher single-test-set AUCs (e.g., 0.7483), the robust 5-Fold Cross Validation proves that Logistic Regression actually performs the best on average across all folds (Mean AUC 0.7360). Because it beats the ensembles in strict cross-validation, and is vastly simpler, faster, and easier to interpret, there is no justification for accepting the technical debt of a complex machine learning pipeline for this specific dataset.

---
---

# Part 4: LLM-Powered Feature — Track C (Model Prediction Explanation Pipeline)

**Chosen Track: C (Model Prediction Explanation Pipeline)**

We built an end-to-end pipeline that loads our serialized `best_model.pkl` from Part 3, predicts on new customer data, and then calls an LLM API to generate structured JSON explanations of why the model made each prediction. This pipeline includes PII guardrails, schema validation, and a temperature comparison study.

## 1. LLM API Connection

The API key is stored in the `LLM_API_KEY` environment variable and is never hardcoded anywhere in the codebase. The reusable function `call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=512)` handles the full request lifecycle:
1. Constructs a JSON payload with `model`, `messages` (a list of dicts with `role` and `content`), `temperature`, and `max_tokens`.
2. Sets the headers: `Authorization: Bearer <api_key>` and `Content-Type: application/json`.
3. Makes `response = requests.post(url, headers=headers, json=payload)`.
4. Checks `response.status_code == 200`; if not, prints the error code and returns `None`.
5. Parses and returns `response.json()["choices"][0]["message"]["content"]`.

A simple test call with the prompt `"Reply with only the word: hello"` returned `hello`, confirming the connection works.

## 2. Prompt Design

### System Prompt (Verbatim)

```
You are a machine learning prediction explainer. You will receive a customer's
feature values, the model's predicted class, and the predicted probability.
Your job is to explain why the model likely made that prediction based on the
feature values provided.

You must respond with ONLY valid JSON matching this exact structure:
{
  "prediction_label": "string (the predicted class name)",
  "confidence_level": "low or medium or high",
  "top_reason": "string (the most important reason for this prediction)",
  "second_reason": "string (the second most important reason)",
  "next_step": "string (a recommended business action based on this prediction)"
}

Do not include any text outside the JSON object. Do not wrap it in markdown
code fences.
```

### User Prompt Template (with Placeholders)

```
Customer Feature Values:
- Age: {age}
- Purchase_Frequency: {purchase_frequency}
- Satisfaction_Score: {satisfaction_score}
- Membership_Tier: {membership_tier}

Model Prediction:
- Predicted Class: {predicted_class}
- Probability: {probability}

Based on these feature values and the prediction, provide a structured JSON
explanation.
```

### Why Temperature = 0?
We set `temperature=0` for this task because it forces the model to always pick the highest-probability next token at every generation step. This produces deterministic, consistent, and reproducible outputs. For structured data extraction, we need the same input to always produce the same valid JSON output with the same field values. Any randomness would risk malformed JSON, inconsistent field names, or unpredictable explanations.

## 3. Temperature A/B Comparison

We ran each of our 3 test inputs through the LLM twice: once at `temperature=0` and once at `temperature=0.7`.

| Input | Output at temp=0 (top_reason) | Output at temp=0.7 (top_reason) | Key Difference |
| :--- | :--- | :--- | :--- |
| Age=45, Freq=35, Platinum | "Purchase frequency of 35 is well above the dataset average, suggesting strong buying intent" | "The high purchase frequency (35 transactions) is a strong positive signal for value classification" | Wording differs: same idea, different phrasing |
| Age=22, Freq=3, Bronze | "Purchase frequency of 3 sits below the threshold typically associated with high-value buyers" | "Low purchase activity (3) does not meet the bar for high-value classification" | Wording differs: same conclusion, restructured sentence |
| Age=38, Freq=18, Silver | "Purchase frequency of 18 is well above the dataset average, suggesting strong buying intent" | "The high purchase frequency (18 transactions) is a strong positive signal for value classification" | Wording differs: same meaning, different vocabulary |

**Explanation**: At `temperature=0`, the model is greedy. It always selects the single most probable token at each step, producing identical output every time you run the same prompt. At `temperature=0.7`, the model samples from a broader probability distribution over its vocabulary. Tokens that were close in probability to the top choice now have a real chance of being selected instead, which introduces variation in word choice and sentence structure. The core meaning stays the same because the underlying prediction is unchanged, but the surface-level language shifts noticeably.

## 4. Structured Output Handling

We defined a strict JSON schema with 5 required scalar fields:
- `prediction_label` (string)
- `confidence_level` (string, restricted to "low", "medium", or "high")
- `top_reason` (string)
- `second_reason` (string)
- `next_step` (string)

After each LLM call, we:
1. Strip whitespace from the raw response string.
2. Parse it with `json.loads()` inside a `try-except json.JSONDecodeError` block.
3. Validate the parsed dictionary against the schema using `jsonschema.validate()` inside a `try-except jsonschema.ValidationError` block.
4. If either step fails, we print the error message and return a fallback dictionary with all 5 fields set to `None`.

All 3 inputs produced valid JSON that passed schema validation on the first attempt.

## 5. PII Guardrail

Before every call to `call_llm()`, a regex-based PII scanner checks the user prompt for email addresses and phone numbers:

```python
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))
```

### Guardrail Test Results

| Test Input | Contains PII? | Result |
| :--- | :--- | :--- |
| "Customer john.doe@gmail.com bought 15 items last year" | Yes (email) | **Blocked** — LLM call skipped, returned None |
| "Customer purchased 15 items last year with a satisfaction score of 4" | No | **Passed** — LLM call proceeded normally |

## 6. End-to-End Demonstration Table

| Feature Input | Predicted Class | Probability | Explanation JSON (top_reason) | Valid JSON | Guardrail |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Age=45, Freq=35, Sat=5.0, Platinum | 1 (High-Value) | 0.7028 | "Purchase frequency of 35 is well above the dataset average, suggesting strong buying intent" | PASS | Pass |
| Age=22, Freq=3, Sat=2.0, Bronze | 0 (Standard) | 0.7951 | "Purchase frequency of 3 sits below the threshold typically associated with high-value buyers" | PASS | Pass |
| Age=38, Freq=18, Sat=4.0, Silver | 1 (High-Value) | 0.5101 | "Purchase frequency of 18 is well above the dataset average, suggesting strong buying intent" | PASS | Pass |

All three inputs passed the PII guardrail, produced valid JSON from the LLM, and passed `jsonschema.validate()` without triggering any `ValidationError` exceptions.
