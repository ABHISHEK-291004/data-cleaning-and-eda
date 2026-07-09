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
