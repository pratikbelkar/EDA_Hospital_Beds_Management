import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.preprocessing import PowerTransformer
#load dataset
df=pd.read_csv('patients.csv');
print(df);
# 3. Dataset overview
print('\n=== HEAD ===')
print(df.head())
print('\n=== TAIL ===')
print(df.tail())
print('\n=== SHAPE ===')
print(df.shape)
print('\n=== COLUMNS ===')
print(df.columns.tolist())
print('\n=== DTYPES ===')
print(df.dtypes)
print('\n=== MISSING VALUES (per column) ===')
print(df.isnull().sum())
print('\n=== UNIQUE VALUES (per column) ===')
print(df.nunique())
# Quick memory usage
print('\nMemory usage:')
print(df.info(memory_usage='deep'))
# 4. Data quality checks

# 4.1 Missing values summary
missing_summary = df.isnull().sum().sort_values(ascending=False)
print('\nColumns with missing values:\n', missing_summary[missing_summary > 0])
# 4.2 Duplicate rows
dup_count = df.duplicated().sum()
print('\nNumber of duplicate rows:', dup_count)
#4.3 Erroneous data examples (generic checks)
# - Negative values where not possible: attempt numeric columns that logically shouldn't be negative
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
print('\nNumeric columns detected:', numeric_cols)
# Example check: negative values
negatives = {}
for c in numeric_cols:
 neg = (df[c] < 0).sum()
if neg > 0:
 negatives[c] = int(neg)
print('\nColumns with negatives (count):', negatives)


# 4.4 Formatting issues: leading/trailing spaces, inconsistent cases
str_cols = df.select_dtypes(include=['object']).columns.tolist()
format_issues = {}
for c in str_cols:
 sample = df[c].dropna().astype(str).head(50)
has_space = sample.str.startswith(' ').any() or sample.str.endswith(' ').any()
mixed_case = sample.str.islower().any() and sample.str.isupper().any()
if has_space or mixed_case:
 format_issues[c] = {'leading/trailing_space': bool(has_space), 'mixed_case_samples': bool(mixed_case)}
print('\nString columns with possible formatting issues:', format_issues)
# 5. Data cleaning
# ---------------------------
# This section provides examples. Adapt decisions to your dataset and justify choices in README.
df_clean = df.copy()


# 5.1 Handle missing data
# Strategy: For numeric cols -> median imputation if < 30% missing, else drop column.
# For categorical -> mode imputation if < 30% missing, else fill with 'Unknown' or drop.


threshold_col_drop = 0.6 # drop columns with >60% missing
for c in df_clean.columns:
 miss_frac = df_clean[c].isnull().mean()
if miss_frac > threshold_col_drop:
 print(f'Dropping column {c} because {miss_frac:.2%} values are missing')
df_clean.drop(columns=[c], inplace=True)


# Impute remaining
for c in df_clean.select_dtypes(include=['float64', 'int64']).columns:
    if df_clean[c].isnull().sum() > 0:
        median = df_clean[c].median()
        df_clean[c].fillna(median, inplace=True)
        print(f'Imputed numeric column {c} with median={median}')

for c in df_clean.select_dtypes(include=['object']).columns:
    if df_clean[c].isnull().sum() > 0:
        mode_val = df_clean[c].mode(dropna=True)
        if mode_val.size > 0:
            fill_val = mode_val[0]
        else:
            fill_val = 'Unknown'
        df_clean[c].fillna(fill_val, inplace=True)
        print(f'Imputed categorical column {c} with mode={fill_val}')
# 5.2 Remove duplicate rows
before_dup = df_clean.shape[0]
df_clean.drop_duplicates(inplace=True)
after_dup = df_clean.shape[0]
print(f'Dropped {before_dup - after_dup} duplicate rows')
# 5.3 Correct or remove incorrect values (example: ages <=0)
# If dataset has an 'age' column, fix negatives or zeros
if 'age' in df_clean.columns:
 invalid_age = df_clean[(df_clean['age'] <= 0) | (df_clean['age'] > 120)].shape[0]
print('Invalid age rows count:', invalid_age)
# Option: set them to median
median_age = df_clean['age'].median()
df_clean.loc[(df_clean['age'] <= 0) | (df_clean['age'] > 120), 'age'] = median_age
print('Replaced invalid ages with median_age =', median_age)


# 5.4 Standardize formatting for string columns
for c in df_clean.select_dtypes(include=['object']).columns:
  df_clean[c] = df_clean[c].astype(str).str.strip()
# optional: lower-case everything for categorical features
# df_clean[c] = df_clean[c].str.lower()
#6. Descriptive statistics
# ---------------------------
print('\n=== Descriptive statistics (numeric) ===')
print(df_clean.describe().T)
# Mean, median, mode (numeric)
for c in df_clean.select_dtypes(include=['float64', 'int64']).columns:
    series = df_clean[c].dropna()
    if series.size == 0:
        continue
    mean = series.mean()
    median = series.median()
    mode = series.mode()
    var = series.var()
    std = series.std()
    skew = series.skew()
    kurt = series.kurtosis()
    print(f"\nColumn: {c} -> mean={mean:.4f}, median={median:.4f}, mode={(list(mode)[:5])}, "
          f"min={series.min()}, max={series.max()}, var={var:.4f}, std={std:.4f}, "
          f"skew={skew:.4f}, kurt={kurt:.4f}")
# Value counts for categorical data
for c in df_clean.select_dtypes(include=['object']).columns:
 print(f"\nValue counts for {c}:\n", df_clean[c].value_counts().head(10))
 # 7. Data Transformation & Encoding
# ---------------------------
# Example: encoding categorical columns and scaling numeric features
encoded_df = df_clean.copy()
# 7.1 Encoding: If ordinal, use LabelEncoder; else, One-Hot encode if low cardinality
cat_cols = encoded_df.select_dtypes(include=['object']).columns.tolist()
low_cardinality = [c for c in cat_cols if encoded_df[c].nunique() <= 10]
high_cardinality = [c for c in cat_cols if encoded_df[c].nunique() > 10]
print('\nLow cardinality categorical columns (will one-hot):', low_cardinality)
print('High cardinality categorical columns (consider target encoding or label):', high_cardinality)


# One-hot encode low-card cols
encoded_df = pd.get_dummies(encoded_df, columns=low_cardinality, drop_first=True)


# Label encode a high cardinality column example (if needed)
for c in high_cardinality:
    le = LabelEncoder()
    try:
        encoded_df[c] = le.fit_transform(encoded_df[c].astype(str))
        print(f'Label encoded column {c}')
    except Exception as e:
        print('Could not label encode', c, e)


# 7.2 Scaling
num_cols_after = encoded_df.select_dtypes(include=['float64','int64']).columns.tolist()
scaler = StandardScaler()
encoded_df[num_cols_after] = scaler.fit_transform(encoded_df[num_cols_after])
print('\nApplied StandardScaler to numeric columns')


from sklearn.preprocessing import PowerTransformer

# Initialize transformer
pt = PowerTransformer(method='yeo-johnson')

# Find skewed numeric columns
skewed_cols = []
for c in df_clean.select_dtypes(include=['float64', 'int64']).columns:
    series = df_clean[c].dropna()
    if series.nunique() > 2 and abs(series.skew()) > 1:
        skewed_cols.append(c)

# Apply transform only if skewed columns exist
if skewed_cols:
    print('Applying power transform to:', skewed_cols)
    encoded_df[skewed_cols] = pt.fit_transform(df_clean[skewed_cols])
else:
    print("No skewed numeric columns found — skipping PowerTransformer.")
# 8. Outlier detection & treatment
# ---------------------------
# IQR method
outlier_report = {}
for c in df_clean.select_dtypes(include=['float64','int64']).columns:
 Q1 = df_clean[c].quantile(0.25)
Q3 = df_clean[c].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df_clean[(df_clean[c] < lower) | (df_clean[c] > upper)].shape[0]
if outliers > 0:
 outlier_report[c] = outliers
print('\nOutlier counts by IQR method:', outlier_report)
# Z-score method
z_outliers = {}
for c in df_clean.select_dtypes(include=['float64','int64']).columns:
 z_scores = np.abs(stats.zscore(df_clean[c].dropna()))
z_out = (z_scores > 3).sum()
if z_out > 0:
 z_outliers[c] = int(z_out)
print('\nOutlier counts by Z-score method:', z_outliers)


# Treatment options (example: capping)
capped_df = df_clean.copy()
for c in df_clean.select_dtypes(include=['float64','int64']).columns:
 Q1 = df_clean[c].quantile(0.25)
Q3 = df_clean[c].quantile(0.75)
lower = Q1 - 1.5 * (Q3 - Q1)
upper = Q3 + 1.5 * (Q3 - Q1)
capped_df[c] = np.where(capped_df[c] < lower, lower, capped_df[c])
capped_df[c] = np.where(capped_df[c] > upper, upper, capped_df[c])
print('\nApplied simple capping to numeric columns (IQR-based)')
#9. Data visualization
# ---------------------------



# 9.1 Univariate: histograms and boxplots for numeric cols (showing first 6 numeric cols)
num_cols = df_clean.select_dtypes(include=['float64','int64']).columns.tolist()
for c in num_cols[:6]:
 fig, axes = plt.subplots(1,2, figsize=(12,4))
sns.histplot(df_clean[c].dropna(), kde=True, ax=axes[0])
axes[0].set_title(f'Histogram of {c}')
sns.boxplot(x=df_clean[c], ax=axes[1])
axes[1].set_title(f'Boxplot of {c}')
plt.tight_layout()
plt.show()


# 9.2 Categorical bar plots (first 4 categorical cols)
cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
for c in cat_cols[:4]:
 plt.figure(figsize=(8,4))
sns.countplot(y=c, data=df_clean, order=df_clean[c].value_counts().index[:20])
plt.title(f'Value counts for {c}')
plt.show()


# 9.3 Bivariate: scatter plot for two numeric features (pick first two)
if len(num_cols) >= 2:
 plt.figure(figsize=(6,5))
sns.scatterplot(x=df_clean[num_cols[0]], y=df_clean[num_cols[1]])
plt.title(f'Scatter: {num_cols[0]} vs {num_cols[1]}')
plt.show()
#9.4 Correlation matrix and heatmap
plt.figure(figsize=(10,8))
corr = df_clean.select_dtypes(include=['float64','int64']).corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Correlation matrix (numeric)')
plt.show()


# 9.5 Pairplot (useful for small number of columns)
if len(num_cols) <= 6 and len(num_cols) >= 2:
 sns.pairplot(df_clean[num_cols].dropna().sample(frac=1.0).iloc[:1000])
plt.show()
#10. Multivariate analysis (PCA example)
# ---------------------------
numeric_for_pca = df_clean.select_dtypes(include=['float64', 'int64']).dropna()

if numeric_for_pca.shape[1] >= 2:
    # Standardize data
    sc = StandardScaler()
    numeric_scaled = sc.fit_transform(numeric_for_pca)

    # Apply PCA (up to 5 components or number of features)
    pca = PCA(n_components=min(5, numeric_scaled.shape[1]))
    pca_result = pca.fit_transform(numeric_scaled)

    # Print explained variance
    explained = pca.explained_variance_ratio_
    print('\nPCA explained variance ratio:', explained)

    # Scatter plot of first two principal components
    if pca_result.shape[1] >= 2:
        plt.figure(figsize=(7, 6))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.6)
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA: PC1 vs PC2')
        plt.show()
else:
    print("Not enough numeric columns for PCA — skipping.")
   # 11. Save cleaned dataset for modelling / submission
# ---------------------------
try:
    df_clean.to_csv(CLEANED_OUTPUT, index=False)
    print('\nCleaned dataset saved to', CLEANED_OUTPUT)
except Exception as e:
    print('Could not save cleaned dataset:', e)

