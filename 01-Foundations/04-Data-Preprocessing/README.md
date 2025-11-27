# 🔧 Data Preprocessing

## 🎯 Learning Objectives
- Master data cleaning and transformation techniques
- Handle missing values and outliers effectively
- Perform feature engineering and selection
- Prepare data for machine learning models

## 📊 Data Exploration

### Initial Data Assessment
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load and explore data
df = pd.read_csv('dataset.csv')

# Basic information
print("Dataset Shape:", df.shape)
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nBasic Statistics:")
print(df.describe())

# First few rows
print("\nFirst 5 rows:")
print(df.head())
```

### Data Quality Assessment
```python
def assess_data_quality(df):
    """Comprehensive data quality assessment"""
    quality_report = {}
    
    # Missing values
    missing_counts = df.isnull().sum()
    quality_report['missing_values'] = missing_counts[missing_counts > 0]
    
    # Duplicate rows
    quality_report['duplicates'] = df.duplicated().sum()
    
    # Data types
    quality_report['data_types'] = df.dtypes.value_counts()
    
    # Unique values per column
    quality_report['unique_values'] = df.nunique()
    
    # Potential categorical columns (low cardinality)
    potential_categorical = df.select_dtypes(include=['object']).columns
    quality_report['categorical_candidates'] = {
        col: df[col].nunique() for col in potential_categorical
    }
    
    return quality_report

# Example usage
quality = assess_data_quality(df)
print("Data Quality Report:")
for key, value in quality.items():
    print(f"\n{key.upper()}:")
    print(value)
```

## 🧹 Data Cleaning

### Handling Missing Values
```python
# Different strategies for missing values
def handle_missing_values(df, strategy='auto'):
    """Handle missing values with different strategies"""
    df_clean = df.copy()
    
    for column in df_clean.columns:
        missing_pct = df_clean[column].isnull().sum() / len(df_clean)
        
        if missing_pct == 0:
            continue
        elif missing_pct > 0.5:
            print(f"Dropping {column} (>{50}% missing)")
            df_clean.drop(column, axis=1, inplace=True)
        else:
            if df_clean[column].dtype in ['int64', 'float64']:
                # Numerical columns
                if strategy == 'mean':
                    df_clean[column].fillna(df_clean[column].mean(), inplace=True)
                elif strategy == 'median':
                    df_clean[column].fillna(df_clean[column].median(), inplace=True)
                elif strategy == 'mode':
                    df_clean[column].fillna(df_clean[column].mode()[0], inplace=True)
                else:  # auto
                    # Use median for skewed data, mean for normal
                    skewness = df_clean[column].skew()
                    if abs(skewness) > 1:
                        df_clean[column].fillna(df_clean[column].median(), inplace=True)
                    else:
                        df_clean[column].fillna(df_clean[column].mean(), inplace=True)
            else:
                # Categorical columns
                df_clean[column].fillna(df_clean[column].mode()[0], inplace=True)
    
    return df_clean

# Advanced imputation
from sklearn.impute import KNNImputer, IterativeImputer

# KNN Imputation
knn_imputer = KNNImputer(n_neighbors=5)
df_knn = pd.DataFrame(knn_imputer.fit_transform(df.select_dtypes(include=[np.number])),
                      columns=df.select_dtypes(include=[np.number]).columns)

# Iterative Imputation (MICE)
iterative_imputer = IterativeImputer(random_state=42)
df_iterative = pd.DataFrame(iterative_imputer.fit_transform(df.select_dtypes(include=[np.number])),
                           columns=df.select_dtypes(include=[np.number]).columns)
```

### Outlier Detection and Treatment
```python
def detect_outliers(df, method='iqr'):
    """Detect outliers using different methods"""
    outliers = {}
    
    for column in df.select_dtypes(include=[np.number]).columns:
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[column] = df[(df[column] < lower_bound) | (df[column] > upper_bound)].index
        
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            outliers[column] = df[z_scores > 3].index
        
        elif method == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = iso_forest.fit_predict(df[[column]])
            outliers[column] = df[outlier_labels == -1].index
    
    return outliers

def treat_outliers(df, outliers, method='cap'):
    """Treat outliers using different methods"""
    df_treated = df.copy()
    
    for column, outlier_indices in outliers.items():
        if len(outlier_indices) == 0:
            continue
            
        if method == 'remove':
            df_treated.drop(outlier_indices, inplace=True)
        elif method == 'cap':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df_treated[column] = df_treated[column].clip(lower=lower_bound, upper=upper_bound)
        elif method == 'transform':
            # Log transformation for right-skewed data
            if df[column].skew() > 1:
                df_treated[column] = np.log1p(df_treated[column])
    
    return df_treated

# Example usage
outliers = detect_outliers(df, method='iqr')
df_clean = treat_outliers(df, outliers, method='cap')
```

## 🔄 Data Transformation

### Scaling and Normalization
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

def scale_features(X_train, X_test, method='standard'):
    """Scale features using different methods"""
    
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    elif method == 'power':
        scaler = PowerTransformer(method='yeo-johnson')
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

# Example
from sklearn.model_selection import train_test_split

# Prepare data
X = df.select_dtypes(include=[np.number]).drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test, method='standard')

# Visualize scaling effect
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(X_train.iloc[:, 0], bins=30, alpha=0.7, label='Original')
axes[0].set_title('Original Distribution')
axes[1].hist(X_train_scaled[:, 0], bins=30, alpha=0.7, label='Scaled')
axes[1].set_title('Scaled Distribution')
plt.show()
```

### Encoding Categorical Variables
```python
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

def encode_categorical(df, method='auto'):
    """Encode categorical variables"""
    df_encoded = df.copy()
    encoders = {}
    
    categorical_columns = df.select_dtypes(include=['object']).columns
    
    for column in categorical_columns:
        unique_values = df[column].nunique()
        
        if method == 'auto':
            # Use one-hot for low cardinality, label encoding for high cardinality
            if unique_values <= 10:
                method_to_use = 'onehot'
            else:
                method_to_use = 'label'
        else:
            method_to_use = method
        
        if method_to_use == 'onehot':
            # One-hot encoding
            dummies = pd.get_dummies(df[column], prefix=column, drop_first=True)
            df_encoded = pd.concat([df_encoded.drop(column, axis=1), dummies], axis=1)
            encoders[column] = {'method': 'onehot', 'columns': dummies.columns.tolist()}
        
        elif method_to_use == 'label':
            # Label encoding
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df[column].astype(str))
            encoders[column] = {'method': 'label', 'encoder': le}
        
        elif method_to_use == 'target':
            # Target encoding (for supervised learning)
            target_means = df.groupby(column)['target'].mean()
            df_encoded[column] = df[column].map(target_means)
            encoders[column] = {'method': 'target', 'mapping': target_means}
    
    return df_encoded, encoders

# Example usage
df_encoded, encoders = encode_categorical(df, method='auto')
print("Encoding summary:")
for col, info in encoders.items():
    print(f"{col}: {info['method']}")
```

## 🎯 Feature Engineering

### Creating New Features
```python
def create_features(df):
    """Create new features from existing ones"""
    df_features = df.copy()
    
    # Numerical feature combinations
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) >= 2:
        # Ratios
        df_features[f'{numerical_cols[0]}_to_{numerical_cols[1]}_ratio'] = (
            df[numerical_cols[0]] / (df[numerical_cols[1]] + 1e-8)
        )
        
        # Products
        df_features[f'{numerical_cols[0]}_times_{numerical_cols[1]}'] = (
            df[numerical_cols[0]] * df[numerical_cols[1]]
        )
        
        # Differences
        df_features[f'{numerical_cols[0]}_minus_{numerical_cols[1]}'] = (
            df[numerical_cols[0]] - df[numerical_cols[1]]
        )
    
    # Polynomial features
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
    poly_features = poly.fit_transform(df[numerical_cols[:3]])  # Limit to avoid explosion
    
    poly_feature_names = poly.get_feature_names_out(numerical_cols[:3])
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)
    
    # Add only interaction terms (not squares)
    interaction_cols = [col for col in poly_df.columns if ' ' in col]
    df_features = pd.concat([df_features, poly_df[interaction_cols]], axis=1)
    
    # Binning continuous variables
    for col in numerical_cols:
        df_features[f'{col}_binned'] = pd.cut(df[col], bins=5, labels=['low', 'med_low', 'med', 'med_high', 'high'])
    
    return df_features

# Date/time features
def extract_datetime_features(df, datetime_col):
    """Extract features from datetime column"""
    df[datetime_col] = pd.to_datetime(df[datetime_col])
    
    df[f'{datetime_col}_year'] = df[datetime_col].dt.year
    df[f'{datetime_col}_month'] = df[datetime_col].dt.month
    df[f'{datetime_col}_day'] = df[datetime_col].dt.day
    df[f'{datetime_col}_dayofweek'] = df[datetime_col].dt.dayofweek
    df[f'{datetime_col}_hour'] = df[datetime_col].dt.hour
    df[f'{datetime_col}_is_weekend'] = df[datetime_col].dt.dayofweek.isin([5, 6]).astype(int)
    
    return df
```

### Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier

def select_features(X, y, method='mutual_info', k=10):
    """Select best features using different methods"""
    
    if method == 'univariate':
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    elif method == 'rfe':
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=k)
    
    X_selected = selector.fit_transform(X, y)
    
    # Get selected feature names
    if hasattr(selector, 'get_support'):
        selected_features = X.columns[selector.get_support()].tolist()
    else:
        selected_features = [f'feature_{i}' for i in range(X_selected.shape[1])]
    
    return X_selected, selected_features, selector

# Feature importance from tree-based models
def get_feature_importance(X, y):
    """Get feature importance using Random Forest"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(15), x='importance', y='feature')
    plt.title('Top 15 Feature Importances')
    plt.tight_layout()
    plt.show()
    
    return importance_df
```

## 📊 Data Validation

### Data Quality Checks
```python
def validate_processed_data(df_original, df_processed):
    """Validate processed data quality"""
    validation_report = {}
    
    # Shape comparison
    validation_report['shape_change'] = {
        'original': df_original.shape,
        'processed': df_processed.shape
    }
    
    # Missing values
    validation_report['missing_values'] = {
        'original': df_original.isnull().sum().sum(),
        'processed': df_processed.isnull().sum().sum()
    }
    
    # Data types
    validation_report['data_types'] = {
        'original': df_original.dtypes.value_counts().to_dict(),
        'processed': df_processed.dtypes.value_counts().to_dict()
    }
    
    # Value ranges for numerical columns
    numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
    validation_report['value_ranges'] = {}
    
    for col in numerical_cols:
        if col in df_original.columns:
            validation_report['value_ranges'][col] = {
                'original_range': [df_original[col].min(), df_original[col].max()],
                'processed_range': [df_processed[col].min(), df_processed[col].max()]
            }
    
    return validation_report

# Data leakage detection
def check_data_leakage(X_train, X_test, threshold=0.95):
    """Check for potential data leakage between train and test sets"""
    leakage_report = {}
    
    for col in X_train.columns:
        if col in X_test.columns:
            # Check correlation between train and test distributions
            train_values = X_train[col].value_counts(normalize=True)
            test_values = X_test[col].value_counts(normalize=True)
            
            # Calculate overlap
            common_values = set(train_values.index) & set(test_values.index)
            if len(common_values) > 0:
                correlation = train_values[list(common_values)].corr(test_values[list(common_values)])
                if correlation > threshold:
                    leakage_report[col] = correlation
    
    return leakage_report
```

## 🧪 Complete Preprocessing Pipeline

```python
class DataPreprocessor:
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selector = None
        self.selected_features = None
    
    def fit_transform(self, X_train, y_train=None):
        """Fit preprocessing pipeline and transform training data"""
        X_processed = X_train.copy()
        
        # 1. Handle missing values
        X_processed = self._handle_missing_values(X_processed)
        
        # 2. Encode categorical variables
        X_processed = self._encode_categorical(X_processed)
        
        # 3. Create new features
        X_processed = self._create_features(X_processed)
        
        # 4. Scale numerical features
        X_processed = self._scale_features(X_processed, fit=True)
        
        # 5. Select features (if target provided)
        if y_train is not None:
            X_processed = self._select_features(X_processed, y_train, fit=True)
        
        return X_processed
    
    def transform(self, X_test):
        """Transform test data using fitted pipeline"""
        X_processed = X_test.copy()
        
        # Apply same transformations
        X_processed = self._handle_missing_values(X_processed)
        X_processed = self._encode_categorical(X_processed)
        X_processed = self._create_features(X_processed)
        X_processed = self._scale_features(X_processed, fit=False)
        
        if self.selected_features is not None:
            X_processed = X_processed[self.selected_features]
        
        return X_processed
    
    def _handle_missing_values(self, X):
        # Implementation details...
        return X
    
    def _encode_categorical(self, X):
        # Implementation details...
        return X
    
    def _create_features(self, X):
        # Implementation details...
        return X
    
    def _scale_features(self, X, fit=False):
        # Implementation details...
        return X
    
    def _select_features(self, X, y, fit=False):
        # Implementation details...
        return X

# Usage
preprocessor = DataPreprocessor()
X_train_processed = preprocessor.fit_transform(X_train, y_train)
X_test_processed = preprocessor.transform(X_test)
```

## 🎯 Key Takeaways

### Essential Steps
1. **Explore**: Understand your data structure and quality
2. **Clean**: Handle missing values, outliers, and duplicates
3. **Transform**: Scale, encode, and normalize features
4. **Engineer**: Create meaningful new features
5. **Select**: Choose the most relevant features
6. **Validate**: Ensure data quality and prevent leakage

### Best Practices
- Always split data before preprocessing to avoid leakage
- Document all preprocessing steps for reproducibility
- Validate transformations don't introduce bias
- Keep preprocessing pipelines modular and reusable
- Monitor data drift in production systems

---
*Continue to [Neural Networks](../../02-Core-Concepts/01-Neural-Networks/) →*