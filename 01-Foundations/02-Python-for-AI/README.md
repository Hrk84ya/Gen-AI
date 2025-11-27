# 🐍 Python for AI

## 🎯 Learning Objectives
- Master essential Python libraries for AI/ML
- Understand data manipulation with NumPy and Pandas
- Create powerful visualizations with Matplotlib and Seaborn
- Build and evaluate ML models with scikit-learn
- Work with real datasets and solve practical problems

## 🚀 Python Fundamentals for AI

### Essential Data Structures
```python
# Lists - Dynamic arrays
data = [1, 2, 3, 4, 5]
features = ['age', 'income', 'score']

# List comprehensions (very important for AI)
squared = [x**2 for x in data]
filtered = [x for x in data if x > 2]

# Dictionaries - Key-value pairs
model_config = {
    'learning_rate': 0.01,
    'epochs': 100,
    'batch_size': 32
}

# Functions for reusable code
def preprocess_data(data, normalize=True):
    """Preprocess data for ML model"""
    if normalize:
        return [(x - min(data)) / (max(data) - min(data)) for x in data]
    return data

# Classes for organizing code
class DataProcessor:
    def __init__(self, method='standard'):
        self.method = method
        self.fitted = False
    
    def fit_transform(self, data):
        # Fit parameters and transform data
        self.mean = sum(data) / len(data)
        self.fitted = True
        return [(x - self.mean) for x in data]
```

## 📊 NumPy - The Foundation of AI

### Array Creation and Manipulation
```python
import numpy as np

# Different ways to create arrays
arr1 = np.array([1, 2, 3, 4, 5])                    # From list
arr2 = np.zeros((3, 4))                             # Zeros matrix
arr3 = np.ones((2, 3))                              # Ones matrix
arr4 = np.random.randn(1000)                        # Random normal
arr5 = np.linspace(0, 10, 100)                      # Linear space
arr6 = np.arange(0, 100, 2)                         # Range with step

# Array properties
print(f"Shape: {arr1.shape}")
print(f"Data type: {arr1.dtype}")
print(f"Number of dimensions: {arr1.ndim}")
print(f"Size: {arr1.size}")

# Reshaping arrays (crucial for ML)
matrix = np.arange(12).reshape(3, 4)                # 3x4 matrix
flattened = matrix.flatten()                         # Back to 1D
reshaped = matrix.reshape(-1, 2)                     # Auto-calculate dimension

print("Original matrix:")
print(matrix)
print("\nReshaped to (-1, 2):")
print(reshaped)
```

### Mathematical Operations
```python
# Element-wise operations
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print(f"Addition: {a + b}")
print(f"Multiplication: {a * b}")
print(f"Power: {a ** 2}")
print(f"Square root: {np.sqrt(a)}")

# Matrix operations (essential for neural networks)
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

matrix_mult = np.dot(A, B)                          # Matrix multiplication
element_mult = A * B                                # Element-wise multiplication
transpose = A.T                                      # Transpose
inverse = np.linalg.inv(A)                          # Inverse

print("Matrix multiplication:")
print(matrix_mult)
print("\nTranspose:")
print(transpose)

# Statistical operations
data = np.random.randn(1000)
print(f"Mean: {np.mean(data):.3f}")
print(f"Standard deviation: {np.std(data):.3f}")
print(f"Min: {np.min(data):.3f}")
print(f"Max: {np.max(data):.3f}")
print(f"Median: {np.median(data):.3f}")

# Advanced indexing (very useful for data manipulation)
arr = np.arange(20).reshape(4, 5)
print("Original array:")
print(arr)
print("\nFirst row:", arr[0, :])                    # First row
print("Last column:", arr[:, -1])                   # Last column
print("Subarray:", arr[1:3, 2:4])                   # Slice
print("Conditional:", arr[arr > 10])                # Boolean indexing
```

### Broadcasting (Advanced but Important)
```python
# Broadcasting allows operations between arrays of different shapes
a = np.array([[1, 2, 3],
              [4, 5, 6]])
b = np.array([10, 20, 30])

result = a + b  # b is "broadcast" to match a's shape
print("Broadcasting result:")
print(result)

# Practical example: Normalizing data
data = np.random.randn(100, 5)  # 100 samples, 5 features
mean = np.mean(data, axis=0)    # Mean of each feature
std = np.std(data, axis=0)      # Std of each feature
normalized = (data - mean) / std  # Broadcasting in action

print(f"Original mean: {np.mean(data, axis=0)}")
print(f"Normalized mean: {np.mean(normalized, axis=0)}")
```

## 🐼 Pandas - Data Manipulation Powerhouse

### DataFrames and Series
```python
import pandas as pd
import numpy as np

# Creating DataFrames
# Method 1: From dictionary
df1 = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'Diana'],
    'age': [25, 30, 35, 28],
    'salary': [50000, 60000, 70000, 55000],
    'department': ['Engineering', 'Marketing', 'Engineering', 'Sales']
})

# Method 2: From NumPy array
data = np.random.randn(100, 4)
df2 = pd.DataFrame(data, columns=['feature1', 'feature2', 'feature3', 'target'])

# Method 3: From CSV (most common in real projects)
# df3 = pd.read_csv('data.csv')

print("DataFrame info:")
print(df1.info())
print("\nFirst few rows:")
print(df1.head())
print("\nBasic statistics:")
print(df1.describe())
```

### Data Selection and Filtering
```python
# Selecting columns
names = df1['name']                                  # Single column (Series)
subset = df1[['name', 'age']]                       # Multiple columns (DataFrame)

# Selecting rows
first_row = df1.iloc[0]                             # By position
first_row_alt = df1.loc[0]                          # By index
slice_rows = df1.iloc[1:3]                          # Slice by position

# Conditional filtering (very important for data analysis)
young_employees = df1[df1['age'] < 30]
high_earners = df1[df1['salary'] > 55000]
engineers = df1[df1['department'] == 'Engineering']

# Multiple conditions
young_engineers = df1[(df1['age'] < 30) & (df1['department'] == 'Engineering')]
high_earners_or_young = df1[(df1['salary'] > 60000) | (df1['age'] < 30)]

print("Young employees:")
print(young_employees)
print("\nYoung engineers:")
print(young_engineers)
```

### Data Manipulation and Cleaning
```python
# Adding new columns
df1['salary_k'] = df1['salary'] / 1000              # Salary in thousands
df1['age_group'] = pd.cut(df1['age'], bins=[0, 30, 40, 100], 
                         labels=['Young', 'Middle', 'Senior'])

# Handling missing data
df_with_missing = df1.copy()
df_with_missing.loc[1, 'salary'] = np.nan           # Introduce missing value

print("Missing values:")
print(df_with_missing.isnull().sum())

# Fill missing values
df_filled = df_with_missing.fillna(df_with_missing['salary'].mean())

# Drop missing values
df_dropped = df_with_missing.dropna()

# Grouping and aggregation (essential for data analysis)
grouped = df1.groupby('department').agg({
    'salary': ['mean', 'max', 'min'],
    'age': 'mean'
})

print("\nGrouped statistics:")
print(grouped)

# Sorting
sorted_by_salary = df1.sort_values('salary', ascending=False)
sorted_multiple = df1.sort_values(['department', 'salary'], ascending=[True, False])

print("\nTop earners:")
print(sorted_by_salary.head())
```

### Advanced Pandas Operations
```python
# Merging DataFrames (like SQL joins)
df_bonus = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'bonus': [5000, 3000, 7000]
})

merged = pd.merge(df1, df_bonus, on='name', how='left')
print("Merged DataFrame:")
print(merged)

# Pivot tables
pivot = df1.pivot_table(values='salary', index='department', 
                       columns='age_group', aggfunc='mean')
print("\nPivot table:")
print(pivot)

# Apply custom functions
def categorize_salary(salary):
    if salary < 55000:
        return 'Low'
    elif salary < 65000:
        return 'Medium'
    else:
        return 'High'

df1['salary_category'] = df1['salary'].apply(categorize_salary)

# String operations (useful for text data)
df1['name_upper'] = df1['name'].str.upper()
df1['name_length'] = df1['name'].str.len()

print("\nDataFrame with new columns:")
print(df1)
```

## 📈 Matplotlib - Data Visualization

### Basic Plots
```python
import matplotlib.pyplot as plt
import numpy as np

# Set up the plotting style
plt.style.use('seaborn-v0_8')  # Modern, clean style
plt.rcParams['figure.figsize'] = (12, 8)

# Generate sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.exp(-x/5)

# Line plots
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(x, y1, 'b-', label='sin(x)', linewidth=2)
plt.plot(x, y2, 'r--', label='cos(x)', linewidth=2)
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Trigonometric Functions')
plt.legend()
plt.grid(True, alpha=0.3)

# Scatter plot
np.random.seed(42)
x_scatter = np.random.randn(100)
y_scatter = 2 * x_scatter + np.random.randn(100) * 0.5

plt.subplot(1, 3, 2)
plt.scatter(x_scatter, y_scatter, alpha=0.6, c=y_scatter, cmap='viridis')
plt.xlabel('X values')
plt.ylabel('Y values')
plt.title('Scatter Plot with Color')
plt.colorbar()

# Histogram
data = np.random.normal(100, 15, 1000)

plt.subplot(1, 3, 3)
plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histogram of Normal Distribution')
plt.axvline(np.mean(data), color='red', linestyle='--', 
           label=f'Mean: {np.mean(data):.1f}')
plt.legend()

plt.tight_layout()
plt.show()
```

### Advanced Visualizations
```python
# Subplots with different types
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Multiple line plots
axes[0, 0].plot(x, y1, label='sin(x)')
axes[0, 0].plot(x, y2, label='cos(x)')
axes[0, 0].plot(x, y3, label='damped sin(x)')
axes[0, 0].set_title('Multiple Functions')
axes[0, 0].legend()
axes[0, 0].grid(True)

# 2. Bar plot
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]
axes[0, 1].bar(categories, values, color=['red', 'blue', 'green', 'orange', 'purple'])
axes[0, 1].set_title('Bar Chart')
axes[0, 1].set_ylabel('Values')

# 3. Box plot
data_groups = [np.random.normal(0, 1, 100), 
               np.random.normal(2, 1.5, 100),
               np.random.normal(-1, 0.5, 100)]
axes[1, 0].boxplot(data_groups, labels=['Group 1', 'Group 2', 'Group 3'])
axes[1, 0].set_title('Box Plot')
axes[1, 0].set_ylabel('Values')

# 4. Heatmap
data_2d = np.random.randn(10, 10)
im = axes[1, 1].imshow(data_2d, cmap='coolwarm', aspect='auto')
axes[1, 1].set_title('Heatmap')
plt.colorbar(im, ax=axes[1, 1])

plt.tight_layout()
plt.show()
```

## 🎨 Seaborn - Statistical Visualization

```python
import seaborn as sns

# Load built-in dataset
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')

# Set style
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

# 1. Distribution plot
plt.subplot(2, 3, 1)
sns.histplot(tips['total_bill'], kde=True)
plt.title('Distribution of Total Bill')

# 2. Scatter plot with regression line
plt.subplot(2, 3, 2)
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time')
sns.regplot(data=tips, x='total_bill', y='tip', scatter=False, color='red')
plt.title('Bill vs Tip')

# 3. Box plot
plt.subplot(2, 3, 3)
sns.boxplot(data=tips, x='day', y='total_bill')
plt.title('Bill by Day')
plt.xticks(rotation=45)

# 4. Violin plot
plt.subplot(2, 3, 4)
sns.violinplot(data=iris, x='species', y='sepal_length')
plt.title('Sepal Length by Species')

# 5. Correlation heatmap
plt.subplot(2, 3, 5)
corr_matrix = iris.select_dtypes(include=[np.number]).corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')

# 6. Pair plot (in separate figure due to size)
plt.subplot(2, 3, 6)
# This would normally be: sns.pairplot(iris, hue='species')
# For demo, we'll show a simple count plot
sns.countplot(data=tips, x='day', hue='time')
plt.title('Count by Day and Time')

plt.tight_layout()
plt.show()

# Separate pair plot
sns.pairplot(iris, hue='species', height=2.5)
plt.show()
```

## 🤖 Scikit-learn - Machine Learning

### Complete ML Pipeline
```python
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Load and prepare data
iris = load_iris()
X, y = iris.data, iris.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Classes: {iris.target_names}")

# Create and compare multiple models
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42)
}

results = {}

for name, model in models.items():
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', model)
    ])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    
    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"\n{name} Results:")
    print(f"Test Accuracy: {accuracy:.3f}")
    print(f"CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Hyperparameter tuning
print("\n" + "="*50)
print("HYPERPARAMETER TUNING")
print("="*50)

# Grid search for Random Forest
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
}

rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

grid_search = GridSearchCV(
    rf_pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1
)

grid_search.fit(X_train, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score: {grid_search.best_score_:.3f}")

# Test the best model
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
print(f"Test accuracy with best model: {accuracy_score(y_test, y_pred_best):.3f}")
```

### Regression Example
```python
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Generate regression data
X_reg, y_reg = make_regression(n_samples=1000, n_features=1, noise=10, random_state=42)

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train models
linear_reg = LinearRegression()
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

linear_reg.fit(X_train_reg, y_train_reg)
rf_reg.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_linear = linear_reg.predict(X_test_reg)
y_pred_rf = rf_reg.predict(X_test_reg)

# Evaluate models
print("Linear Regression:")
print(f"MSE: {mean_squared_error(y_test_reg, y_pred_linear):.2f}")
print(f"R²: {r2_score(y_test_reg, y_pred_linear):.3f}")

print("\nRandom Forest:")
print(f"MSE: {mean_squared_error(y_test_reg, y_pred_rf):.2f}")
print(f"R²: {r2_score(y_test_reg, y_pred_rf):.3f}")

# Visualize results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(X_test_reg, y_test_reg, alpha=0.6, label='True values')
plt.scatter(X_test_reg, y_pred_linear, alpha=0.6, label='Predictions')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Linear Regression')
plt.legend()

plt.subplot(1, 3, 2)
plt.scatter(X_test_reg, y_test_reg, alpha=0.6, label='True values')
plt.scatter(X_test_reg, y_pred_rf, alpha=0.6, label='Predictions')
plt.xlabel('Feature')
plt.ylabel('Target')
plt.title('Random Forest')
plt.legend()

plt.subplot(1, 3, 3)
plt.scatter(y_test_reg, y_pred_linear, alpha=0.6, label='Linear Reg')
plt.scatter(y_test_reg, y_pred_rf, alpha=0.6, label='Random Forest')
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Predictions vs True Values')
plt.legend()

plt.tight_layout()
plt.show()
```

## 🧪 Comprehensive Hands-On Project

### Real-World Data Analysis Project
```python
# Let's create a comprehensive project using all the skills
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

# Create a synthetic dataset (in real projects, you'd load from CSV)
np.random.seed(42)
n_samples = 1000

# Generate synthetic customer data
data = {
    'age': np.random.randint(18, 80, n_samples),
    'income': np.random.normal(50000, 20000, n_samples),
    'spending_score': np.random.randint(1, 100, n_samples),
    'years_customer': np.random.randint(0, 20, n_samples),
    'num_purchases': np.random.poisson(10, n_samples),
    'gender': np.random.choice(['Male', 'Female'], n_samples),
    'city_tier': np.random.choice(['Tier1', 'Tier2', 'Tier3'], n_samples)
}

# Create target variable (will customer churn?)
# Make it somewhat realistic - higher income and spending score = less likely to churn
churn_probability = (
    0.5 - (data['income'] - 30000) / 100000 * 0.3 - 
    (data['spending_score'] - 50) / 100 * 0.2 +
    np.random.normal(0, 0.1, n_samples)
)
data['churn'] = (np.random.random(n_samples) < np.clip(churn_probability, 0, 1)).astype(int)

df = pd.DataFrame(data)

print("=" * 60)
print("CUSTOMER CHURN PREDICTION PROJECT")
print("=" * 60)

# 1. DATA EXPLORATION
print("\n1. DATA EXPLORATION")
print("-" * 30)
print(f"Dataset shape: {df.shape}")
print(f"\nChurn distribution:")
print(df['churn'].value_counts(normalize=True))

print("\nBasic statistics:")
print(df.describe())

# 2. DATA VISUALIZATION
print("\n2. DATA VISUALIZATION")
print("-" * 30)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Age distribution by churn
axes[0, 0].hist([df[df['churn']==0]['age'], df[df['churn']==1]['age']], 
               bins=20, alpha=0.7, label=['No Churn', 'Churn'])
axes[0, 0].set_title('Age Distribution by Churn')
axes[0, 0].legend()

# Income vs Spending Score
scatter = axes[0, 1].scatter(df['income'], df['spending_score'], 
                           c=df['churn'], cmap='coolwarm', alpha=0.6)
axes[0, 1].set_xlabel('Income')
axes[0, 1].set_ylabel('Spending Score')
axes[0, 1].set_title('Income vs Spending Score (colored by churn)')
plt.colorbar(scatter, ax=axes[0, 1])

# Churn by gender
churn_by_gender = df.groupby(['gender', 'churn']).size().unstack()
churn_by_gender.plot(kind='bar', ax=axes[0, 2])
axes[0, 2].set_title('Churn by Gender')
axes[0, 2].set_xticklabels(['Female', 'Male'], rotation=0)

# Years as customer vs churn
axes[1, 0].boxplot([df[df['churn']==0]['years_customer'], 
                   df[df['churn']==1]['years_customer']], 
                  labels=['No Churn', 'Churn'])
axes[1, 0].set_title('Years as Customer by Churn')

# Correlation heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Correlation Matrix')

# Churn by city tier
churn_by_tier = df.groupby(['city_tier', 'churn']).size().unstack()
churn_by_tier.plot(kind='bar', ax=axes[1, 2])
axes[1, 2].set_title('Churn by City Tier')
axes[1, 2].set_xticklabels(['Tier1', 'Tier2', 'Tier3'], rotation=0)

plt.tight_layout()
plt.show()

# 3. DATA PREPROCESSING
print("\n3. DATA PREPROCESSING")
print("-" * 30)

# Handle categorical variables
le_gender = LabelEncoder()
le_city = LabelEncoder()

df_processed = df.copy()
df_processed['gender_encoded'] = le_gender.fit_transform(df['gender'])
df_processed['city_tier_encoded'] = le_city.fit_transform(df['city_tier'])

# Select features for modeling
feature_columns = ['age', 'income', 'spending_score', 'years_customer', 
                  'num_purchases', 'gender_encoded', 'city_tier_encoded']

X = df_processed[feature_columns]
y = df_processed['churn']

print(f"Features selected: {feature_columns}")
print(f"Feature matrix shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# 4. MODEL TRAINING AND EVALUATION
print("\n4. MODEL TRAINING AND EVALUATION")
print("-" * 30)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Create and train models
models = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ]),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
}

for name, model in models.items():
    print(f"\n{name}:")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance (for Random Forest)
    if name == 'Random Forest':
        feature_importance = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Feature Importance - Random Forest')
        plt.tight_layout()
        plt.show()

# 5. INSIGHTS AND RECOMMENDATIONS
print("\n5. INSIGHTS AND RECOMMENDATIONS")
print("-" * 30)
print("Based on the analysis:")
print("• Income and spending score are strong predictors of churn")
print("• Customers with lower income are more likely to churn")
print("• Years as customer also influences churn probability")
print("• Gender and city tier have some impact but are less important")
print("\nRecommendations:")
print("• Focus retention efforts on lower-income customers")
print("• Implement loyalty programs for long-term customers")
print("• Monitor spending patterns for early churn detection")
```

## 🎯 Advanced Tips and Best Practices

### Code Organization
```python
# Use functions to organize your code
def load_and_clean_data(filepath):
    """Load and perform basic cleaning of data"""
    df = pd.read_csv(filepath)
    # Remove duplicates
    df = df.drop_duplicates()
    # Handle missing values
    df = df.fillna(df.mean(numeric_only=True))
    return df

def create_features(df):
    """Create new features from existing ones"""
    df = df.copy()
    # Example feature engineering
    df['feature_ratio'] = df['feature1'] / (df['feature2'] + 1e-8)
    df['feature_interaction'] = df['feature1'] * df['feature2']
    return df

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model):
    """Train model and return evaluation metrics"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'predictions': y_pred,
        'model': model
    }

# Use classes for complex workflows
class MLPipeline:
    def __init__(self, model, preprocessor=None):
        self.model = model
        self.preprocessor = preprocessor
        self.is_fitted = False
    
    def fit(self, X, y):
        if self.preprocessor:
            X = self.preprocessor.fit_transform(X)
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        if self.preprocessor:
            X = self.preprocessor.transform(X)
        return self.model.predict(X)
```

### Performance Optimization
```python
# Use vectorized operations instead of loops
# Slow
result = []
for i in range(len(data)):
    result.append(data[i] ** 2)

# Fast
result = np.array(data) ** 2

# Use appropriate data types
df['category'] = df['category'].astype('category')  # For categorical data
df['integer_col'] = df['integer_col'].astype('int32')  # If values fit in int32

# Use chunking for large datasets
chunk_size = 10000
for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
    # Process each chunk
    processed_chunk = process_data(chunk)
    # Save or accumulate results
```

## 🎯 Key Takeaways

### Essential Skills Mastered
- **NumPy**: Efficient numerical computing and array operations
- **Pandas**: Data manipulation, cleaning, and analysis
- **Matplotlib/Seaborn**: Data visualization and exploration
- **Scikit-learn**: Machine learning model building and evaluation
- **Best Practices**: Code organization, performance optimization

### Real-World Applications
- Data cleaning and preprocessing pipelines
- Exploratory data analysis and visualization
- Feature engineering and selection
- Model training, evaluation, and comparison
- End-to-end machine learning projects

### Next Steps
- Practice with real datasets from Kaggle or UCI repository
- Learn advanced pandas operations (pivot tables, merging, groupby)
- Explore more visualization libraries (Plotly, Bokeh)
- Study different machine learning algorithms in depth
- Work on complete projects from data collection to deployment

---
*Continue to [Mathematics for AI](../03-Math-for-AI/) →*