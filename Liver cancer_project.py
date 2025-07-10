import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Set random seeds for reproducibility
np.random.seed(42)

### (1) LOAD & EXPLORE THE DATASET ###
# Load dataset
file_path = "liver_cancer_expression_data.csv"
df = pd.read_csv(file_path)

# Display dataset information
print("Dataset Shape:", df.shape)
print("\nDataset Info:")
print(df.info())
print("\nFirst 5 rows:\n", df.head())

# Check class distribution
if 'type' in df.columns:
    print("\nClass Distribution:\n", df['type'].value_counts())
else:
    print("\nError: 'type' column not found!")

# Identify missing values
missing_values = df.isnull().sum().sum()
print(f"\nTotal Missing Values: {missing_values}")
print("\nMissing Values per Column:\n", df.isnull().sum())

# Handle missing values by imputing the mean (for numerical columns)
# Identify numeric columns only
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Fill missing values only in numeric columns
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()), axis=0)

# Normalize Data: Standardization (z-score normalization)
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Visualize data normalization using boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numerical_cols].sample(20, axis=1))  # Sample 20 features for readability
plt.xticks(rotation=90)
plt.title("Feature Distribution After Normalization")
plt.show()

# PCA for visualization
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df[numerical_cols])

df_pca = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
df_pca['type'] = df['type']

# Scatter plot of PCA results
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='type', palette='viridis', alpha=0.7)
plt.title("PCA Visualization of Feature Space")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

### (2) SETTING UP ML CLASSIFICATION ###

# Encode categorical labels
df['type'] = df['type'].astype('category').cat.codes  # Converts 'HCC' & 'normal' to 0/1

# Splitting data into train and test sets (80-20 split)
X = df[numerical_cols]
y = df['type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Dimensionality issue: Too many features (10,000) for limited samples (165) → Risk of overfitting
print(f"\nTrain Set Shape: {X_train.shape}, Test Set Shape: {X_test.shape}")

# Feature Selection: Using PCA to reduce dimensions
pca = PCA(n_components=50)  # Keeping top 50 principal components
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Explained variance ratio
explained_variance = np.sum(pca.explained_variance_ratio_)
print(f"\nExplained Variance by 50 PCs: {explained_variance:.2%}")

### (3) TRAINING AND EVALUATING CLASSIFIERS ###

classifiers = {
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

param_grids = {
    "Random Forest": {'n_estimators': [50, 100, 200], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5, 10]},
    "SVM": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']},
    "XGBoost": {'n_estimators': [50, 100], 'max_depth': [3, 6, 10]},
    "Neural Network": {'alpha': [0.0001, 0.001, 0.01]}
}

best_models = {}

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    grid_search = GridSearchCV(clf, param_grids[name], cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_pca, y_train)
    best_models[name] = grid_search.best_estimator_
    y_pred = grid_search.best_estimator_.predict(X_test_pca)
    y_proba = grid_search.best_estimator_.predict_proba(X_test_pca)[:, 1] if hasattr(grid_search.best_estimator_,
                                                                                     'predict_proba') else None

    print(f"\n{name} Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    if y_proba is not None:
        auc_score = roc_auc_score(y_test, y_proba)
        print(f"AUC-ROC Score for {name}: {auc_score:.4f}")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f}')

# Plot ROC Curves
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Classifiers')
plt.legend()
plt.show()
