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
from scipy.spatial.distance import mahalanobis
import shap

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
    raise ValueError("Error: 'type' column not found!")

# Identify missing values
missing_values = df.isnull().sum().sum()
print(f"\nTotal Missing Values: {missing_values}")
print("\nMissing Values per Column:\n", df.isnull().sum())

# Handle missing values (Impute Mean)
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].apply(lambda x: x.fillna(x.mean()), axis=0)

# Standardize Data
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Visualize data normalization using boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_cols].sample(20, axis=1))  # Sample 20 features for readability
plt.xticks(rotation=90)
plt.title("Feature Distribution After Normalization")
plt.show()

# Display original dataset size
print(f"\nOriginal Dataset Size: {df.shape[0]} samples")

# **Apply PCA for visualization (2D for Plotting)**
pca_vis = PCA(n_components=2)
X_pca_vis = pca_vis.fit_transform(df[numeric_cols])

# Create DataFrame for PCA visualization
df_pca_vis = pd.DataFrame(X_pca_vis, columns=['PC1', 'PC2'])
df_pca_vis['type'] = df['type'].values  # Preserve class labels

# **Plot PCA Visualization BEFORE Outlier Removal**
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca_vis, x='PC1', y='PC2', hue='type', palette='viridis', alpha=0.7)
plt.title("PCA Visualization Before Outlier Removal")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.show()

# **OUTLIER DETECTION USING PCA & MAHALANOBIS DISTANCE**
pca = PCA(n_components=50)  # Use 50 PCs for outlier detection & model training
X_pca = pca.fit_transform(df[numeric_cols])

# Compute Mahalanobis Distance
mean_pca = np.mean(X_pca, axis=0)
cov_inv_pca = np.linalg.inv(np.cov(X_pca, rowvar=False))
mahal_distances = np.array([mahalanobis(x, mean_pca, cov_inv_pca) for x in X_pca])

# Define Outlier Threshold (Top 5% Most Extreme Points)
threshold = np.percentile(mahal_distances, 95)
outlier_mask = mahal_distances > threshold  # True for outliers

# **Correctly Assign Outlier Information to `df_pca_vis`**
df_pca_vis["Outlier"] = np.where(outlier_mask, "Outlier", "Normal")  # Convert to string for seaborn coloring

# Print dataset sizes before & after removing outliers
print(f"\nOriginal Dataset Size: {df.shape[0]} samples")
print(f"Number of Outliers Detected: {outlier_mask.sum()} samples")

# **Plot PCA Visualization WITH Outliers Highlighted**
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_pca_vis, x='PC1', y='PC2', hue='Outlier', palette={"Normal": "blue", "Outlier": "red"},
                alpha=0.7)
plt.title("Outlier Detection Using PCA")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Outlier")
plt.show()

# **Remove Outliers**
df_cleaned = df[~outlier_mask].reset_index(drop=True)
X_pca_cleaned = X_pca[~outlier_mask]  # Keep only inliers
y_cleaned = df_cleaned['type'].astype('category').cat.codes  # Encode labels

# Print cleaned dataset size
print(f"Dataset Size After Removing Outliers: {df_cleaned.shape[0]} samples")

### (2) SETTING UP ML CLASSIFICATION ###
# Train-Test Split (After Outlier Removal)
X_train, X_test, y_train, y_test = train_test_split(X_pca_cleaned, y_cleaned, test_size=0.2, random_state=42,
                                                    stratify=y_cleaned)

# Explained Variance Ratio
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
    grid_search.fit(X_train, y_train)
    best_models[name] = grid_search.best_estimator_
    y_pred = grid_search.best_estimator_.predict(X_test)
    y_proba = grid_search.best_estimator_.predict_proba(X_test)[:, 1] if hasattr(grid_search.best_estimator_,
                                                                                 'predict_proba') else None

    print(f"\n{name} Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

    if y_proba is not None:
        auc_score = roc_auc_score(y_test, y_proba)
        print(f"AUC-ROC Score for {name}: {auc_score:.4f}")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.2f})')

# Plot ROC Curves
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Different Classifiers')
plt.legend()
plt.show()

# Initialize SHAP explainer and compute SHAP values
shap_explainers = {}
shap_values_dict = {}

for name, model in best_models.items():
    print(f"\nComputing SHAP values for {name}...")

    try:
        if name == "SVM":  # Use KernelExplainer for SVM
            explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_train, 100))
            shap_values = explainer.shap_values(X_test[:50])  # Limit for performance

            shap_explainers[name] = explainer
            shap_values_dict[name] = shap_values

            # Summary Plot
            plt.figure()
            shap.summary_plot(shap_values[1], X_test[:50], show=False)  # Index [1] for positive class
            plt.title(f"SHAP Summary Plot - {name}")
            plt.show()

        else:  # Use standard SHAP explainer for tree-based models
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_test)

            shap_explainers[name] = explainer
            shap_values_dict[name] = shap_values

            # Summary Plot
            plt.figure()
            shap.summary_plot(shap_values, X_test, show=False)
            plt.title(f"SHAP Summary Plot - {name}")
            plt.show()

    except Exception as e:
        print(f"SHAP computation failed for {name}: {e}")
