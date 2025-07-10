# 🧬 Liver Cancer Gene Expression Classification

This repository contains a complete machine learning pipeline for classifying liver cancer based on gene expression data. It includes data preprocessing, exploratory analysis, dimensionality reduction using PCA, and classification using various ML models.

## 📁 Dataset

The dataset used in this project is assumed to be in CSV format and named `liver_cancer_expression_data.csv`, containing gene expression features and a `type` column indicating class labels (e.g., `HCC` vs. `normal`).

---

## 🛠️ Features of the Code

- Missing value handling with column-wise mean imputation.
- Feature normalization using z-score standardization.
- Dimensionality reduction using Principal Component Analysis (PCA).
- Visualizations for data distribution and PCA projection.
- ML models: Random Forest, SVM, XGBoost, Neural Network (MLP).
- Hyperparameter tuning using GridSearchCV.
- Evaluation using accuracy, classification report, confusion matrix, and AUC-ROC.

---

## 🔧 Requirements

Install the following Python packages before running the script:

pip install pandas numpy matplotlib seaborn scikit-learn xgboost



## 🧪 How to Run

Place your dataset file as `liver_cancer_expression_data.csv` in the project directory.

Run the script using:

python liver_cancer_classification.py

---

## 🧹 Data Preprocessing

- Loaded dataset and checked for null values.
- Imputed missing values with mean (for numeric columns).
- Standardized features using `StandardScaler`.
- Visualized 20 randomly selected standardized features using boxplots.

---

## 📊 PCA Visualization

- Reduced data to 2D using PCA for visualization.
- Plotted principal components with color-coded classes.

---

## ⚙️ Machine Learning Models

Trained and evaluated the following models:

| Model           | Hyperparameters Tuned                         | Evaluation Metrics     |
|----------------|-----------------------------------------------|------------------------|
| Random Forest  | `n_estimators`, `max_depth`, `min_samples_split` | Accuracy, AUC-ROC     |
| SVM            | `C`, `kernel`                                 | Accuracy, AUC-ROC     |
| XGBoost        | `n_estimators`, `max_depth`                   | Accuracy, AUC-ROC     |
| Neural Network | `alpha` (regularization)                      | Accuracy, AUC-ROC     |

> **Note:** Dimensionality reduction to 50 principal components was performed before training to avoid overfitting.

---

## 📈 Model Evaluation

- Accuracy and classification reports printed for each model.
- ROC curves plotted with AUC scores for probabilistic models.

---

## 📌 Example Output (Console)

Random Forest Test Accuracy: 0.8750

Classification Report:
              precision    recall  f1-score   support
     Normal       0.89      0.86      0.87        14
        HCC       0.86      0.89      0.88        14

AUC-ROC Score for Random Forest: 0.93 ```

## 📎 Notes

- Set random seed for reproducibility.
- Suitable for high-dimensional datasets with limited samples.
- Uses `stratify=y` in train-test split to preserve class balance.

---

## 🧠 Author

**Divya Karade**  
Chemoinformatics & ML in Drug Discovery  
[LinkedIn](https://www.linkedin.com/in/divyakarade/) | [GitHub](https://github.com/your-github)

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
