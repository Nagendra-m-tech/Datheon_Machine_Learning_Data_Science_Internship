# Task 3: Predictive Customer Churn Classifier (Random Forest & XGBoost)

Solve severe class imbalance and optimize ROC-AUC for customer attrition prediction using ensemble learning algorithms on the Telco Customer Churn dataset.

---

## 📋 Project Summary & Deliverables

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **Primary Deliverable** | [`customer_churn_pipeline.ipynb`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task3/customer_churn_pipeline.ipynb) | Complete Jupyter Notebook with step-by-step documentation, executed cells, metrics, and embedded visual plots. |
| **Executable Script** | [`pipeline.py`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task3/pipeline.py) | Standalone Python pipeline script that handles data cleaning, SMOTE, Random Forest, and XGBoost training. |
| **Dataset** | [`data/telco_customer_churn.csv`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task3/data/telco_customer_churn.csv) | Telco Customer Churn records (7,043 samples, 20 features). |
| **Visual Outputs** | [`outputs/`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task3/outputs/) | High-resolution PNG figures (ROC-AUC curves, confusion matrices, top 10 feature importances). |

---

## 🛠️ Implemented Pipeline Architecture

1. **Dataset Ingestion & Data Hygiene**:
   - Ingested 7,043 customer records with 20 features.
   - Cleaned `TotalCharges` converting blank characters into numerical float types and imputing missing records using median.
   - Dropped non-predictive identifier `customerID`.
2. **Feature Preprocessing & Encoding**:
   - Standardized continuous features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler`.
   - Encoded 16 categorical variables using `OneHotEncoder(drop='first')`, yielding 30 predictive features.
3. **Handling Severe Class Imbalance (SMOTE)**:
   - Target baseline: 73.46% retained vs. 26.54% churn.
   - Applied **SMOTE (Synthetic Minority Over-sampling Technique)** on training data to balance minority churn cases to 1:1 parity (4,139 samples each).
4. **Ensemble Modeling & 5-Fold Cross-Validation**:
   - Trained `RandomForestClassifier` (150 trees, max depth 9).
   - Trained `XGBClassifier` (150 estimators, learning rate 0.07, subsample 0.8).
5. **Visualizations**:
   - Plotted **ROC-AUC Curves** demonstrating performance against the required >80% threshold.
   - Plotted **Confusion Matrix Heatmaps** for both models side-by-side.
   - Extracted and visualized the **Top 10 Most Influential Feature Importances**.

---

## 📊 Benchmark Results (Unseen Test Set)

| Model | ROC-AUC Score | 5-Fold CV AUC | Target (>80%) | Accuracy | Recall (Churn) | Precision (Churn) | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **84.33%** | **90.30% ± 0.02** | **✅ PASSED** | **77.08%** | **74.60%** | **55.03%** | **0.6334** |
| **XGBoost** | **84.32%** | **93.43% ± 0.05** | **✅ PASSED** | **77.64%** | **64.97%** | **56.91%** | **0.6067** |

---

## 🚀 How to Run

1. **Jupyter Notebook**:
   Open [`customer_churn_pipeline.ipynb`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task3/customer_churn_pipeline.ipynb) in VS Code or Jupyter.
2. **Standalone Script**:
   ```bash
   python pipeline.py
   ```
