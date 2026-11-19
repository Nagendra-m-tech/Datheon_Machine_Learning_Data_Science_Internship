# Task 1: Multi-Class Benchmark Classification Pipeline (Iris / Wine)

Build end-to-end multi-class classification pipelines using **Scikit-Learn**, feature normalization (**StandardScaler**), and **Confusion Matrix** / **Decision Boundary** visualizations.

---

## 📋 Project Summary & Deliverables

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **Primary Deliverable** | [`multi_class_classification_pipeline.ipynb`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task1/multi_class_classification_pipeline.ipynb) | Complete Jupyter Notebook with step-by-step documentation, executed cells, metrics, and embedded visual plots. |
| **Executable Script** | [`pipeline.py`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task1/pipeline.py) | Standalone Python pipeline script that trains models, computes benchmark metrics, and exports all figures to `outputs/`. |
| **Datasets** | [`data/iris.csv`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task1/data/iris.csv) & [`data/wine.csv`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task1/data/wine.csv) | Benchmark CSV files loaded using Pandas. |
| **Visual Outputs** | [`outputs/`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task1/outputs/) | High-resolution PNG figures (pairplots, confusion matrices, 2D decision boundaries, and PCA decision surfaces). |

---

## 🛠️ Implemented Pipeline Architecture

1. **Dataset Ingestion**:
   - Loaded benchmark datasets (**Iris** and **Wine**) via `pandas.read_csv()`.
   - Verified data shapes, column types, statistical distributions, and missing values.
2. **Exploratory Data Analysis (EDA)**:
   - Generated **Seaborn pairplots** (`sns.pairplot`) with KDE distributions on the diagonal, grouped and colored by target species/class.
   - Evaluated feature correlation matrices via annotated heatmaps (`sns.heatmap`).
3. **Train/Test Splitting**:
   - Stratified **80/20 train/test split** (`test_size=0.20, random_state=42, stratify=y`) preserving class balances in train and test splits.
4. **Feature Normalization**:
   - Applied `StandardScaler()`.
   - **Data Leakage Prevention**: Scaler is fitted strictly on the training set (`scaler.fit_transform(X_train)`), and the learned parameters are used to transform the test set (`scaler.transform(X_test)`).
5. **Model Training**:
   - **Logistic Regression**: Linear multi-class model (`solver='lbfgs'`, `max_iter=500`).
   - **Decision Tree Classifier**: Non-linear tree partitioner (`criterion='gini'`, `max_depth=3`).
6. **Performance Benchmarking**:
   - Computed multi-class **Accuracy**, **Precision (Macro & Weighted)**, **Recall (Macro & Weighted)**, and **F1-Score (Macro & Weighted)**.
   - Generated complete Scikit-Learn `classification_report` for both models.
7. **Visualizations**:
   - **Plotted Confusion Matrix Heatmaps**: Side-by-side heatmaps with counts and normalized color scales.
   - **Decision Boundary Visualizations**:
     - *2D Feature Space*: Visualized boundaries on primary features (e.g., Petal Length vs Petal Width).
     - *PCA 2D Space*: Visualized boundaries across all features projected onto the first two principal components.

---

## 📊 Benchmark Results

### 1. Fisher's Iris Benchmark (Primary)
- **Train Samples**: 120 (80%) | **Test Samples**: 30 (20%) | **Classes**: Setosa (10), Versicolor (10), Virginica (10)

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **93.33%** | 0.9333 | 0.9333 | 0.9333 | 0.9333 |
| **Decision Tree (depth=3)** | **96.67%** | **0.9697** | **0.9667** | **0.9666** | **0.9666** |

#### Iris Classification Reports:
**Logistic Regression**:
```text
              precision    recall  f1-score   support
      setosa     1.0000    1.0000    1.0000        10
  versicolor     0.9000    0.9000    0.9000        10
   virginica     0.9000    0.9000    0.9000        10
    accuracy                         0.9333        30
   macro avg     0.9333    0.9333    0.9333        30
weighted avg     0.9333    0.9333    0.9333        30
```

**Decision Tree**:
```text
              precision    recall  f1-score   support
      setosa     1.0000    1.0000    1.0000        10
  versicolor     1.0000    0.9000    0.9474        10
   virginica     0.9091    1.0000    0.9524        10
    accuracy                         0.9667        30
   macro avg     0.9697    0.9667    0.9666        30
weighted avg     0.9697    0.9667    0.9666        30
```

---

### 2. Wine Recognition Benchmark
- **Train Samples**: 142 (80%) | **Test Samples**: 36 (20%) | **Features**: 13 | **Classes**: 3

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **97.22%** | **0.9778** | **0.9667** | **0.9710** | **0.9720** |
| **Decision Tree (depth=3)** | **94.44%** | 0.9583 | 0.9389 | 0.9457 | 0.9450 |

---

## 🚀 How to Run

### 1. View or Run the Jupyter Notebook
Open [`multi_class_classification_pipeline.ipynb`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task1/multi_class_classification_pipeline.ipynb) in VS Code or JupyterLab. The notebook is already pre-executed with all output tables and plots embedded.

### 2. Run the Standalone Script
To re-run the full pipeline from the terminal:
```bash
python pipeline.py
```
This re-computes all metrics and refreshes all plot figures in the [`outputs/`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task1/outputs/) directory.
