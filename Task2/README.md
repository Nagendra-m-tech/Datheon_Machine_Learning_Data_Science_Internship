# Task 2: Real Estate Housing Price Prediction (Ridge & Lasso Regression)

Perform continuous regression modeling, feature engineering, multicollinearity handling, and residual error analysis on the California Housing dataset.

---

## 📋 Project Summary & Deliverables

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **Primary Deliverable** | [`housing_price_prediction_pipeline.ipynb`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task2/housing_price_prediction_pipeline.ipynb) | Complete Jupyter Notebook with step-by-step documentation, executed cells, metrics, and embedded visual plots. |
| **Executable Script** | [`pipeline.py`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task2/pipeline.py) | Standalone Python pipeline script that performs feature engineering, 5-fold GridSearchCV tuning, and exports all diagnostic plots. |
| **Dataset** | [`data/california_housing.csv`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task2/data/california_housing.csv) | California housing records (20,640 samples). |
| **Visual Outputs** | [`outputs/`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task2/outputs/) | High-resolution PNG figures (GridSearchCV tuning curves, predicted vs. actual, residual distributions, and coefficient comparisons). |

---

## 🛠️ Implemented Pipeline Architecture

1. **Dataset Ingestion & Imputation**:
   - Ingested California Housing dataset (`fetch_california_housing`) into Pandas.
   - Cleaned and confirmed zero missing values using `SimpleImputer(strategy='median')`.
2. **Domain Feature Engineering**:
   - `BedroomsPerRoom`: Ratio of bedrooms to total rooms (overcrowding index).
   - `RoomsPerHousehold`: Overall space available per household.
   - `PopulationPerHousehold`: Neighborhood density indicator.
   - `MedInc_x_HouseAge`: Interaction term between income and house age.
   - `MedInc_Per_Room`: Economic density per room.
3. **Multicollinearity Analysis**:
   - Generated feature correlation heatmap (`outputs/correlation_heatmap.png`).
4. **Data Splitting & Scaling**:
   - 80/20 train/test split (16,512 train samples, 4,128 test samples).
   - `StandardScaler()` fitted **solely on the training set** to prevent data leakage.
5. **Model Training & Hyperparameter Tuning**:
   - **Ordinary Least Squares (OLS)** baseline linear regression.
   - **Ridge Regression (L2)**: Tuned over 25 alpha values using 5-fold cross-validation with `GridSearchCV`.
   - **Lasso Regression (L1)**: Tuned over 25 alpha values using 5-fold cross-validation with `GridSearchCV`.
6. **Residual Diagnostics & Visualizations**:
   - Predicted vs. Actual scatter plots (with ideal $y = \hat{y}$ reference lines).
   - Residual distribution curves (KDE & histograms checking zero-centering).
   - Residuals vs. Fitted values (homoscedasticity diagnostics).
   - Regularization shrinkage bar chart comparing OLS, Ridge, and Lasso coefficients.

---

## 📊 Benchmark Results (Test Set Evaluation)

| Model | Optimal Hyperparameter ($\alpha$) | RMSE ($100k) | MAE ($100k) | $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **OLS Linear Regression** | — | **0.6947** | **0.5000** | **0.6317** |
| **Ridge Regression (L2)** | $\alpha = 10.0$ | **0.6948** | **0.5000** | **0.6316** |
| **Lasso Regression (L1)** | $\alpha = 0.0022$ | **0.6968** | **0.5010** | **0.6295** |

### Key Findings:
- **Feature Engineering** improved linear explanation capability to over **63.1% explained variance ($R^2 \approx 0.632$)**.
- **Ridge Regression** effectively stabilized coefficient variance across collinear features (`AveRooms`, `BedroomsPerRoom`).
- **Lasso Regression** provided automatic feature selection by zeroing out weak predictors.
- **Residual Distributions** are symmetric and centered at $0.0000$, validating standard linear regression assumptions.

---

## 🚀 How to Run

1. **Jupyter Notebook**:
   Open [`housing_price_prediction_pipeline.ipynb`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task2/housing_price_prediction_pipeline.ipynb) in VS Code or Jupyter to view all executed cells, charts, and metrics.
2. **Standalone Script**:
   ```bash
   python pipeline.py
   ```
