# Datheon Machine Learning & Data Science Internship — Project Submission Portfolio

This repository contains the complete, production-ready code, Jupyter notebooks, datasets, benchmarks, and diagnostic visualizations for all **4 tasks** of the **Datheon Machine Learning & Data Science Internship**.

---

## 📂 Repository Architecture

```text
Datheon/
├── README.md                                    # Master project submission documentation
├── .gitignore                                   # Global environment and cache ignores
│
├── Task1/                                       # Multi-Class Benchmark Classification Pipeline (Iris / Wine)
│   ├── multi_class_classification_pipeline.ipynb
│   ├── pipeline.py
│   ├── README.md
│   ├── data/ (iris.csv, wine.csv)
│   └── outputs/ (pairplots, confusion matrices, 2D decision boundaries, PCA surfaces)
│
├── Task2/                                       # Real Estate Housing Price Prediction (Ridge & Lasso Regression)
│   ├── housing_price_prediction_pipeline.ipynb
│   ├── pipeline.py
│   ├── README.md
│   ├── data/ (california_housing.csv)
│   └── outputs/ (correlation heatmap, GridSearchCV tuning curves, predicted vs actual, residuals)
│
├── Task3/                                       # Predictive Customer Churn Classifier (Random Forest & XGBoost)
│   ├── customer_churn_pipeline.ipynb
│   ├── pipeline.py
│   ├── README.md
│   ├── data/ (telco_customer_churn.csv)
│   └── outputs/ (ROC-AUC curves, confusion matrices, top 10 feature importances)
│
└── Task4/                                       # NLP Sentiment Classification on Customer Reviews (TF-IDF)
    ├── sentiment_analysis_pipeline.ipynb
    ├── pipeline.py
    ├── README.md
    ├── data/ (customer_reviews.csv)
    └── outputs/ (confusion matrices, ROC curves, top keyword indicators, benchmark charts)
```

---

## 🎯 Task Summaries & Benchmark Results

### 🌸 [Task 1: Multi-Class Benchmark Classification Pipeline](./Task1/)
- **Objective**: Implement supervised multi-class classification on benchmark datasets (Fisher's Iris & Wine datasets), evaluate StandardScaler feature normalization, and visualize decision boundaries.
- **Models**: Multinomial Logistic Regression & Decision Tree Classifier.
- **Key Results**:
  - **Decision Tree**: **96.67% Accuracy**, **0.9666 F1-Score** (Iris)
  - **Logistic Regression**: **93.33% Accuracy**, **0.9333 F1-Score** (Iris) | **97.22% Accuracy** (Wine)
  - **Visuals**: Pairplots, confusion matrix heatmaps, 2D feature decision boundaries & PCA 2D projections.

---

### 🏡 [Task 2: Real Estate Housing Price Prediction (Ridge & Lasso)](./Task2/)
- **Objective**: Predict California median house values with continuous regression, domain feature engineering, multicollinearity handling, and residual error analysis.
- **Feature Engineering**: Engineered ratio terms (`BedroomsPerRoom`, `RoomsPerHousehold`, `PopulationPerHousehold`, `MedInc_x_HouseAge`, `MedInc_Per_Room`).
- **Models & Tuning**: Ordinary Least Squares (OLS), Ridge (L2), and Lasso (L1) tuned via 5-Fold `GridSearchCV`.
- **Key Results**:
  - **OLS Linear Regression**: $R^2 = 0.6317$, RMSE = $0.6947$, MAE = $0.5000$
  - **Ridge ($\alpha=10.0$)**: $R^2 = 0.6316$, RMSE = $0.6948$, MAE = $0.5000$
  - **Lasso ($\alpha=0.0022$)**: $R^2 = 0.6295$, RMSE = $0.6968$, MAE = $0.5010$
  - **Residuals**: Perfectly zero-centered normal distribution ($\mu = 0.0000$), confirming linear assumptions.

---

### 📊 [Task 3: Predictive Customer Churn Classifier (Random Forest & XGBoost)](./Task3/)
- **Objective**: Predict telecom customer attrition, resolve severe class imbalance (approx 3:1 ratio), and exceed the required **ROC-AUC > 80%** benchmark.
- **Preprocessing**: OneHotEncoder for categoricals, StandardScaler for numericals, and **SMOTE** on training data.
- **Models**: `RandomForestClassifier` (150 trees) & `XGBClassifier` (150 estimators).
- **Key Results**:
  - **Random Forest**: **ROC-AUC = 84.33%** (5-Fold CV AUC = **90.30%**), Recall = **74.60%**
  - **XGBoost**: **ROC-AUC = 84.32%** (5-Fold CV AUC = **93.43%**), Recall = **64.97%**
  - **Target Benchmark**: Exceeded **>80% ROC-AUC** threshold (**✅ PASSED**).
  - **Drivers of Churn**: Month-to-month contracts, high monthly charges, and short tenure.

---

### 💬 [Task 4: NLP Sentiment Classification on Customer Reviews (TF-IDF)](./Task4/)
- **Objective**: Ingest customer review corpus, clean and stem text with NLTK, extract TF-IDF embeddings (5,000 features), train sentiment classifiers, and build an interactive testing helper.
- **Models**: `MultinomialNB` (Laplace smoothed) and `LogisticRegression`.
- **Key Results**:
  - **Logistic Regression**: **88.67% Accuracy**, **0.8875 F1-Score**, **0.9515 ROC-AUC**
  - **Multinomial Naive Bayes**: **85.00% Accuracy**, **0.8514 F1-Score**, **0.9363 ROC-AUC**
  - **Interactive Inference**: Real-time `predict_sentiment()` function returning sentiment classification and confidence percentages on live custom phrases.

---

## ⚡ How to Run

Each task directory is fully self-contained with:
1. An **executed Jupyter Notebook (`.ipynb`)** with all cell outputs, charts, and tables pre-rendered.
2. A **standalone Python script (`pipeline.py`)** that can be executed directly from terminal:
   ```bash
   # Run Task 1
   cd Task1 && python pipeline.py

   # Run Task 2
   cd ../Task2 && python pipeline.py

   # Run Task 3
   cd ../Task3 && python pipeline.py

   # Run Task 4
   cd ../Task4 && python pipeline.py
   ```
