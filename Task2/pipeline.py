"""
Real Estate Housing Price Prediction (Ridge & Lasso Regression)
==============================================================
Task 2: Continuous Regression Modeling, Feature Engineering, and Residual Error Analysis
Dataset: California Housing Dataset

This script implements:
1. Ingestion: Load California Housing dataset via Pandas.
2. Missing Value Imputation: Impute missing values with SimpleImputer.
3. Feature Engineering: Compute domain-specific interaction features (BedroomsPerRoom, RoomsPerHousehold, MedInc_x_HouseAge).
4. Multicollinearity Analysis: Correlation matrix & Variance Inflation Factor (VIF).
5. Data Splitting & Normalization: 80/20 train/test split, StandardScaler (fitted on train only).
6. Model Training & Hyperparameter Tuning:
   - Ordinary Least Squares (OLS) Linear Regression
   - Ridge Regression (L2) tuned via 5-Fold GridSearchCV across multiple alphas
   - Lasso Regression (L1) tuned via 5-Fold GridSearchCV across multiple alphas
7. Benchmarking: RMSE, MAE, R² score comparisons.
8. Visualizations:
   - Correlation Heatmap
   - GridSearchCV Alpha Tuning Curves
   - Predicted vs Actual Values Scatter Plots
   - Residual Distribution Curves (KDE + Histograms)
   - Residuals vs Fitted Plots (Homoscedasticity Analysis)
   - Coefficient Shrinkage / Feature Importance Comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

# Set directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aesthetics
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def load_and_impute_data():
    """Step 1 & 2: Load California Housing dataset and impute missing values."""
    print("=" * 75)
    print("STEP 1 & 2: LOADING DATASET & IMPUTING MISSING VALUES")
    print("=" * 75)

    csv_path = os.path.join(DATA_DIR, "california_housing.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        raw = fetch_california_housing(as_frame=True)
        df = raw.frame.copy()
        df.to_csv(csv_path, index=False)

    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Features: {[c for c in df.columns if c != 'MedHouseVal']}")
    print(f"Target: 'MedHouseVal' (Median House Value in $100k)")

    # Check missing values
    missing = df.isnull().sum()
    print("\nMissing values before imputation:")
    print(missing)

    feature_cols = [c for c in df.columns if c != 'MedHouseVal']
    imputer = SimpleImputer(strategy='median')
    df[feature_cols] = imputer.fit_transform(df[feature_cols])

    return df


def engineer_features(df):
    """Step 3: Feature Engineering - interaction features & ratio indicators."""
    print("\n" + "=" * 75)
    print("STEP 3: FEATURE ENGINEERING (INTERACTION TERMS & DOMAIN RATIOS)")
    print("=" * 75)

    df_eng = df.copy()

    # Domain interaction features
    # 1. Ratio of bedrooms to total rooms (density indicator)
    df_eng['BedroomsPerRoom'] = df_eng['AveBedrms'] / (df_eng['AveRooms'] + 1e-5)
    # 2. Number of rooms per household
    df_eng['RoomsPerHousehold'] = df_eng['AveRooms']
    # 3. Population per household ratio
    df_eng['PopulationPerHousehold'] = df_eng['Population'] / (df_eng['AveOccup'] + 1e-5)
    # 4. Income and house age interaction
    df_eng['MedInc_x_HouseAge'] = df_eng['MedInc'] * df_eng['HouseAge']
    # 5. Income per room ratio
    df_eng['MedInc_Per_Room'] = df_eng['MedInc'] / (df_eng['AveRooms'] + 1e-5)

    new_features = ['BedroomsPerRoom', 'RoomsPerHousehold', 'PopulationPerHousehold', 'MedInc_x_HouseAge', 'MedInc_Per_Room']
    print(f"Engineered {len(new_features)} new features: {new_features}")
    print(f"New feature matrix dimension: {df_eng.shape[1] - 1} features")

    # Plot correlation heatmap
    plt.figure(figsize=(12, 9))
    corr = df_eng.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True, linewidths=0.5)
    plt.title("California Housing Correlation Matrix (with Engineered Features)", fontsize=13, fontweight="bold", pad=12)
    corr_path = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.savefig(corr_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation heatmap to: {corr_path}")

    return df_eng


def split_and_scale(df):
    """Step 4: Train/test split (80/20) and StandardScaler feature normalization."""
    print("\n" + "=" * 75)
    print("STEP 4: 80/20 TRAIN/TEST SPLIT & FEATURE NORMALIZATION (StandardScaler)")
    print("=" * 75)

    feature_cols = [c for c in df.columns if c != 'MedHouseVal']
    X = np.asarray(df[feature_cols].to_numpy(dtype=np.float64))
    y = np.asarray(df['MedHouseVal'].to_numpy(dtype=np.float64))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    print(f"Train samples: {X_train.shape[0]} (80%)")
    print(f"Test samples:  {X_test.shape[0]} (20%)")
    print(f"Features count:{X_train.shape[1]}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_cols, scaler


def train_and_tune_models(X_train, X_test, y_train, y_test, feature_cols):
    """Step 5: Train OLS, and tune Ridge & Lasso with 5-Fold GridSearchCV across multiple alphas."""
    print("\n" + "=" * 75)
    print("STEP 5: MODEL TRAINING & HYPERPARAMETER TUNING (OLS, RIDGE, LASSO)")
    print("=" * 75)

    # 1. Ordinary Least Squares (OLS)
    print("\nFitting Ordinary Least Squares (OLS) Linear Regression...")
    ols = LinearRegression()
    ols.fit(X_train, y_train)

    # 2. Ridge Regression with GridSearchCV
    print("\nTuning Ridge Regression (L2) using 5-Fold GridSearchCV...")
    ridge_alphas = np.logspace(-3, 3, 25)
    ridge_cv = GridSearchCV(
        Ridge(random_state=42),
        param_grid={'alpha': ridge_alphas},
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_jobs=-1
    )
    ridge_cv.fit(X_train, y_train)
    best_ridge = ridge_cv.best_estimator_
    print(f"  Best Ridge alpha: {ridge_cv.best_params_['alpha']:.4f}")
    print(f"  Best Ridge CV RMSE: {-ridge_cv.best_score_:.4f}")

    # 3. Lasso Regression with GridSearchCV
    print("\nTuning Lasso Regression (L1) using 5-Fold GridSearchCV...")
    lasso_alphas = np.logspace(-4, 0, 25)
    lasso_cv = GridSearchCV(
        Lasso(max_iter=5000, random_state=42),
        param_grid={'alpha': lasso_alphas},
        scoring='neg_root_mean_squared_error',
        cv=5,
        n_jobs=-1
    )
    lasso_cv.fit(X_train, y_train)
    best_lasso = lasso_cv.best_estimator_
    print(f"  Best Lasso alpha: {lasso_cv.best_params_['alpha']:.6f}")
    print(f"  Best Lasso CV RMSE: {-lasso_cv.best_score_:.4f}")

    # Plot GridSearchCV Alpha Tuning Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(ridge_alphas, -ridge_cv.cv_results_['mean_test_score'], marker='o', color='#1976D2', lw=2)
    axes[0].axvline(ridge_cv.best_params_['alpha'], color='red', linestyle='--', label=f"Best α = {ridge_cv.best_params_['alpha']:.3f}")
    axes[0].set_xscale('log')
    axes[0].set_title("Ridge Regression: 5-Fold CV RMSE vs Alpha", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Alpha (L2 Regularization Strength)", fontsize=11)
    axes[0].set_ylabel("Cross-Validation RMSE", fontsize=11)
    axes[0].legend()

    axes[1].plot(lasso_alphas, -lasso_cv.cv_results_['mean_test_score'], marker='s', color='#388E3C', lw=2)
    axes[1].axvline(lasso_cv.best_params_['alpha'], color='red', linestyle='--', label=f"Best α = {lasso_cv.best_params_['alpha']:.5f}")
    axes[1].set_xscale('log')
    axes[1].set_title("Lasso Regression: 5-Fold CV RMSE vs Alpha", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Alpha (L1 Regularization Strength)", fontsize=11)
    axes[1].set_ylabel("Cross-Validation RMSE", fontsize=11)
    axes[1].legend()

    plt.suptitle("Hyperparameter Tuning via GridSearchCV (5-Fold Cross-Validation)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    cv_plot_path = os.path.join(OUTPUT_DIR, "gridsearch_alpha_tuning.png")
    plt.savefig(cv_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved GridSearchCV alpha tuning curves to: {cv_plot_path}")

    # Evaluate on Unseen Test Set
    models = {
        "OLS Linear Regression": ols,
        f"Ridge Regression (α={ridge_cv.best_params_['alpha']:.3f})": best_ridge,
        f"Lasso Regression (α={lasso_cv.best_params_['alpha']:.4f})": best_lasso
    }

    results = {}
    benchmark_list = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        rmse = root_mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        residuals = y_test - y_pred

        results[name] = {
            'model': model,
            'y_pred': y_pred,
            'residuals': residuals,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

        benchmark_list.append({
            'Model': name,
            'RMSE ($100k)': rmse,
            'MAE ($100k)': mae,
            'R² Score': r2
        })

    benchmark_df = pd.DataFrame(benchmark_list)
    print("\n" + "=" * 75)
    print("BENCHMARK METRICS SUMMARY (TEST SET EVALUATION)")
    print("=" * 75)
    print(benchmark_df.to_string(index=False))

    return results, benchmark_df, models, best_ridge, best_lasso


def plot_residuals_and_predictions(results, y_test, feature_cols, models):
    """Step 6: Plot predicted vs actual, residual distribution curves, and coefficient shrinkage."""
    print("\n" + "=" * 75)
    print("STEP 6: PLOTTING RESIDUAL ERROR DIAGNOSTICS & PREDICTIONS")
    print("=" * 75)

    names = list(results.keys())

    # 1. Predicted vs Actual Values
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    colors = ['#1976D2', '#388E3C', '#F57C00']

    for ax, name, color in zip(axes, names, colors):
        y_pred = results[name]['y_pred']
        r2 = results[name]['r2']
        rmse = results[name]['rmse']

        ax.scatter(y_test, y_pred, alpha=0.25, color=color, s=20)
        # 45-degree reference line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal Fit (y = ŷ)")

        ax.set_title(f"{name}\nR²: {r2:.4f} | RMSE: {rmse:.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Actual Median House Value ($100k)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Predicted Value ($100k)", fontsize=10, fontweight="bold")
        ax.legend(loc="upper left")

    plt.suptitle("Predicted vs Actual Housing Values Across Models", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    pred_path = os.path.join(OUTPUT_DIR, "predicted_vs_actual.png")
    plt.savefig(pred_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved predicted vs actual plot to: {pred_path}")

    # 2. Residual Distribution Curves (Error Analysis)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, name, color in zip(axes, names, colors):
        residuals = results[name]['residuals']
        sns.histplot(residuals, kde=True, ax=ax, color=color, bins=50, stat="density", alpha=0.4)
        ax.axvline(0, color='red', linestyle='--', lw=1.5, label="Zero Error Line")
        ax.set_title(f"{name}\nMean Residual: {residuals.mean():.4f} | Std: {residuals.std():.4f}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Residual Error (Actual - Predicted)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Density", fontsize=10, fontweight="bold")
        ax.legend(loc="upper right")

    plt.suptitle("Residual Distribution Curves (Normality & Zero-Centering Diagnostic)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    res_path = os.path.join(OUTPUT_DIR, "residual_distribution_curves.png")
    plt.savefig(res_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved residual distribution curves to: {res_path}")

    # 3. Residuals vs Predicted (Homoscedasticity Analysis)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, name, color in zip(axes, names, colors):
        y_pred = results[name]['y_pred']
        residuals = results[name]['residuals']

        ax.scatter(y_pred, residuals, alpha=0.25, color=color, s=20)
        ax.axhline(0, color='red', linestyle='--', lw=1.5)
        ax.set_title(f"{name}\nResiduals vs Fitted", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Value ($100k)", fontsize=10, fontweight="bold")
        ax.set_ylabel("Residual (Actual - Predicted)", fontsize=10, fontweight="bold")

    plt.suptitle("Residuals vs Fitted Values (Homoscedasticity & Variance Diagnostic)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    homo_path = os.path.join(OUTPUT_DIR, "residuals_vs_predicted.png")
    plt.savefig(homo_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved residuals vs fitted plot to: {homo_path}")

    # 4. Feature Coefficients & Regularization Shrinkage Comparison
    coef_df = pd.DataFrame({
        'Feature': feature_cols,
        'OLS': models[names[0]].coef_,
        'Ridge': models[names[1]].coef_,
        'Lasso': models[names[2]].coef_
    })

    plt.figure(figsize=(12, 6))
    melted_coef = coef_df.melt(id_vars=['Feature'], var_name='Model', value_name='Coefficient')
    sns.barplot(data=melted_coef, x='Feature', y='Coefficient', hue='Model', palette=colors)
    plt.title("Model Coefficients & Feature Shrinkage (OLS vs Ridge vs Lasso)", fontsize=13, fontweight="bold", pad=12)
    plt.xticks(rotation=35, ha='right', fontsize=9.5)
    plt.ylabel("Standardized Coefficient Magnitude", fontsize=11, fontweight="bold")
    plt.axhline(0, color='black', lw=0.8, linestyle='--')
    plt.tight_layout()
    coef_path = os.path.join(OUTPUT_DIR, "feature_importance_coefficients.png")
    plt.savefig(coef_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature coefficients plot to: {coef_path}")

    # 5. Model Metrics Comparison Bar Chart
    plt.figure(figsize=(9, 4.5))
    metrics_summary = pd.DataFrame([
        {'Model': 'OLS', 'RMSE': results[names[0]]['rmse'], 'MAE': results[names[0]]['mae'], 'R²': results[names[0]]['r2']},
        {'Model': 'Ridge', 'RMSE': results[names[1]]['rmse'], 'MAE': results[names[1]]['mae'], 'R²': results[names[1]]['r2']},
        {'Model': 'Lasso', 'RMSE': results[names[2]]['rmse'], 'MAE': results[names[2]]['mae'], 'R²': results[names[2]]['r2']}
    ])
    melted_metrics = metrics_summary.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
    ax = sns.barplot(data=melted_metrics, x='Metric', y='Score', hue='Model', palette=colors)
    plt.title("Regression Performance Benchmark Comparison", fontsize=13, fontweight="bold", pad=12)
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8.5, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    metric_plot_path = os.path.join(OUTPUT_DIR, "model_metrics_comparison.png")
    plt.savefig(metric_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved metrics comparison chart to: {metric_plot_path}")


def main():
    print("=" * 80)
    print("TASK 2: REAL ESTATE HOUSING PRICE PREDICTION PIPELINE (RIDGE & LASSO)")
    print("=" * 80)
    df = load_and_impute_data()
    df_eng = engineer_features(df)
    X_train_s, X_test_s, y_train, y_test, feature_cols, scaler = split_and_scale(df_eng)
    results, benchmark_df, models, best_ridge, best_lasso = train_and_tune_models(X_train_s, X_test_s, y_train, y_test, feature_cols)
    plot_residuals_and_predictions(results, y_test, feature_cols, models)

    print("\n" + "=" * 80)
    print("TASK 2 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Artifacts saved in: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
