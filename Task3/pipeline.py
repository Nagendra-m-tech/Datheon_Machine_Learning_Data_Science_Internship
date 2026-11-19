"""
Predictive Customer Churn Classifier (Random Forest & XGBoost)
==============================================================
Task 3: Solve Severe Class Imbalance and Optimize ROC-AUC for Customer Attrition Prediction
Dataset: Telco Customer Churn Dataset

Pipeline stages:
1. Data Ingestion: Load Telco Customer Churn dataset using Pandas.
2. Data Preprocessing & Cleaning: Convert TotalCharges to numeric, impute nulls, encode target binary.
3. Feature Encoding & Scaling: OneHotEncoder for categoricals, StandardScaler for numericals.
4. Class Imbalance Handling: Apply SMOTE (Synthetic Minority Over-sampling Technique) on train data.
5. Model Training: Train RandomForestClassifier and XGBClassifier with cross-validation.
6. Evaluation: ROC-AUC score (>80% benchmark target), accuracy, precision, recall, F1-score.
7. Visualizations:
   - ROC-AUC curves comparison
   - Confusion matrix heatmaps
   - Top 10 feature importances for both models
   - Benchmark performance metrics
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def load_and_clean_data():
    """Step 1 & 2: Ingest Telco data and clean numeric types & missing values."""
    print("=" * 75)
    print("STEP 1 & 2: LOADING & CLEANING TELCO CUSTOMER CHURN DATASET")
    print("=" * 75)

    csv_path = os.path.join(DATA_DIR, "telco_customer_churn.csv")
    df = pd.read_csv(csv_path)

    print(f"Raw dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")

    # Drop customerID (non-predictive identifier)
    if 'customerID' in df.columns:
        df.drop(columns=['customerID'], inplace=True)

    # Convert TotalCharges to numeric (handles blank strings ' ' -> NaN)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    null_total = df['TotalCharges'].isnull().sum()
    print(f"Blank TotalCharges values detected: {null_total}")

    # Impute missing TotalCharges with median
    median_val = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_val)
    print(f"Imputed missing TotalCharges with median ({median_val:.2f})")

    # Drop any remaining rows with missing values
    df.dropna(inplace=True)

    # Convert target Churn: Yes -> 1, No -> 0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    churn_counts = df['Churn'].value_counts()
    churn_pct = df['Churn'].value_counts(normalize=True) * 100
    print(f"\nTarget Class Distribution:")
    print(f"  Non-Churn (0): {churn_counts[0]} ({churn_pct[0]:.2f}%)")
    print(f"  Churn (1):     {churn_counts[1]} ({churn_pct[1]:.2f}%)")
    print("  -> Significant class imbalance detected (approx 3:1 ratio).")

    return df


def preprocess_and_split(df):
    """Step 3 & 4: Encode categoricals, scale numericals, 80/20 split, and apply SMOTE."""
    print("\n" + "=" * 75)
    print("STEP 3 & 4: PREPROCESSING, 80/20 SPLIT, AND SMOTE RESAMPLING")
    print("=" * 75)

    X = df.drop(columns=['Churn'])
    y = df['Churn'].values

    # Differentiate categorical and numerical features
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    cat_cols = [c for c in X.columns if c not in num_cols]

    print(f"Numerical features ({len(num_cols)}): {num_cols}")
    print(f"Categorical features ({len(cat_cols)}): {cat_cols}")

    # ColumnTransformer for encoding and scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat_cols)
        ]
    )

    # Stratified 80/20 train/test split
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Fit preprocessor strictly on training data
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Extract transformed feature names
    cat_encoder = preprocessor.named_transformers_['cat']
    encoded_cat_names = cat_encoder.get_feature_names_out(cat_cols)
    all_feature_names = list(num_cols) + list(encoded_cat_names)

    print(f"Transformed feature matrix: {X_train.shape[1]} features")
    print(f"Train samples before SMOTE: {X_train.shape[0]} (Non-Churn: {(y_train==0).sum()}, Churn: {(y_train==1).sum()})")

    # Apply SMOTE to balance training classes
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"Train samples after SMOTE:  {X_train_res.shape[0]} (Non-Churn: {(y_train_res==0).sum()}, Churn: {(y_train_res==1).sum()})")
    print(f"Test samples (unbalanced, real-world distribution): {X_test.shape[0]}")

    return X_train_res, X_test, y_train_res, y_test, all_feature_names


def train_and_benchmark(X_train, X_test, y_train, y_test, feature_names):
    """Step 5: Train RandomForestClassifier and XGBClassifier, benchmark ROC-AUC."""
    print("\n" + "=" * 75)
    print("STEP 5: MODEL TRAINING & BENCHMARKING (RANDOM FOREST & XGBOOST)")
    print("=" * 75)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=150,
            max_depth=9,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.07,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    }

    results = {}
    benchmark_rows = []

    for name, clf in models.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)

        # Probabilities for ROC-AUC
        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, target_names=['Retained (0)', 'Churn (1)'], digits=4)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        # 5-fold Cross-Validation ROC-AUC score
        cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)

        results[name] = {
            'model': clf,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'roc_auc': auc,
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'confusion_matrix': cm,
            'report': cr,
            'fpr': fpr,
            'tpr': tpr
        }

        benchmark_rows.append({
            'Model': name,
            'ROC-AUC Score': auc,
            '5-Fold CV AUC': f"{cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
            'Accuracy': acc,
            'Precision (Churn)': prec,
            'Recall (Churn)': rec,
            'F1-Score (Churn)': f1
        })

        print(f"--- {name} Results ---")
        print(f"Test ROC-AUC:    {auc:.4f} (Goal > 80%: {'✅ PASSED' if auc > 0.80 else '❌ FAILED'})")
        print(f"5-Fold CV AUC:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"Accuracy:        {acc:.4f} ({acc*100:.2f}%)")
        print(f"Recall (Churn):  {rec:.4f}")
        print(f"F1-Score:        {f1:.4f}")
        print("\nClassification Report:")
        print(cr)

    benchmark_df = pd.DataFrame(benchmark_rows)
    print("\nBENCHMARK COMPARISON TABLE:")
    print(benchmark_df.to_string(index=False))

    return results, benchmark_df, models


def plot_visualizations(results, models, feature_names):
    """Step 6: Plot ROC-AUC curves, confusion matrices, and top 10 feature importances."""
    print("\n" + "=" * 75)
    print("STEP 6: PLOTTING ROC-AUC CURVES, CONFUSION MATRICES & FEATURE IMPORTANCES")
    print("=" * 75)

    # 1. ROC-AUC Curves Plot
    plt.figure(figsize=(8, 6.5))
    colors = ['#1976D2', '#388E3C']

    for (name, res), color in zip(results.items(), colors):
        plt.plot(res['fpr'], res['tpr'], color=color, lw=2.5,
                 label=f"{name} (AUC = {res['roc_auc']:.4f})")

    plt.plot([0, 1], [0, 1], color='#888888', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.50)')
    plt.axhline(0.80, color='red', linestyle=':', lw=1.2, label='Target Benchmark (AUC = 0.80)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC-AUC) Curves', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "roc_auc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC-AUC curves -> {roc_path}")

    # 2. Confusion Matrices Side-by-Side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    cm_cmaps = ['Blues', 'Greens']

    for ax, (name, res), cmap in zip(axes, results.items(), cm_cmaps):
        sns.heatmap(
            res['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap=cmap,
            cbar=True,
            ax=ax,
            xticklabels=['Retained (0)', 'Churn (1)'],
            yticklabels=['Retained (0)', 'Churn (1)'],
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_title(f"{name}\nROC-AUC: {res['roc_auc']:.2%} | Recall: {res['recall']:.2%}", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')

    plt.suptitle("Customer Churn Confusion Matrix Heatmaps", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices -> {cm_path}")

    # 3. Top 10 Feature Importances
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (name, clf), color in zip(axes, models.items(), ['#1976D2', '#388E3C']):
        importances = clf.feature_importances_
        idx = np.argsort(importances)[::-1][:10]
        top_features = [feature_names[i] for i in idx]
        top_importances = importances[idx]

        sns.barplot(x=top_importances, y=top_features, ax=ax, color=color, alpha=0.85)
        ax.set_title(f"{name}: Top 10 Most Influential Features", fontsize=12, fontweight='bold')
        ax.set_xlabel('Relative Feature Importance Score', fontsize=11, fontweight='bold')
        ax.set_ylabel('Feature Name', fontsize=11, fontweight='bold')

        for i, v in enumerate(top_importances):
            ax.text(v + 0.002, i, f"{v:.3f}", va='center', fontsize=9, fontweight='bold')

    plt.suptitle("Top 10 Feature Importances for Customer Churn Prediction", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fi_path = os.path.join(OUTPUT_DIR, "feature_importance.png")
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved feature importances -> {fi_path}")

    # 4. Benchmark Metrics Comparison Bar Chart
    plt.figure(figsize=(9, 4.5))
    metrics_summary = pd.DataFrame([
        {'Model': 'Random Forest', 'ROC-AUC': results['Random Forest']['roc_auc'], 'Accuracy': results['Random Forest']['accuracy'], 'Recall': results['Random Forest']['recall'], 'F1': results['Random Forest']['f1']},
        {'Model': 'XGBoost', 'ROC-AUC': results['XGBoost']['roc_auc'], 'Accuracy': results['XGBoost']['accuracy'], 'Recall': results['XGBoost']['recall'], 'F1': results['XGBoost']['f1']}
    ])
    melted = metrics_summary.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
    ax = sns.barplot(data=melted, x='Metric', y='Score', hue='Model', palette=['#1976D2', '#388E3C'])
    plt.title("Customer Churn Model Benchmark Performance", fontsize=13, fontweight='bold', pad=12)
    plt.ylim(0.5, 1.0)
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8.5, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    bench_chart_path = os.path.join(OUTPUT_DIR, "model_benchmark_comparison.png")
    plt.savefig(bench_chart_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved benchmark comparison chart -> {bench_chart_path}")


def main():
    print("=" * 80)
    print("TASK 3: PREDICTIVE CUSTOMER CHURN CLASSIFIER (RANDOM FOREST & XGBOOST)")
    print("=" * 80)
    df = load_and_clean_data()
    X_train, X_test, y_train, y_test, feature_names = preprocess_and_split(df)
    results, benchmark_df, models = train_and_benchmark(X_train, X_test, y_train, y_test, feature_names)
    plot_visualizations(results, models, feature_names)
    print("\n" + "=" * 80)
    print("TASK 3 COMPLETED SUCCESSFULLY!")
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
