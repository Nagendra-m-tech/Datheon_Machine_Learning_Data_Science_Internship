"""
Multi-Class Benchmark Classification Pipeline (Iris & Wine)
===========================================================
End-to-End Supervised Machine Learning Pipeline implementing:
1. Dataset ingestion using Pandas (Iris and Wine benchmarks)
2. Exploratory Data Analysis & Pairplot visualization with Seaborn
3. Train/Test dataset splitting (80/20 stratified split)
4. Feature Normalization using StandardScaler
5. Model Training (Logistic Regression & Decision Tree Classifier)
6. Model Benchmarking (Accuracy, Precision, Recall, F1-Score, Classification Report)
7. Confusion Matrix Heatmaps
8. Decision Boundary Visualizations (2D Feature Space & PCA Space)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

# Directory setup
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def prepare_datasets():
    """Ensure Iris and Wine CSV datasets are saved and available for Pandas loading."""
    iris_path = os.path.join(DATA_DIR, "iris.csv")
    if not os.path.exists(iris_path):
        raw = load_iris(as_frame=True)
        df_iris = raw.frame.copy()
        df_iris.rename(columns={'target': 'species_id'}, inplace=True)
        df_iris['species'] = df_iris['species_id'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
        df_iris.to_csv(iris_path, index=False)
        print(f"Saved {iris_path}")

    wine_path = os.path.join(DATA_DIR, "wine.csv")
    if not os.path.exists(wine_path):
        raw_w = load_wine(as_frame=True)
        df_wine = raw_w.frame.copy()
        df_wine.rename(columns={'target': 'class_id'}, inplace=True)
        df_wine['class_name'] = df_wine['class_id'].map({0: 'class_0', 1: 'class_1', 2: 'class_2'})
        df_wine.to_csv(wine_path, index=False)
        print(f"Saved {wine_path}")


def load_dataset(dataset_name="iris"):
    """
    Step 1: Load the benchmark dataset using Pandas.
    """
    print(f"\n{'='*75}")
    print(f"STEP 1: LOADING BENCHMARK DATASET ({dataset_name.upper()}) USING PANDAS")
    print(f"{'='*75}")

    if dataset_name.lower() == "iris":
        csv_path = os.path.join(DATA_DIR, "iris.csv")
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c not in ['species_id', 'species']]
        target_col = 'species'
        target_classes = ['setosa', 'versicolor', 'virginica']
    else:
        csv_path = os.path.join(DATA_DIR, "wine.csv")
        df = pd.read_csv(csv_path)
        feature_cols = [c for c in df.columns if c not in ['class_id', 'class_name']]
        target_col = 'class_name'
        target_classes = ['class_0', 'class_1', 'class_2']

    print(f"Loaded '{csv_path}' successfully.")
    print(f"Dimensions: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Features ({len(feature_cols)}): {feature_cols}")
    print(f"Target Column: '{target_col}' | Classes: {target_classes}")
    print("\nHead (First 5 Rows):")
    print(df.head())
    print("\nTarget Class Distribution:")
    print(df[target_col].value_counts())

    return df, feature_cols, target_col, target_classes


def perform_eda(df, feature_cols, target_col, dataset_name="iris"):
    """
    Step 2: Perform Exploratory Data Analysis & Pairplot with Seaborn.
    """
    print(f"\n{'='*75}")
    print(f"STEP 2: EXPLORATORY DATA ANALYSIS & SEABORN PAIRPLOTS")
    print(f"{'='*75}")

    # Pairplot (use first 4 features for high-dimensional datasets like Wine)
    plot_features = feature_cols[:4] if len(feature_cols) > 4 else feature_cols
    print(f"Generating pairplot for features: {plot_features}...")

    plot_df = df[plot_features + [target_col]].copy()
    pp = sns.pairplot(
        plot_df,
        hue=target_col,
        diag_kind="kde",
        palette="viridis",
        markers=["o", "s", "D"],
        plot_kws={'alpha': 0.8, 's': 45}
    )
    pp.fig.subplots_adjust(top=0.93)
    pp.fig.suptitle(f"{dataset_name.upper()} Dataset: Exploratory Pairplot Grouped by Class", fontsize=14, fontweight="bold")
    pairplot_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_pairplot.png")
    pp.savefig(pairplot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved pairplot -> {pairplot_file}")

    # Correlation Matrix Heatmap
    plt.figure(figsize=(8, 6.5))
    corr = df[feature_cols].corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f"{dataset_name.upper()} Feature Correlation Heatmap", fontsize=13, fontweight="bold", pad=12)
    corr_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_correlation_heatmap.png")
    plt.savefig(corr_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation heatmap -> {corr_file}")


def split_and_scale(df, feature_cols, target_col, test_size=0.2, random_state=42):
    """
    Step 3: Split train/test (80/20 stratified) and normalize using StandardScaler.
    Prevents data leakage by fitting StandardScaler exclusively on training data.
    """
    print(f"\n{'='*75}")
    print(f"STEP 3: 80/20 TRAIN/TEST SPLIT & FEATURE SCALING (StandardScaler)")
    print(f"{'='*75}")

    # Convert to pure numpy arrays to ensure compatibility
    X = np.asarray(df[feature_cols].to_numpy(dtype=np.float64))
    y = np.asarray(df[target_col].tolist())

    # Stratified 80/20 train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    print(f"Stratified Split (80/20) Completed:")
    print(f"  Training set size: {X_train.shape[0]} samples ({1-test_size:.0%})")
    print(f"  Testing set size:  {X_test.shape[0]} samples ({test_size:.0%})")
    print(f"  Feature count:     {X_train.shape[1]}")

    # Feature Scaling using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nStandardScaler Normalization Applied:")
    print(f"  X_train_scaled Mean (per feature): {np.round(X_train_scaled.mean(axis=0), 4)}")
    print(f"  X_train_scaled Std  (per feature): {np.round(X_train_scaled.std(axis=0), 4)}")

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_and_benchmark_models(X_train, X_test, y_train, y_test, target_classes):
    """
    Step 4: Train LogisticRegression and DecisionTreeClassifier, benchmark metrics.
    """
    print(f"\n{'='*75}")
    print(f"STEP 4: MODEL TRAINING & BENCHMARKING (ACCURACY, PRECISION, RECALL, F1)")
    print(f"{'='*75}")

    models = {
        "Logistic Regression": LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="gini",
            max_depth=3,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\nFitting {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

        prec_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        cm = confusion_matrix(y_test, y_pred, labels=target_classes)
        cr = classification_report(y_test, y_pred, target_names=target_classes, digits=4)

        results[name] = {
            "model": model,
            "y_pred": y_pred,
            "accuracy": acc,
            "precision_macro": prec_macro,
            "recall_macro": rec_macro,
            "f1_macro": f1_macro,
            "precision_weighted": prec_weighted,
            "recall_weighted": rec_weighted,
            "f1_weighted": f1_weighted,
            "confusion_matrix": cm,
            "classification_report": cr
        }

        print(f"--- {name} Benchmark Results ---")
        print(f"Accuracy:          {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision (Macro): {prec_macro:.4f}")
        print(f"Recall (Macro):    {rec_macro:.4f}")
        print(f"F1-Score (Macro):  {f1_macro:.4f}")
        print(f"F1-Score (Weight): {f1_weighted:.4f}")
        print("\nClassification Report:")
        print(cr)

    # Benchmark summary DataFrame
    benchmark_df = pd.DataFrame([
        {
            "Model": name,
            "Accuracy": res["accuracy"],
            "Precision (Macro)": res["precision_macro"],
            "Recall (Macro)": res["recall_macro"],
            "F1-Score (Macro)": res["f1_macro"],
            "Precision (Weighted)": res["precision_weighted"],
            "Recall (Weighted)": res["recall_weighted"],
            "F1-Score (Weighted)": res["f1_weighted"]
        }
        for name, res in results.items()
    ])

    print("\nBENCHMARK SUMMARY COMPARISON TABLE:")
    print(benchmark_df.to_string(index=False))

    return results, benchmark_df


def plot_confusion_matrices(results, target_classes, dataset_name="iris"):
    """
    Step 5.1: Plot Confusion Matrix Heatmaps side-by-side using Seaborn.
    """
    print(f"\n{'='*75}")
    print(f"STEP 5.1: PLOTTING CONFUSION MATRIX HEATMAPS")
    print(f"{'='*75}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    cm_palettes = ["Blues", "Greens"]

    for ax, (name, res), cmap in zip(axes, results.items(), cm_palettes):
        cm = res["confusion_matrix"]
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap=cmap,
            cbar=True,
            ax=ax,
            xticklabels=target_classes,
            yticklabels=target_classes,
            annot_kws={"size": 15, "weight": "bold"}
        )
        ax.set_title(f"{name}\nAccuracy: {res['accuracy']:.2%} | F1-Macro: {res['f1_macro']:.4f}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=11, fontweight="bold")

    plt.suptitle(f"{dataset_name.upper()} Dataset: Confusion Matrix Comparison", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    cm_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_confusion_matrices.png")
    plt.savefig(cm_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix heatmaps -> {cm_file}")


def plot_decision_boundaries(df, feature_cols, target_col, target_classes, dataset_name="iris"):
    """
    Step 5.2: Plot Decision Boundary Visualizations.
    Compares Logistic Regression (linear hyperplanes) against Decision Tree (orthogonal step boundaries).
    """
    print(f"\n{'='*75}")
    print(f"STEP 5.2: PLOTTING DECISION BOUNDARY VISUALIZATIONS")
    print(f"{'='*75}")

    class_to_idx = {name: i for i, name in enumerate(target_classes)}
    y_int = np.array([class_to_idx[v] for v in df[target_col]])

    # 1. 2D Feature Space (e.g. Petal Length vs Petal Width)
    if dataset_name.lower() == "iris":
        feat1, feat2 = "petal length (cm)", "petal width (cm)"
        if feat1 not in df.columns:
            feat1, feat2 = feature_cols[2], feature_cols[3]
    else:
        feat1, feat2 = feature_cols[0], feature_cols[1]

    X_2d = np.asarray(df[[feat1, feat2]].to_numpy(dtype=np.float64))

    X_tr_2d, X_te_2d, y_tr_2d, y_te_2d = train_test_split(
        X_2d, y_int, test_size=0.2, random_state=42, stratify=y_int
    )
    scaler_2d = StandardScaler()
    X_tr_2d_s = scaler_2d.fit_transform(X_tr_2d)
    X_te_2d_s = scaler_2d.transform(X_te_2d)

    models_2d = {
        "Logistic Regression (Linear Boundaries)": LogisticRegression(max_iter=500, random_state=42),
        "Decision Tree (Orthogonal Axis-Aligned Splits)": DecisionTreeClassifier(max_depth=3, random_state=42)
    }

    # Meshgrid
    x_min, x_max = X_tr_2d_s[:, 0].min() - 0.7, X_tr_2d_s[:, 0].max() + 0.7
    y_min, y_max = X_tr_2d_s[:, 1].min() - 0.7, X_tr_2d_s[:, 1].max() + 0.7
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    cmap_light = ListedColormap(["#E3F2FD", "#E8F5E9", "#FFF3E0"])
    colors = ["#1976D2", "#388E3C", "#F57C00"]
    markers = ["o", "s", "^"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (title, clf) in zip(axes, models_2d.items()):
        clf.fit(X_tr_2d_s, y_tr_2d)
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        ax.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.9)
        ax.contour(xx, yy, Z, colors="#555555", linewidths=0.8, alpha=0.7)

        for i, (cls_name, color, marker) in enumerate(zip(target_classes, colors, markers)):
            m_tr = (y_tr_2d == i)
            ax.scatter(X_tr_2d_s[m_tr, 0], X_tr_2d_s[m_tr, 1], c=color, label=f"{cls_name} (Train)",
                       marker=marker, edgecolor="k", s=50, alpha=0.7)
            m_te = (y_te_2d == i)
            ax.scatter(X_te_2d_s[m_te, 0], X_te_2d_s[m_te, 1], c=color, label=f"{cls_name} (Test)",
                       marker=marker, edgecolor="white", linewidth=1.5, s=90, alpha=1.0)

        acc_2d = accuracy_score(y_te_2d, clf.predict(X_te_2d_s))
        ax.set_title(f"{title}\n2D Test Accuracy: {acc_2d:.2%}", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"{feat1} (Standardized)", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"{feat2} (Standardized)", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", frameon=True, fontsize=8.5)

    plt.suptitle(f"{dataset_name.upper()}: Decision Boundary Comparison in 2D Feature Space ({feat1} vs {feat2})", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    db_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_decision_boundaries_2d.png")
    plt.savefig(db_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved 2D feature decision boundaries -> {db_file}")

    # 2. PCA 2D Space
    X_all = np.asarray(df[feature_cols].to_numpy(dtype=np.float64))
    scaler_all = StandardScaler()
    X_all_s = scaler_all.fit_transform(X_all)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_all_s)
    var_exp = pca.explained_variance_ratio_

    X_tr_p, X_te_p, y_tr_p, y_te_p = train_test_split(
        X_pca, y_int, test_size=0.2, random_state=42, stratify=y_int
    )

    px_min, px_max = X_pca[:, 0].min() - 0.7, X_pca[:, 0].max() + 0.7
    py_min, py_max = X_pca[:, 1].min() - 0.7, X_pca[:, 1].max() + 0.7
    pxx, pyy = np.meshgrid(np.arange(px_min, px_max, 0.02), np.arange(py_min, py_max, 0.02))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, (title, clf) in zip(axes, models_2d.items()):
        clf.fit(X_tr_p, y_tr_p)
        Z_p = clf.predict(np.c_[pxx.ravel(), pyy.ravel()]).reshape(pxx.shape)

        ax.contourf(pxx, pyy, Z_p, cmap=cmap_light, alpha=0.9)
        ax.contour(pxx, pyy, Z_p, colors="#555555", linewidths=0.8, alpha=0.7)

        for i, (cls_name, color, marker) in enumerate(zip(target_classes, colors, markers)):
            m_tr = (y_tr_p == i)
            ax.scatter(X_tr_p[m_tr, 0], X_tr_p[m_tr, 1], c=color, label=f"{cls_name} (Train)",
                       marker=marker, edgecolor="k", s=50, alpha=0.7)
            m_te = (y_te_p == i)
            ax.scatter(X_te_p[m_te, 0], X_te_p[m_te, 1], c=color, label=f"{cls_name} (Test)",
                       marker=marker, edgecolor="white", linewidth=1.5, s=90, alpha=1.0)

        acc_p = accuracy_score(y_te_p, clf.predict(X_te_p))
        ax.set_title(f"{title}\nPCA 2D Test Accuracy: {acc_p:.2%}", fontsize=12, fontweight="bold")
        ax.set_xlabel(f"Principal Component 1 ({var_exp[0]:.1%} Variance)", fontsize=11, fontweight="bold")
        ax.set_ylabel(f"Principal Component 2 ({var_exp[1]:.1%} Variance)", fontsize=11, fontweight="bold")
        ax.legend(loc="upper left", frameon=True, fontsize=8.5)

    plt.suptitle(f"{dataset_name.upper()}: Decision Boundary in PCA Space ({var_exp.sum():.1%} Variance Explained)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    pca_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_decision_boundaries_pca.png")
    plt.savefig(pca_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved PCA decision boundaries -> {pca_file}")


def plot_benchmark_bar_chart(benchmark_df, dataset_name="iris"):
    """Plot summary comparison bar chart across all metrics."""
    melted = benchmark_df.melt(id_vars=["Model"], var_name="Metric", value_name="Score")

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(data=melted, x="Metric", y="Score", hue="Model", palette=["#2196F3", "#4CAF50"])
    plt.title(f"{dataset_name.upper()} Model Performance Benchmark Comparison", fontsize=13, fontweight="bold", pad=12)
    plt.ylim(0.7, 1.05)
    plt.ylabel("Score", fontsize=11, fontweight="bold")
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.legend(title="Classifier", loc="lower right")

    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8, xytext=(0, 3), textcoords='offset points')

    plt.tight_layout()
    chart_file = os.path.join(OUTPUT_DIR, f"{dataset_name}_benchmark_chart.png")
    plt.savefig(chart_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved benchmark comparison bar chart -> {chart_file}")


def run_pipeline(dataset_name="iris"):
    """Executes the full pipeline for a given benchmark dataset."""
    df, feature_cols, target_col, target_classes = load_dataset(dataset_name)
    perform_eda(df, feature_cols, target_col, dataset_name)
    X_train_s, X_test_s, y_train, y_test, scaler = split_and_scale(df, feature_cols, target_col, test_size=0.2, random_state=42)
    results, benchmark_df = train_and_benchmark_models(X_train_s, X_test_s, y_train, y_test, target_classes)
    plot_confusion_matrices(results, target_classes, dataset_name)
    plot_decision_boundaries(df, feature_cols, target_col, target_classes, dataset_name)
    plot_benchmark_bar_chart(benchmark_df, dataset_name)

    print(f"\nPipeline successfully completed for {dataset_name.upper()}!")
    return benchmark_df


if __name__ == "__main__":
    prepare_datasets()

    print("=" * 80)
    print("EXECUTING MULTI-CLASS BENCHMARK PIPELINE: IRIS DATASET")
    print("=" * 80)
    iris_benchmark = run_pipeline("iris")

    print("\n" + "=" * 80)
    print("EXECUTING MULTI-CLASS BENCHMARK PIPELINE: WINE DATASET")
    print("=" * 80)
    wine_benchmark = run_pipeline("wine")
