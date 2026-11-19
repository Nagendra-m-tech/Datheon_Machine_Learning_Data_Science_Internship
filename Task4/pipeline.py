"""
NLP Sentiment Classification on Customer Reviews (TF-IDF)
=========================================================
Task 4: Preprocess Raw Textual Corpus, Extract TF-IDF Embeddings, and Classify Review Polarity
Dataset: Customer Product & Entertainment Reviews (50,000 samples)

Pipeline stages:
1. Data Ingestion: Ingest review corpus and label mapping (positive -> 1, negative -> 0).
2. Text Preprocessing: HTML tag stripping, regex token filtering, stopword removal, and Porter stemming.
3. Feature Extraction: TF-IDF vectorization with unigrams & bigrams (max_features=5000).
4. Model Training: Multinomial Naive Bayes (MultinomialNB) and Logistic Regression.
5. Evaluation & Benchmarking: Accuracy, Precision, Recall, F1-score, ROC-AUC score, Confusion Matrix.
6. Most Informative Features: Extract top positive and negative indicator words.
7. Interactive Inference: predict_sentiment() function for testing custom live phrases.
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
    confusion_matrix
)

# Directories
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

# Initialize NLP resources
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

stemmer = PorterStemmer()


def clean_text(text):
    """Clean and preprocess raw text input."""
    if not isinstance(text, str):
        return ""
    # 1. Strip HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # 2. Strip URLs
    text = re.sub(r'http\S+|www\S+', ' ', text)
    # 3. Strip non-alphabetic characters
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # 4. Lowercase and tokenize
    tokens = text.lower().split()
    # 5. Remove stopwords and apply Porter stemming
    filtered = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return ' '.join(filtered)


def load_and_preprocess_data(sample_size=12000):
    """Step 1 & 2: Ingest dataset and clean text corpus."""
    print("=" * 75)
    print("STEP 1 & 2: INGESTION & TEXT PREPROCESSING (CLEANING, TOKENIZING, STEMMING)")
    print("=" * 75)

    csv_path = os.path.join(DATA_DIR, "customer_reviews.csv")
    df = pd.read_csv(csv_path)
    print(f"Total reviews in database: {df.shape[0]} records")

    # Sample balanced subset for efficient high-quality training
    if len(df) > sample_size:
        pos_df = df[df['sentiment'] == 'positive'].sample(sample_size // 2, random_state=42)
        neg_df = df[df['sentiment'] == 'negative'].sample(sample_size // 2, random_state=42)
        df = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
        print(f"Sampled balanced subset: {len(df)} reviews ({sample_size//2} positive, {sample_size//2} negative)")

    # Map target: positive -> 1, negative -> 0
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    print("\nRunning NLP text cleaning pipeline...")
    df['cleaned_review'] = df['review'].apply(clean_text)

    print("\nSample Preprocessed Reviews:")
    for i in range(2):
        print(f"\n[Raw #{i+1}]: {df['review'].iloc[i][:110]}...")
        print(f"[Cleaned #{i+1}]: {df['cleaned_review'].iloc[i][:110]}...")

    return df


def vectorize_and_split(df):
    """Step 3: 80/20 train/test split and TF-IDF extraction (max_features=5000)."""
    print("\n" + "=" * 75)
    print("STEP 3: 80/20 SPLIT & TF-IDF FEATURE EXTRACTION (max_features=5000)")
    print("=" * 75)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['cleaned_review'], df['label'].values,
        test_size=0.20, random_state=42, stratify=df['label'].values
    )

    print(f"Train split: {len(X_train_raw)} reviews (80%)")
    print(f"Test split:  {len(X_test_raw)} reviews (20%)")

    # TF-IDF Vectorization with unigrams and bigrams
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    X_test_tfidf = vectorizer.transform(X_test_raw)

    print(f"Extracted TF-IDF Vocabulary Size: {X_train_tfidf.shape[1]} n-gram terms")
    print(f"TF-IDF Matrix Shape (Train): {X_train_tfidf.shape}")
    print(f"TF-IDF Matrix Shape (Test):  {X_test_tfidf.shape}")

    return X_train_tfidf, X_test_tfidf, y_train, y_test, vectorizer


def train_and_benchmark(X_train, X_test, y_train, y_test, vectorizer):
    """Step 4: Train MultinomialNB and LogisticRegression, benchmark metrics."""
    print("\n" + "=" * 75)
    print("STEP 4: MODEL TRAINING & METRICS BENCHMARKING (NAIVE BAYES & LOGISTIC REGRESSION)")
    print("=" * 75)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42)
    }

    results = {}
    benchmark_rows = []

    for name, clf in models.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)

        y_prob = clf.predict_proba(X_test)[:, 1]
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, target_names=['Negative (0)', 'Positive (1)'], digits=4)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        results[name] = {
            'model': clf,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': auc,
            'confusion_matrix': cm,
            'report': cr,
            'fpr': fpr,
            'tpr': tpr
        }

        benchmark_rows.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC Score': auc
        })

        print(f"--- {name} Results ---")
        print(f"Accuracy:      {acc:.4f} ({acc*100:.2f}%)")
        print(f"Precision:     {prec:.4f}")
        print(f"Recall:        {rec:.4f}")
        print(f"F1-Score:      {f1:.4f}")
        print(f"ROC-AUC:       {auc:.4f}")
        print("\nClassification Report:")
        print(cr)

    benchmark_df = pd.DataFrame(benchmark_rows)
    print("\nBENCHMARK COMPARISON TABLE:")
    print(benchmark_df.to_string(index=False))

    return results, benchmark_df, models


def plot_visualizations(results, models, vectorizer):
    """Step 5: Visualizations - Confusion matrices, ROC curves, and top keyword indicators."""
    print("\n" + "=" * 75)
    print("STEP 5: PLOTTING VISUALIZATIONS & MOST INFORMATIVE FEATURES")
    print("=" * 75)

    # 1. Confusion Matrix Heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    cm_palettes = ['Blues', 'Greens']

    for ax, (name, res), cmap in zip(axes, results.items(), cm_palettes):
        sns.heatmap(
            res['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap=cmap,
            cbar=True,
            ax=ax,
            xticklabels=['Negative (0)', 'Positive (1)'],
            yticklabels=['Negative (0)', 'Positive (1)'],
            annot_kws={"size": 15, "weight": "bold"}
        )
        ax.set_title(f"{name}\nAccuracy: {res['accuracy']:.2%} | F1: {res['f1']:.4f}", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel('Predicted Sentiment', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Sentiment', fontsize=11, fontweight='bold')

    plt.suptitle("Sentiment Classification Confusion Matrix Heatmaps", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrices -> {cm_path}")

    # 2. ROC-AUC Curves
    plt.figure(figsize=(8, 6))
    colors = ['#1976D2', '#388E3C']

    for (name, res), color in zip(results.items(), colors):
        plt.plot(res['fpr'], res['tpr'], color=color, lw=2.5,
                 label=f"{name} (AUC = {res['roc_auc']:.4f})")

    plt.plot([0, 1], [0, 1], color='#888888', linestyle='--', lw=1.5, label='Random Guess (AUC = 0.50)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    plt.title('Sentiment Classification ROC-AUC Curves', fontsize=13, fontweight='bold', pad=12)
    plt.legend(loc='lower right', frameon=True, fontsize=10)
    plt.tight_layout()
    roc_path = os.path.join(OUTPUT_DIR, "roc_auc_curve.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC curves -> {roc_path}")

    # 3. Top Most Informative Features (Logistic Regression Coefficients)
    lr_model = models["Logistic Regression"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    coef = lr_model.coef_[0]

    top_pos_idx = np.argsort(coef)[-15:]
    top_neg_idx = np.argsort(coef)[:15]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top Negative Words
    sns.barplot(x=-coef[top_neg_idx], y=feature_names[top_neg_idx], ax=axes[0], color='#D32F2F', alpha=0.85)
    axes[0].set_title("Top 15 Negative Sentiment Keywords (Stemmed)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Negative Weight Magnitude", fontsize=11, fontweight='bold')

    # Top Positive Words
    sns.barplot(x=coef[top_pos_idx][::-1], y=feature_names[top_pos_idx][::-1], ax=axes[1], color='#388E3C', alpha=0.85)
    axes[1].set_title("Top 15 Positive Sentiment Keywords (Stemmed)", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Positive Weight Magnitude", fontsize=11, fontweight='bold')

    plt.suptitle("Most Informative TF-IDF Keywords for Review Sentiment", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    feat_path = os.path.join(OUTPUT_DIR, "top_sentiment_keywords.png")
    plt.savefig(feat_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved sentiment keywords -> {feat_path}")

    # 4. Benchmark Bar Chart
    plt.figure(figsize=(9, 4.5))
    metrics_summary = pd.DataFrame([
        {'Model': 'Naive Bayes', 'Accuracy': results['Multinomial Naive Bayes']['accuracy'], 'F1-Score': results['Multinomial Naive Bayes']['f1'], 'ROC-AUC': results['Multinomial Naive Bayes']['roc_auc']},
        {'Model': 'Logistic Regression', 'Accuracy': results['Logistic Regression']['accuracy'], 'F1-Score': results['Logistic Regression']['f1'], 'ROC-AUC': results['Logistic Regression']['roc_auc']}
    ])
    melted = metrics_summary.melt(id_vars=['Model'], var_name='Metric', value_name='Score')
    ax = sns.barplot(data=melted, x='Metric', y='Score', hue='Model', palette=['#1976D2', '#388E3C'])
    plt.title("NLP Sentiment Model Benchmark Performance", fontsize=13, fontweight='bold', pad=12)
    plt.ylim(0.7, 1.0)
    for p in ax.patches:
        h = p.get_height()
        if not np.isnan(h) and h > 0:
            ax.annotate(f"{h:.3f}", (p.get_x() + p.get_width() / 2., h),
                        ha='center', va='bottom', fontsize=8.5, xytext=(0, 3), textcoords='offset points')
    plt.tight_layout()
    bench_path = os.path.join(OUTPUT_DIR, "model_benchmark_comparison.png")
    plt.savefig(bench_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved benchmark comparison chart -> {bench_path}")


def create_inference_helper(model, vectorizer):
    """Step 6: Live predict_sentiment() function testing custom user phrases."""
    print("\n" + "=" * 75)
    print("STEP 6: INTERACTIVE INFERENCE HELPER FUNCTION (predict_sentiment)")
    print("=" * 75)

    def predict_sentiment(text):
        cleaned = clean_text(text)
        tfidf_vec = vectorizer.transform([cleaned])
        prob = model.predict_proba(tfidf_vec)[0]
        pred_label = model.predict(tfidf_vec)[0]
        sentiment = "Positive 😊" if pred_label == 1 else "Negative 😞"
        confidence = prob[pred_label] * 100
        return {
            'input_text': text,
            'cleaned_text': cleaned,
            'sentiment': sentiment,
            'confidence': f"{confidence:.2f}%",
            'prob_positive': f"{prob[1]:.4f}",
            'prob_negative': f"{prob[0]:.4f}"
        }

    test_phrases = [
        "This product is an absolute masterpiece! Outstanding quality, super fast delivery, highly recommend!",
        "Extremely bad experience. Broke on the second day, horrible customer support, total waste of money.",
        "Average purchase. Some features work nicely while others are clunky, but reasonable for the price.",
        "Terrible and frustrating. Avoid at all costs!",
        "Loved everything about this, brilliant craftsmanship and wonderful packaging."
    ]

    print("\n--- Live Test Predictions on Custom User Phrases ---")
    for phrase in test_phrases:
        res = predict_sentiment(phrase)
        print(f"\n[Phrase]:     \"{res['input_text']}\"")
        print(f"  -> Prediction:  {res['sentiment']} (Confidence: {res['confidence']})")
        print(f"  -> Probabilities: P(Positive)={res['prob_positive']}, P(Negative)={res['prob_negative']}")

    return predict_sentiment


def main():
    print("=" * 80)
    print("TASK 4: NLP SENTIMENT CLASSIFICATION ON CUSTOMER REVIEWS (TF-IDF)")
    print("=" * 80)
    df = load_and_preprocess_data(sample_size=12000)
    X_train, X_test, y_train, y_test, vectorizer = vectorize_and_split(df)
    results, benchmark_df, models = train_and_benchmark(X_train, X_test, y_train, y_test, vectorizer)
    plot_visualizations(results, models, vectorizer)
    predict_fn = create_inference_helper(models["Logistic Regression"], vectorizer)
    print("\n" + "=" * 80)
    print("TASK 4 COMPLETED SUCCESSFULLY!")
    print(f"All figures saved to: {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
