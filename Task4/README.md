# Task 4: NLP Sentiment Classification on Customer Reviews (TF-IDF)

Preprocess raw textual corpus, extract TF-IDF embeddings, and classify review polarity using **Multinomial Naive Bayes** and **Logistic Regression**. Includes an interactive `predict_sentiment()` live testing helper.

---

## 📋 Project Summary & Deliverables

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **Primary Deliverable** | [`sentiment_analysis_pipeline.ipynb`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task4/sentiment_analysis_pipeline.ipynb) | Complete Jupyter Notebook with step-by-step documentation, executed cells, metrics, and embedded visual plots. |
| **Executable Script** | [`pipeline.py`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task4/pipeline.py) | Standalone Python pipeline script that handles NLP preprocessing, TF-IDF vectorization, model training, and live inference testing. |
| **Dataset** | [`data/customer_reviews.csv`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task4/data/customer_reviews.csv) | Real-world customer review corpus (50,000 samples). |
| **Visual Outputs** | [`outputs/`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task4/outputs/) | High-resolution PNG figures (confusion matrices, ROC-AUC curves, top positive/negative keywords). |

---

## 🛠️ Implemented Pipeline Architecture

1. **Text Preprocessing & Cleaning**:
   - Stripped HTML tags (`<br />`) and URLs using regular expressions.
   - Filtered non-alphabetic characters and converted text to lowercase.
   - Removed English stopwords using **NLTK**.
   - Reduced tokens to morphological roots using the **Porter Stemmer**.
2. **Feature Extraction (TF-IDF)**:
   - Extracted unigrams and bigrams (`ngram_range=(1, 2)`) using `TfidfVectorizer(max_features=5000)`.
   - Scaled inverse document frequency with sublinear scaling.
3. **Model Training & Benchmarking**:
   - **Multinomial Naive Bayes (`MultinomialNB`)**: Laplace smoothed ($\alpha=1.0$) generative model.
   - **Logistic Regression**: Regularized discriminative linear model.
4. **Visualizations**:
   - Plotted **Confusion Matrix Heatmaps** for both models side-by-side.
   - Plotted **ROC-AUC Curves** demonstrating classification discrimination.
   - Extracted **Top 15 Most Informative Positive and Negative Keywords**.
5. **Interactive Live Inference**:
   - Built deployable `predict_sentiment(text)` function that accepts custom user phrases, cleans them, computes class probabilities, and outputs predicted sentiment.

---

## 📊 Benchmark Results (Unseen Test Set)

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | **88.67%** | **88.10%** | **89.42%** | **0.8875** | **0.9515** |
| **Multinomial Naive Bayes** | **85.00%** | **84.37%** | **85.92%** | **0.8514** | **0.9363** |

### 🔍 Live Test Inference Sample
- *"This product is an absolute masterpiece! Outstanding quality, super fast delivery, highly recommend!"* $\rightarrow$ **Positive 😊 (87.42% confidence)**
- *"Extremely bad experience. Broke on the second day, horrible customer support, total waste of money."* $\rightarrow$ **Negative 😞 (96.27% confidence)**
- *"Terrible and frustrating. Avoid at all costs!"* $\rightarrow$ **Negative 😞 (95.20% confidence)**

---

## 🚀 How to Run

1. **Jupyter Notebook**:
   Open [`sentiment_analysis_pipeline.ipynb`](file:///c:/Users/Nagendra%20M/OneDrive/Desktop/Datheon/Task4/sentiment_analysis_pipeline.ipynb) in VS Code or Jupyter.
2. **Standalone Script**:
   ```bash
   python pipeline.py
   ```
