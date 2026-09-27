 LIAR Fake News Detector

A Streamlit web application that analyses political claims for authenticity
using **internal linguistic consistency**, **part-of-speech patterns**,
and **speaker metadata**. Built as an NLP capstone project on the
[LIAR benchmark dataset](https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
(Wang, ACL 2017).

---

 Project Overview

Fake news detection is typically framed as a binary text classification
problem. This project goes a step further by combining **TF-IDF text features**
with **hand-engineered linguistic features** — punctuation style, readability,
sentiment, POS distributions, speaker history, and context — to detect
deceptive patterns that raw bag-of-words models miss.

The application classifies any pasted statement as **fake** or **real** and
returns:

- A verdict with confidence percentage
- A probability breakdown for both classes
- A list of **problematic linguistic areas** (e.g., excessive exclamation
  marks, high subjectivity, low readability, high pronoun usage)
- A **POS tag distribution** table and bar chart
- Sample POS-tagged tokens for the input text

---

 Dataset

**LIAR** (Wang, 2017) — 12,836 short political statements manually fact-checked
by PolitiFact.

| Label (original) | Mapped to |
|---|---|
| `pants-fire`, `false`, `barely-true` | **Fake (1)** |
| `half-true`, `mostly-true`, `true` | **Real (0)** |

Each row contains the statement text, subject, speaker, job title, state,
party affiliation, speaker credit history, and context (venue).

Split sizes:
- Train: 10,269
- Validation: 1,284
- Test: 1,283

---

 Features Engineered

### 1. Text Features
- TF-IDF over word n-grams (1–3), 10,000 features, sublinear TF, `min_df=2`

### 2. Style / Punctuation Features
- `exclamation_count`, `question_count`
- `all_caps_ratio`, `punctuation_ratio`

### 3. Readability Features
- Flesch Reading Ease score (`textstat`)

### 4. Sentiment Features
- Polarity and subjectivity (`textblob`)

### 5. Lexical Diversity
- `avg_word_length`, `unique_word_ratio`, `stopword_ratio`

### 6. POS Features
- Ratios for nouns, verbs, adjectives, adverbs, pronouns, proper nouns,
  numbers, and other tags (NLTK averaged perceptron tagger)

### 7. Speaker History Features
- Counts of `barely_true`, `false`, `half_true`, `mostly_true`, `pants_fire`
- `truth_ratio`, `fake_ratio`, `speaker_reliability`

### 8. Context Features
- Binary flags: `is_debate`, `is_rally`, `is_interview`, `is_ad`,
  `is_tweet`, `is_news_release`, `is_speech`

### 9. Categorical Metadata
- `party`, `job_title`, `subject_primary` (one-hot encoded)

---

 Model

**Pipeline:** TF-IDF + standardised numeric features + one-hot categoricals
→ **Logistic Regression** (`class_weight='balanced'`, `max_iter=2000`).

### Results (test set)

| Metric | Real (0) | Fake (1) | Overall |
|---|---|---|---|
| Precision | 0.73 | 0.68 | 0.71 |
| Recall | 0.76 | 0.65 | 0.71 |
| F1-score | 0.74 | 0.66 | 0.70 |
| **Accuracy** | — | — | **0.709** |

### Baselines for Context

| Method | Accuracy |
|---|---|
| Majority class | 0.56 |
| Text-only TF-IDF + LogReg | 0.65 |
| **This project (all features)** | **0.71** |
| Original Wang (2017) LR + metadata | 0.72 |
| Fine-tuned BERT (upper bound) | 0.82–0.88 |

> The dataset is intentionally hard because statements come from the same
> pool of politicians, so the model cannot rely on source artifacts. Anything
> above 0.90 on LIAR binary classification is almost always data leakage.

---

 Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/liar-fake-news-detector.git
cd liar-fake-news-detector

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Streamlit app
streamlit run app.py