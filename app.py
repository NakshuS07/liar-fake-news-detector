import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk import pos_tag, word_tokenize
from collections import Counter
import textstat
from textblob import TextBlob

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

stop_words = set(stopwords.words('english'))

@st.cache_resource
def load_model():
    return joblib.load('fake_news_model.pkl')

model = load_model()

# ---------- Helper functions ----------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_pos_distribution(text):
    default = {
        'pos_noun_ratio': 0, 'pos_verb_ratio': 0, 'pos_adj_ratio': 0,
        'pos_adv_ratio': 0, 'pos_pron_ratio': 0, 'pos_propn_ratio': 0,
        'pos_num_ratio': 0, 'pos_other_ratio': 0
    }
    if not text or len(text.split()) < 3:
        return default, []
    try:
        tokens = word_tokenize(text)
        tags = pos_tag(tokens)
    except Exception:
        return default, []
    total = len(tags) if tags else 1
    counts = Counter(tag for _, tag in tags)

    def ratio(prefixes):
        return sum(v for k, v in counts.items() if any(k.startswith(p) for p in prefixes)) / total

    dist = {
        'pos_noun_ratio':  ratio(['NN']),
        'pos_verb_ratio':  ratio(['VB']),
        'pos_adj_ratio':   ratio(['JJ']),
        'pos_adv_ratio':   ratio(['RB']),
        'pos_pron_ratio':  ratio(['PRP']),
        'pos_propn_ratio': ratio(['NNP']),
        'pos_num_ratio':   ratio(['CD']),
    }
    dist['pos_other_ratio'] = max(0.0, 1 - sum(dist.values()))
    return dist, tags

def extract_features(text):
    if not text or len(text.split()) < 3:
        base = {k: 0 for k in [
            'exclamation_count','question_count','all_caps_ratio',
            'punctuation_ratio','readability_flesch','sentiment_polarity',
            'sentiment_subjectivity','avg_word_length','unique_word_ratio',
            'stopword_ratio']}
        pos_dist, _ = get_pos_distribution(text)
        base.update(pos_dist)
        return base

    words = text.split()
    total_chars = len(text)
    exclamation = text.count('!')
    question    = text.count('?')
    caps_words  = sum(1 for w in words if w.isupper() and len(w) > 2)
    all_caps_ratio = caps_words / max(len(words), 1)
    punct_chars = sum(1 for c in text if c in string.punctuation)
    punct_ratio = punct_chars / max(total_chars, 1)

    try:
        flesch = textstat.flesch_reading_ease(text)
    except Exception:
        flesch = 0

    blob = TextBlob(text)
    polarity     = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    avg_word_len = np.mean([len(w) for w in words]) if words else 0
    unique_ratio = len(set(words)) / max(len(words), 1)
    stop_ratio   = sum(1 for w in words if w in stop_words) / max(len(words), 1)

    features = {
        'exclamation_count': exclamation,
        'question_count': question,
        'all_caps_ratio': all_caps_ratio,
        'punctuation_ratio': punct_ratio,
        'readability_flesch': flesch,
        'sentiment_polarity': polarity,
        'sentiment_subjectivity': subjectivity,
        'avg_word_length': avg_word_len,
        'unique_word_ratio': unique_ratio,
        'stopword_ratio': stop_ratio
    }
    pos_dist, _ = get_pos_distribution(text)
    features.update(pos_dist)
    return features

def get_problem_flags(f):
    flags = []
    if f['exclamation_count'] > 3:
        flags.append(f"Excessive exclamation marks ({f['exclamation_count']})")
    if f['question_count'] > 2:
        flags.append(f"High rhetorical questions ({f['question_count']})")
    if f['all_caps_ratio'] > 0.15:
        flags.append(f"High ALL-CAPS ratio ({f['all_caps_ratio']:.1%})")
    if f['readability_flesch'] < 30:
        flags.append(f"Low readability (Flesch: {f['readability_flesch']:.0f})")
    if f['sentiment_polarity'] < -0.3:
        flags.append(f"Strong negative sentiment ({f['sentiment_polarity']:.2f})")
    if f['sentiment_subjectivity'] > 0.6:
        flags.append(f"Highly subjective ({f['sentiment_subjectivity']:.2f})")
    if f['unique_word_ratio'] < 0.6:
        flags.append(f"Low lexical diversity ({f['unique_word_ratio']:.2f})")
    if f['pos_pron_ratio'] > 0.12:
        flags.append(f"High pronoun usage ({f['pos_pron_ratio']:.1%}) — vague attribution")
    if f['pos_adj_ratio'] > 0.10:
        flags.append(f"High adjective density ({f['pos_adj_ratio']:.1%}) — opinion-heavy")
    if f['pos_adv_ratio'] > 0.08:
        flags.append(f"High adverb density ({f['pos_adv_ratio']:.1%}) — intensifiers/hyperbole")
    return flags

POS_LABELS = {
    'pos_noun_ratio':  'Nouns (NN*)',
    'pos_verb_ratio':  'Verbs (VB*)',
    'pos_adj_ratio':   'Adjectives (JJ*)',
    'pos_adv_ratio':   'Adverbs (RB*)',
    'pos_pron_ratio':  'Pronouns (PRP*)',
    'pos_propn_ratio': 'Proper Nouns (NNP*)',
    'pos_num_ratio':   'Numbers (CD)',
    'pos_other_ratio': 'Other'
}

# ---------- UI ----------
st.set_page_config(page_title="LIAR Fake News Detector", page_icon="📰", layout="wide")
st.title("📰 LIAR Fake News Detector")
st.markdown("Paste a political claim below. The system analyses **linguistic consistency**, "
            "**POS patterns**, and (optionally) **speaker metadata**.")

with st.sidebar:
    st.header("About")
    st.info("Trained on 12,836 PolitiFact statements (LIAR dataset). "
            "Binary classification: fake vs. real. "
            "Realistic accuracy on LIAR is ~0.70–0.85.")

user_title = st.text_input("Statement Title (optional)", placeholder="Enter headline...")
user_text  = st.text_area("Statement / Claim", height=200,
                          placeholder="Paste the claim or statement here...")

# ---------- Optional speaker/context metadata (defaults are neutral) ----------
with st.expander("Optional: Speaker & context metadata (improves accuracy)"):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        party = st.selectbox("Speaker party",
                             ['unknown', 'republican', 'democrat', 'independent', 'none', 'libertarian'])
    with col_b:
        job_title = st.selectbox("Speaker job title",
                                 ['unknown', 'president', 'senator', 'representative',
                                  'governor', 'candidate', 'pundit', 'talk show host',
                                  'radio host', 'columnist', 'activist'])
    with col_c:
        subject_primary = st.selectbox("Primary subject",
                                       ['unknown', 'economy', 'healthcare', 'immigration',
                                        'taxes', 'education', 'jobs', 'elections',
                                        'terrorism', 'gun control', 'abortion'])
    context_choice = st.selectbox("Context",
                                  ['unknown', 'speech', 'debate', 'rally', 'interview',
                                   'ad', 'tweet', 'news release'])
    speaker_reliability = st.slider("Speaker reliability score (-1 = unreliable, +1 = reliable)",
                                    min_value=-1.0, max_value=1.0, value=0.0, step=0.05)

if st.button("🔍 Analyze", type="primary"):
    if not user_text or len(user_text.split()) < 5:
        st.warning("Please enter at least 5 words of statement text.")
    else:
        clean_title = clean_text(user_title)
        clean_body  = clean_text(user_text)
        full_text   = (clean_title + ' ' + clean_body).strip()

        feats = extract_features(user_text)

        # ---- Fill in the metadata columns the model now expects ----
        feats.update({
            # Speaker credit history — defaults to zeros when unknown
            'count_barely_true': 0,
            'count_false': 0,
            'count_half_true': 0,
            'count_mostly_true': 0,
            'count_pants_fire': 0,
            'truth_ratio': max(0.0, speaker_reliability),
            'fake_ratio': max(0.0, -speaker_reliability),
            'speaker_reliability': speaker_reliability,

            # Context flags
            'is_debate':       int(context_choice == 'debate'),
            'is_rally':        int(context_choice == 'rally'),
            'is_interview':    int(context_choice == 'interview'),
            'is_ad':           int(context_choice == 'ad'),
            'is_tweet':        int(context_choice == 'tweet'),
            'is_news_release': int(context_choice == 'news release'),
            'is_speech':       int(context_choice == 'speech'),

            # Categorical metadata
            'party':           party,
            'job_title':       job_title,
            'subject_primary': subject_primary,
        })

        input_df = pd.DataFrame([{**{'full_text': full_text}, **feats}])

        proba = model.predict_proba(input_df)[0]
        pred  = model.predict(input_df)[0]
        confidence = proba[pred] * 100

        col1, col2 = st.columns([1, 2])

        with col1:
            if pred == 1:
                st.error(f"### 🚨 Likely FAKE\n**Confidence: {confidence:.1f}%**")
            else:
                st.success(f"### ✅ Likely REAL\n**Confidence: {confidence:.1f}%**")

            st.markdown("#### Probability Breakdown")
            st.write(f"Real: {proba[0]*100:.1f}%")
            st.progress(float(proba[0]))
            st.write(f"Fake: {proba[1]*100:.1f}%")
            st.progress(float(proba[1]))

        with col2:
            st.markdown("#### 🚩 Problematic Linguistic Areas")
            flags = get_problem_flags(feats)
            if flags:
                for flag in flags:
                    st.warning(f"• {flag}")
            else:
                st.success("No major linguistic red flags detected.")

            st.markdown("#### 🧩 POS Tag Distribution")
            pos_data = {POS_LABELS[k]: round(feats[k] * 100, 2) for k in POS_LABELS}
            pos_df = pd.DataFrame(list(pos_data.items()),
                                  columns=["POS Category", "Percentage (%)"])
            st.dataframe(pos_df, use_container_width=True, hide_index=True)
            st.bar_chart(pos_df.set_index("POS Category"))

            with st.expander("🔠 Sample POS-Tagged Tokens (first 25)"):
                _, tags = get_pos_distribution(user_text)
                if tags:
                    st.dataframe(pd.DataFrame(tags[:25], columns=["Token", "POS Tag"]),
                                 use_container_width=True, hide_index=True)
                else:
                    st.info("Not enough text to extract POS tags.")

st.markdown("---")
st.caption("⚠️ Student capstone project using the LIAR benchmark. "
           "Always verify claims through multiple trusted sources.")
