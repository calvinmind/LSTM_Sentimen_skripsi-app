import asyncio
import sys

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
import re
import json
import time
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import altair as alt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer



from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit.components.v1 as components

from playwright.sync_api import sync_playwright



# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="Analisis Sentimen Reply X", layout="wide")
st.title("Analisis Sentimen Reply Tweet (X)")
st.caption("Input link tweet (X) → website akan mengambil reply dari tweet tersebut → tekan tombol analisis")

# =========================
# PATHS / PARAMS (HARDCODE)
# =========================
ARTIFACT_TOKENIZER_PATH = "tokenizer.pkl"
ARTIFACT_EMBEDDING_PATH = "embedding_matrix.npy"
ARTIFACT_WEIGHTS_PATH = "lstm_weights.h5"
ARTIFACT_LABEL_ENCODER_PATH = "config.json"

COOKIES_JSON_PATH = "cookies.json"   # <- cookies kamu (file)
USE_COOKIES = True                  # <- kalau mau nonaktifkan, ubah jadi False

REPLY_LIMIT = 600                   # <- target 400-600
MAX_LEN = 100                       # <- samakan dengan training


# =========================
# PREPROCESSING
# =========================
def cleaningText(text):
    text = str(text).lower()
    text = re.sub(r'@\w+', ' ', text)                 # mention
    text = re.sub(r'\brt\b', ' ', text)               # RT
    text = re.sub(r'(http\S+|www\.\S+)', ' ', text)   # URL
    text = text.replace('#', ' ')                     # hashtag symbol
    text = re.sub(r'\d+', ' ', text)                  # angka
    text = re.sub(r'[^a-z0-9\s]', ' ', text)          # non-alnum
    text = re.sub(r'\s+', ' ', text).strip()          # whitespace
    return text


def tokenizingText(text):
    return word_tokenize(text)


def filteringText(tokens):
    stopwords_id = set(stopwords.words('indonesian'))
    stopwords_en = set(stopwords.words('english'))

    # pertahankan negasi
    negasi = {"tidak", "bukan", "jangan", "kurang"}
    stopwords_id = stopwords_id - negasi

    extra_stopwords = {"iya","yaa","nya","na","sih","ku","loh","kah","woi","woii","woy","ya"}
    STOPWORDS = stopwords_id | stopwords_en | extra_stopwords

    return [t for t in tokens if t not in STOPWORDS]


def toSentence(list_words):
    return ' '.join(list_words)


slangwords = {
    "abis": "habis",
    "wtb": "beli",
    "wts": "jual",
    "wtt": "tukar",
    "masi": "masih",
    "bgt": "banget",
    "maks": "maksimal",
    "gk": "tidak", "ga": "tidak", "gak": "tidak", "gaa": "tidak",
    "tdk": "tidak", "nggak": "tidak",
    "yg": "yang", "dgn": "dengan", "krn": "karena",
    "dr": "dari", "utk": "untuk"
}


def fix_slangwords(text):
    words = text.split()
    fixed = [slangwords.get(w.lower(), w) for w in words]
    return " ".join(fixed)


def preprocess_text(text):
    try:
        text = cleaningText(text)
        text = fix_slangwords(text)
        tokens = tokenizingText(text)
        tokens = filteringText(tokens)
        result = toSentence(tokens)

        if not result or result.strip() == "":
            return None

        return result
    except Exception:
        return None



def infer_labels(texts, model, tokenizer, label_encoder, max_len: int):
    processed = []
    valid_index = []

    for i, t in enumerate(texts):
        clean = preprocess_text(t)
        if clean is not None:
            processed.append(clean)
            valid_index.append(i)

    if len(processed) == 0:
        raise ValueError("Semua data kosong setelah preprocessing.")

    seqs = tokenizer.texts_to_sequences(processed)
    X = pad_sequences(seqs, maxlen=max_len, padding="post", truncating="post")

    probs = model.predict(X, verbose=0)
    pred_ids = np.argmax(probs, axis=1)
    labels = label_encoder.inverse_transform(pred_ids)

    return labels, probs, valid_index


# =========================
# HELPERS
# =========================

def ensure_nltk():
    nltk.download("punkt")
    nltk.download("stopwords")

ensure_nltk()

def extract_tweet_id(url: str):
    if not url:
        return None
    m = re.search(r"/status/(\d+)", url)
    return m.group(1) if m else None


def render_tweet_embed(url: str):
    html = f"""
    <blockquote class="twitter-tweet">
      <a href="{url}"></a>
    </blockquote>
    <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>
    """
    components.html(html, height=520, scrolling=True)


def load_cookies(driver, cookies_path: str, base_url="https://x.com/"):
    driver.get(base_url)
    time.sleep(2)

    with open(cookies_path, "r", encoding="utf-8") as f:
        cookies = json.load(f)

    for c in cookies:
        c.pop("sameSite", None)
        if "expiry" in c and isinstance(c["expiry"], float):
            c["expiry"] = int(c["expiry"])
        try:
            driver.add_cookie(c)
        except Exception:
            pass

    driver.get(base_url)
    time.sleep(2)



def scrape_replies_playwright(tweet_url: str):
    tweet_id = extract_tweet_id(tweet_url)
    if not tweet_id:
        raise ValueError("Tweet URL tidak valid.")

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-gpu"
            ]
        )

        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )

        # load cookies
        if USE_COOKIES:
            with open(COOKIES_JSON_PATH, "r", encoding="utf-8") as f:
                cookies = json.load(f)
            for c in cookies:
                c.pop("sameSite", None)
            context.add_cookies(cookies)

        page = context.new_page()
        page.goto(tweet_url, timeout=60000)
        page.wait_for_selector("article", timeout=20000)

        # tweet utama
        main_tweet = ""
        try:
            blocks = page.locator(
                f"article a[href*='/status/{tweet_id}'] >> xpath=ancestor::article"
            ).locator("div[data-testid='tweetText']")
            main_tweet = " ".join(blocks.all_inner_texts()).strip()
        except Exception:
            pass

        replies = []
        seen = set()
        
        def collect_replies():
            articles = page.locator("article").all()[:50]
            for art in articles:

                # skip tweet utama
                if art.locator(f"a[href*='/status/{tweet_id}']").count() > 0:
                    continue

                blocks = art.locator("div[data-testid='tweetText']")
                if blocks.count() == 0:
                    continue

                txt = " ".join(blocks.all_inner_texts()).strip()
                if not txt:
                    continue

                # dedup by status id
                sid = None
                links = art.locator("a[href*='/status/']")
                for j in range(links.count()):
                    href = links.nth(j).get_attribute("href") or ""
                    m = re.search(r"/status/(\d+)", href)
                    if m:
                        sid = m.group(1)
                        break

                key = f"id:{sid}" if sid else f"tx:{txt}"
                if key in seen:
                    continue

                seen.add(key)
                replies.append(txt)

        stagnant = 0
        last_count = 0

        page.wait_for_load_state("networkidle", timeout=30000)

        while len(replies) < REPLY_LIMIT and stagnant < 25:
            page.mouse.wheel(0, 2000)
            page.wait_for_timeout(3000)

            collect_replies()
            # klik expand replies
            for label in ["Show", "Lihat", "Tampilkan", "Balasan", "replies"]:
                btns = page.locator(f"button:has-text('{label}')")
                for i in range(min(btns.count(), 8)):
                    try:
                        btns.nth(i).click(timeout=500)
                        page.wait_for_timeout(200)
                    except Exception:
                        pass

            if len(replies) <= last_count:
                stagnant += 1
            else:
                stagnant = 0
                last_count = len(replies)

        browser.close()
        return main_tweet, replies[:REPLY_LIMIT]



TFIDF_STOPWORDS = {
    "tidak", "bukan", "boleh", "bolehnya", "saja", "aja",
    "yang", "dan", "atau", "ini", "itu", "ada", "jadi",
    "kalo", "kalau", "karena", "untuk", "dengan",
    "nih", "dong", "lah", "kok", "deh",
    "nya", "sih"
}


def top_words_tfidf(texts, top_n=5):
    # preprocessing semua reply
    cleaned_texts = [preprocess_text(t) for t in texts]

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 1)  # unigram (sesuai skripsi LSTM)
    )

    tfidf_matrix = vectorizer.fit_transform(cleaned_texts)

    # rata-rata skor TF-IDF tiap kata
    tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)

    # mapping kata → skor
    feature_names = vectorizer.get_feature_names_out()
    tfidf_dict = {
        word: score
        for word, score in zip(feature_names, tfidf_scores)
        if word not in TFIDF_STOPWORDS and len(word) > 2
    }

    # ambil top-N
    top_words = sorted(
        tfidf_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return top_words


# =========================
# LOAD ARTIFACTS
# =========================
@st.cache_resource
def build_model(vocab_size, embedding_matrix):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(100,), name="input_ids"),
            tf.keras.layers.Embedding(
                vocab_size,
                300,
                weights=[embedding_matrix],
                trainable=False,
                mask_zero=True
            ),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(3, activation="softmax")
        ])
        return model

def load_artifacts():

    with open(ARTIFACT_TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    
    embedding_matrix = np.load(ARTIFACT_EMBEDDING_PATH)
    vocab_size = embedding_matrix.shape[0]
    
    model = build_model(vocab_size, embedding_matrix)
    model.load_weights(ARTIFACT_WEIGHTS_PATH)

    with open(ARTIFACT_LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    class SimpleLabelDecoder:
        def __init__(self, labels):
            self.id2label = {i: label for i, label in enumerate(labels)}
        def inverse_transform(self, ids):
            return np.array([self.id2label[int(i)] for i in ids])

    label_encoder = SimpleLabelDecoder(config["labels"])
    return model, tokenizer, label_encoder


try:
    model, tokenizer, label_encoder = load_artifacts()
except Exception as e:
    st.error("Gagal load model/tokenizer/label_encoder. Cek path artifacts.")
    st.exception(e)
    st.stop()


# =========================
# UI
# =========================
tweet_url = st.text_input("🔗 Link Tweet (X)", placeholder="https://x.com/username/status/1234567890")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Tweet")
    main_tweet_box = st.empty()
    main_tweet_box.write(st.session_state.get("main_tweet", "Belum diambil. Klik tombol analisis dulu."))

with col2:
    st.subheader("Analisis Reply")
    st.caption(f"Limit reply : {REPLY_LIMIT} | MAX_LEN : {MAX_LEN}")
    run_btn = st.button("Analisis Sentimen Tweet", use_container_width=True)

if run_btn:
    if not tweet_url or not extract_tweet_id(tweet_url):
        st.warning("Link tweet tidak valid. Pastikan formatnya ada /status/<id>.")
        st.stop()

    with st.spinner("Scraping replies via Playwright..."):
        try:
            main_tweet, replies = scrape_replies_playwright(tweet_url=tweet_url)
            st.session_state["main_tweet"] = main_tweet
            main_tweet_box.write(main_tweet if main_tweet else "(Teks tweet utama tidak terbaca)")

        except Exception as e:
            st.error("Gagal scraping reply, pastikan cookies valid.")
            st.exception(e)
            st.stop()

    if not replies:
        st.warning("Reply tidak ditemukan / tidak berhasil ter-load. Coba cookies.json.")
        st.stop()


    with st.spinner("Memprediksi sentimen reply..."):
        labels, probs, valid_index = infer_labels(replies, model, tokenizer, label_encoder, MAX_LEN)
        filtered_replies = [replies[i] for i in valid_index]

    df = pd.DataFrame({"reply": filtered_replies, "sentiment": labels})
    counts = df["sentiment"].value_counts()
    perc = (counts / len(df) * 100).round(2)
    top5 = top_words_tfidf(filtered_replies, top_n=5)
    st.session_state["last_df"] = df
    st.session_state["last_counts"] = counts
    st.session_state["last_perc"] = perc
    st.session_state["last_top_df"] = pd.DataFrame(top5, columns=["Kata", "Frekuensi"])

if "last_df" in st.session_state:
    df = st.session_state["last_df"]
    counts = st.session_state["last_counts"]
    perc = st.session_state["last_perc"]
    top_df = st.session_state["last_top_df"]

    st.success(f"Selesai. Total reply dianalisis: {len(df)}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Persentase Sentimen Reply")
        st.dataframe(pd.DataFrame({"count": counts, "percent": perc}).sort_index(), use_container_width=True)
        chart_df = pd.DataFrame({
            "sentiment": perc.index,
            "percent": perc.values
        })

        # horizontal bar chart
        chart = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X("percent:Q", title="Persentase (%)"),
            y=alt.Y("sentiment:N", sort="-x", title="Sentimen"),
            tooltip=["sentiment", "percent"]
        ).properties(
            height=250
        )

        st.altair_chart(chart, use_container_width=True)

    with right:
        st.subheader("Beberapa Contoh Reply + Label")
        st.dataframe(df.head(30), use_container_width=True)

    st.download_button(
        "Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="hasil_sentimen_reply.csv",
        mime="text/csv",
        use_container_width=True,
        key="download_csv_result"
    )

    st.subheader("5 Kata Paling Sering Muncul")
    st.dataframe(top_df, use_container_width=True)
