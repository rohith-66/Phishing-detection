# train_llm_model.py
import os
import re
import json
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from typing import Dict, List
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

DATA_CSV = "final_dataset_with_features.csv"
SENTENCE_MODEL_NAME = "all-MiniLM-L6-v2"
SENTENCE_MODEL_DIR = "./sentence_transformer_model"
CLASSIFIER_PATH = "phishing_detector_model.joblib"
TOP_TOKENS_PATH = "top_tokens.json"
METADATA_PATH = "model_metadata.json"
RANDOM_STATE = 42
TOP_K_TOKENS = 300  # how many most discriminative tokens to keep

# ---------------------------
# Tokenization & Token Scoring
# ---------------------------
def tokenize_url(url: str) -> List[str]:
    # split on non-alphanumeric; keep tokens >1 char
    tokens = re.split(r"[^A-Za-z0-9]+", url.lower())
    return [t for t in tokens if len(t) > 1]

def learn_top_tokens(df_train: pd.DataFrame, url_col="url", label_col="label", top_k=300) -> Dict[str, float]:
    pos_counts = Counter()
    neg_counts = Counter()
    for url, lbl in zip(df_train[url_col], df_train[label_col]):
        toks = set(tokenize_url(url))  # set to reduce repeats within a single URL
        if lbl == 1:
            pos_counts.update(toks)
        else:
            neg_counts.update(toks)

    vocab = set(pos_counts) | set(neg_counts)
    total_pos = sum(pos_counts.values())
    total_neg = sum(neg_counts.values())
    alpha = 1.0  # Laplace smoothing

    token_scores = {}
    V = len(vocab) if len(vocab) > 0 else 1
    for tok in vocab:
        p_pos = (pos_counts.get(tok, 0) + alpha) / (total_pos + alpha * V)
        p_neg = (neg_counts.get(tok, 0) + alpha) / (total_neg + alpha * V)
        token_scores[tok] = float(np.log(p_pos / p_neg))

    # keep most discriminative by |log-odds|
    sorted_tokens = sorted(token_scores.items(), key=lambda kv: -abs(kv[1]))
    return dict(sorted_tokens[:top_k])

# ---------------------------
# Handcrafted URL Features
# ---------------------------
FEATURE_NAMES = [
    "url_length",
    "count_digits",
    "count_dots",
    "count_hyphens",
    "count_at",
    "count_slashes",
    "digit_ratio",
    "char_entropy",
    "path_depth",
    "ip_in_domain",
    "suspicious_token_score",
]

def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    probs = np.array([c / len(s) for c in counts.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-12)))

def looks_like_ip(host: str) -> int:
    # crude IPv4 check: 4 dot-separated integers 0..255
    parts = host.split(".")
    if len(parts) != 4:
        return 0
    try:
        return int(all(0 <= int(p) <= 255 for p in parts))
    except ValueError:
        return 0

def extract_url_features(url: str, top_tokens: Dict[str, float]) -> List[float]:
    u = url.strip()
    low = u.lower()

    # basic counts
    url_length = len(u)
    count_digits = sum(ch.isdigit() for ch in u)
    count_dots = u.count(".")
    count_hyphens = u.count("-")
    count_at = u.count("@")
    count_slashes = u.count("/")

    digit_ratio = count_digits / max(1, url_length)
    char_entropy = shannon_entropy(u)

    # path depth ("/" segments minus protocol)
    # e.g., https://a.b/c/d -> segments after "://" counted
    path = re.sub(r"^https?://", "", low)
    path_segments = [seg for seg in path.split("/") if seg]
    path_depth = max(0, len(path_segments) - 1)  # subtract domain segment

    # domain/host
    host = path_segments[0] if path_segments else ""
    ip_in_domain = looks_like_ip(host)

    # suspicious token score (learned log-odds)
    toks = set(tokenize_url(low))
    suspicious_token_score = float(sum(top_tokens.get(t, 0.0) for t in toks))

    return [
        float(url_length),
        float(count_digits),
        float(count_dots),
        float(count_hyphens),
        float(count_at),
        float(count_slashes),
        float(digit_ratio),
        float(char_entropy),
        float(path_depth),
        float(ip_in_domain),
        float(suspicious_token_score),
    ]

# ---------------------------
# Main Training Pipeline
# ---------------------------
def main():
    # 1) Load data
    if not os.path.exists(DATA_CSV):
        print(f"Error: '{DATA_CSV}' not found.")
        return
    df = pd.read_csv(DATA_CSV)
    if not {"url", "label"}.issubset(df.columns):
        print("Error: CSV must contain 'url' and 'label' columns.")
        return

    # Optional: balance classes by downsampling majority (keeps pipeline simple)
    df_pos = df[df["label"] == 1]
    df_neg = df[df["label"] == 0]
    if len(df_pos) == 0 or len(df_neg) == 0:
        print("Error: Need both phishing (1) and legitimate (0) samples.")
        return

    n_pos = len(df_pos)
    df_neg_down = df_neg.sample(n=min(len(df_neg), n_pos), random_state=RANDOM_STATE)
    df_bal = pd.concat([df_pos, df_neg_down], ignore_index=True).sample(
        frac=1.0, random_state=RANDOM_STATE
    )
    print(f"Balanced dataset size: {len(df_bal)} (pos={len(df_pos)}, neg_down={len(df_neg_down)})")

    # 2) Train/test split
    X_urls = df_bal["url"].values
    y = df_bal["label"].values
    X_train_urls, X_test_urls, y_train, y_test = train_test_split(
        X_urls, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 3) Learn top tokens from TRAIN ONLY (avoid leakage)
    df_train = pd.DataFrame({"url": X_train_urls, "label": y_train})
    top_tokens = learn_top_tokens(df_train, top_k=TOP_K_TOKENS)
    with open(TOP_TOKENS_PATH, "w") as f:
        json.dump(top_tokens, f)
    print(f"Saved learned tokens -> {TOP_TOKENS_PATH} (K={len(top_tokens)})")

    # 4) Load or download sentence model
    if os.path.exists(SENTENCE_MODEL_DIR):
        print(f"Loading sentence model from {SENTENCE_MODEL_DIR}")
        sbert = SentenceTransformer(SENTENCE_MODEL_DIR)
    else:
        print(f"Downloading '{SENTENCE_MODEL_NAME}' and saving to {SENTENCE_MODEL_DIR}")
        sbert = SentenceTransformer(SENTENCE_MODEL_NAME)
        sbert.save(SENTENCE_MODEL_DIR)

    # 5) Embeddings (train/test) + L2 normalize
    print("Encoding SentenceTransformer embeddings (train)...")
    emb_train = sbert.encode(list(X_train_urls), show_progress_bar=True)
    print("Encoding SentenceTransformer embeddings (test)...")
    emb_test = sbert.encode(list(X_test_urls), show_progress_bar=True)

    emb_train = normalize(emb_train)  # L2 normalization
    emb_test = normalize(emb_test)

    emb_dim = emb_train.shape[1]
    print(f"Embedding dimension: {emb_dim}")

    # 6) Handcrafted features (train/test)
    print("Computing handcrafted features (train/test)...")
    feats_train = np.array([extract_url_features(u, top_tokens) for u in X_train_urls], dtype=float)
    feats_test = np.array([extract_url_features(u, top_tokens) for u in X_test_urls], dtype=float)

    # 7) Concatenate: [embeddings || features]
    X_train = np.hstack([emb_train, feats_train])
    X_test = np.hstack([emb_test, feats_test])

    # 8) Train classifier (balanced LR). You can swap with RandomForest/XGBoost if desired.
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=None,  # set to None for mac compatibility
        solver="lbfgs",
    )
    print("Training classifier...")
    clf.fit(X_train, y_train)

    # 9) Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Evaluation ===")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", cm)
    print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))

    # 10) Save artifacts
    joblib.dump(clf, CLASSIFIER_PATH)
    print(f"Saved classifier -> {CLASSIFIER_PATH}")

    metadata = {
        "embedding_model_dir": SENTENCE_MODEL_DIR,
        "embedding_dim": emb_dim,
        "extra_feature_names": FEATURE_NAMES,
        "top_tokens_path": TOP_TOKENS_PATH,
        "concat_order": ["embeddings_first", "extra_features_second"],
        "random_state": RANDOM_STATE,
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata -> {METADATA_PATH}")
    print("\nTraining complete.")

if __name__ == "__main__":
    main()
