# diagnose_model.py
import json, joblib, numpy as np, re
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from collections import Counter
import os

SENTENCE_MODEL_DIR = "./sentence_transformer_model"
CLASSIFIER_PATH = "phishing_detector_model.joblib"
TOP_TOKENS_PATH = "top_tokens.json"
METADATA_PATH = "model_metadata.json"

def tokenize_url(url):
    return [t for t in re.split(r"[^A-Za-z0-9]+", url.lower()) if len(t) > 1]

def shannon_entropy(s: str):
    if not s: return 0.0
    counts = Counter(s)
    probs = np.array([c/len(s) for c in counts.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-12)))

def looks_like_ip(host: str):
    parts = host.split(".")
    if len(parts) != 4: return 0
    try: return int(all(0 <= int(p) <= 255 for p in parts))
    except: return 0

def extract_url_features(url, top_tokens):
    u = url.strip()
    low = u.lower()
    url_length = len(u)
    count_digits = sum(ch.isdigit() for ch in u)
    count_dots = u.count(".")
    count_hyphens = u.count("-")
    count_at = u.count("@")
    count_slashes = u.count("/")
    digit_ratio = count_digits / max(1, url_length)
    char_entropy = shannon_entropy(u)
    path = re.sub(r"^https?://", "", low)
    path_segments = [seg for seg in path.split("/") if seg]
    path_depth = max(0, len(path_segments) - 1)
    host = path_segments[0] if path_segments else ""
    ip_in_domain = looks_like_ip(host)
    toks = set(tokenize_url(low))
    suspicious_token_score = float(sum(top_tokens.get(t, 0.0) for t in toks))
    return np.array([
        url_length, count_digits, count_dots, count_hyphens,
        count_at, count_slashes, digit_ratio, char_entropy,
        path_depth, ip_in_domain, suspicious_token_score
    ], dtype=float)

# --- Load artifacts
sbert = SentenceTransformer(SENTENCE_MODEL_DIR, device="cpu")
clf = joblib.load(CLASSIFIER_PATH)
top_tokens = {}
if os.path.exists(TOP_TOKENS_PATH):
    top_tokens = json.load(open(TOP_TOKENS_PATH, "r"))
meta = {}
if os.path.exists(METADATA_PATH):
    meta = json.load(open(METADATA_PATH, "r"))

print("Loaded model, top_tokens count:", len(top_tokens))
print("Classifier type:", type(clf))
if hasattr(clf, "coef_"):
    print("Classifier coef shape:", clf.coef_.shape)

def inspect(url):
    emb = sbert.encode([url])
    emb = normalize(emb)
    feats = extract_url_features(url, top_tokens).reshape(1, -1)
    X = np.hstack([emb, feats])
    print("URL:", url)
    print("embedding shape:", emb.shape, "features:", feats.tolist()[0])
    print("final X shape:", X.shape)

    # predict
    proba = clf.predict_proba(X)[0]
    pred = clf.predict(X)[0]
    print("pred:", int(pred), "proba:", proba)

    # if logistic regression, show top positive/negative contributions
    if hasattr(clf, "coef_"):
        coefs = clf.coef_.ravel()  # shape (n_features,)
        intercept = clf.intercept_.ravel()
        # compute per-feature contribution to decision function: coef * x
        contributions = (coefs * X.ravel())
        # show top contributors (abs)
        top_idx = np.argsort(-np.abs(contributions))[:12]
        print("Top contributors (index, contribution, coef, value):")
        for idx in top_idx:
            print(f"  idx={idx:3d} | contrib={contributions[idx]:+.6f} | coef={coefs[idx]:+.6f} | val={X.ravel()[idx]:.6f}")
        # map last 11 features to human names if available
        if "extra_feature_names" in meta:
            names = meta["extra_feature_names"]
            emb_dim = meta.get("embedding_dim", emb.shape[1])
            print("\nFeature name for indexes (last features):")
            for i, name in enumerate(names):
                print(f"  idx={emb_dim + i} -> {name}")
    else:
        print("No coef_ available for classifier type:", type(clf))

# Inspect your problematic URLs
inspect("https://paypal-login-secure-page.com")
print("\n---\n")
inspect("https://www.google.com")
