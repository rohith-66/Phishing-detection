# app.py
import logging
import traceback
import json
import re
import os
import numpy as np
from collections import Counter
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import joblib
import torch

# -------------------------
# Configuration
# -------------------------
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

SENTENCE_MODEL_DIR = "./sentence_transformer_model"
CLASSIFIER_PATH = "phishing_detector_model.joblib"
TOP_TOKENS_PATH = "top_tokens.json"
METADATA_PATH = "model_metadata.json"

# -------------------------
# Helper functions
# -------------------------
def tokenize_url(url):
    tokens = re.split(r"[^A-Za-z0-9]+", url.lower())
    return [t for t in tokens if len(t) > 1]

def shannon_entropy(s: str):
    if not s:
        return 0.0
    counts = Counter(s)
    probs = np.array([c / len(s) for c in counts.values()])
    return float(-np.sum(probs * np.log2(probs + 1e-12)))

def looks_like_ip(host: str):
    parts = host.split(".")
    if len(parts) != 4:
        return 0
    try:
        return int(all(0 <= int(p) <= 255 for p in parts))
    except ValueError:
        return 0

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

# -------------------------
# Load models & metadata
# -------------------------
# SentenceTransformer: load on CPU to avoid MPS/forking issues
try:
    logger.info("Loading SentenceTransformer (device=cpu)...")
    sbert = SentenceTransformer(SENTENCE_MODEL_DIR, device="cpu")
    logger.info("SentenceTransformer loaded.")
except Exception:
    logger.exception("Failed to load SentenceTransformer.")
    sbert = None

# top_tokens
try:
    with open(TOP_TOKENS_PATH, "r") as f:
        top_tokens = json.load(f)
    logger.info(f"Loaded top_tokens (count={len(top_tokens)})")
except Exception:
    top_tokens = {}
    logger.warning("top_tokens.json not found or failed to load; suspicious_token_score will be 0")

# metadata (optional)
try:
    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)
    logger.info("Loaded model_metadata.json")
except Exception:
    metadata = {}
    logger.warning("model_metadata.json not found or failed to load")

# classifier
try:
    classifier = joblib.load(CLASSIFIER_PATH)
    logger.info("Classifier loaded.")
except Exception:
    classifier = None
    logger.exception(f"Failed to load classifier from {CLASSIFIER_PATH}")

# Determine expected input dimensionality if available
expected_input_dim = None
try:
    if classifier is not None and hasattr(classifier, "coef_"):
        # logistic/regression: coef_.shape = (n_classes_or_1, n_features)
        expected_input_dim = classifier.coef_.shape[1]
        logger.info(f"Classifier expects input dim = {expected_input_dim}")
except Exception:
    logger.warning("Could not infer expected input dim from classifier metadata.")

# -------------------------
# Routes
# -------------------------
@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "alive"}), 200

@app.route("/predict", methods=["POST"])
def predict_full():
    try:
        logger.debug("=== /predict called ===")
        data = request.get_json(force=True)
        logger.debug(f"Request JSON: {data}")
        url = data.get("url")
        if not url:
            return jsonify({"error": "Missing 'url' in JSON"}), 400

        if sbert is None:
            return jsonify({"error": "Embedding model not loaded"}), 500
        if classifier is None:
            return jsonify({"error": "Classifier model not loaded"}), 500

        # Encode embedding (no_grad)
        try:
            with torch.no_grad():
                emb = sbert.encode([url], show_progress_bar=False)
        except Exception as e:
            logger.exception("Error during sbert.encode()")
            tb = traceback.format_exc()
            return jsonify({"error": "encode error", "traceback": tb}), 500

        # Normalize
        try:
            emb = normalize(emb)
        except Exception as e:
            logger.exception("Error during normalize()")
            tb = traceback.format_exc()
            return jsonify({"error": "normalize error", "traceback": tb}), 500

        logger.debug(f"Embedding shape: {emb.shape}")

        # Extract features
        try:
            feats = np.array([extract_url_features(url, top_tokens)], dtype=float)
        except Exception as e:
            logger.exception("Error during feature extraction")
            tb = traceback.format_exc()
            return jsonify({"error": "feature extraction error", "traceback": tb}), 500

        logger.debug(f"Features shape: {feats.shape}, values: {feats.tolist()}")

        # Concatenate
        X = np.hstack([emb, feats])
        logger.debug(f"Final input shape: {X.shape}")

        # Optionally check expected dim
        if expected_input_dim is not None and X.shape[1] != expected_input_dim:
            logger.error(f"Input dim mismatch: got {X.shape[1]}, expected {expected_input_dim}")
            return jsonify({"error": "input dim mismatch", "got": int(X.shape[1]), "expected": int(expected_input_dim)}), 500

        # Predict
        try:
            pred = classifier.predict(X)[0]
            proba = classifier.predict_proba(X)[0].tolist()
        except Exception as e:
            logger.exception("Error during classifier prediction")
            tb = traceback.format_exc()
            return jsonify({"error": "prediction error", "traceback": tb}), 500

        logger.debug(f"Prediction: {pred}, probabilities: {proba}")

        return jsonify({
            "url": url,
            "is_phishing": int(pred),
            "probabilities": {"legitimate": proba[0], "phishing": proba[1]}
        }), 200

    except Exception as e:
        logger.exception("Unhandled exception in /predict")
        tb = traceback.format_exc()
        return jsonify({"error": str(e), "traceback": tb}), 500

# -------------------------
# Dev run
# -------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
