
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import entropy

from config.path import (
    PROCESSED_DIR,
    DRIFT_DIR,
    DRIFT_BASELINE_PATH,
    DRIFT_LOG_PATH
)

from HR_project_logging.project_logging import get_logger

logger = get_logger(__name__)


# ===============================
# Load Documents
# ===============================
def load_processed_documents() -> Dict[str, str]:
    docs = {}

    for file in PROCESSED_DIR.glob("*.txt"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                docs[file.name] = f.read()
        except Exception as e:
            logger.error(f"Error reading {file}: {e}")

    return docs


# ===============================
# TF-IDF Drift
# ===============================
def compute_tfidf_shift(old_docs: List[str], new_docs: List[str]) -> float:
    vectorizer = TfidfVectorizer(max_features=1000)

    combined = old_docs + new_docs
    tfidf = vectorizer.fit_transform(combined)

    old_vec = tfidf[:len(old_docs)].mean(axis=0)
    new_vec = tfidf[len(old_docs):].mean(axis=0)

    similarity = cosine_similarity(old_vec, new_vec)[0][0]
    return float(1 - similarity)


# ===============================
# Length Drift
# ===============================
def compute_length_drift(old_docs: List[str], new_docs: List[str]) -> float:
    old_len = np.mean([len(d) for d in old_docs])
    new_len = np.mean([len(d) for d in new_docs])

    return abs(new_len - old_len) / (old_len + 1e-5)


# ===============================
# KL Divergence
# ===============================
def compute_kl_divergence(old_docs: List[str], new_docs: List[str]) -> float:
    vectorizer = TfidfVectorizer(max_features=1000)

    old_tfidf = vectorizer.fit_transform(old_docs).toarray()
    new_tfidf = vectorizer.transform(new_docs).toarray()

    old_dist = np.mean(old_tfidf, axis=0) + 1e-10
    new_dist = np.mean(new_tfidf, axis=0) + 1e-10

    old_dist /= old_dist.sum()
    new_dist /= new_dist.sum()

    return float(entropy(old_dist, new_dist))


# ===============================
# Timestamp
# ===============================
def get_latest_timestamp() -> str:
    times = [file.stat().st_mtime for file in PROCESSED_DIR.glob("*.txt")]
    return datetime.fromtimestamp(max(times)).isoformat() if times else None


# ===============================
# Baseline
# ===============================
def load_baseline():
    if DRIFT_BASELINE_PATH.exists():
        with open(DRIFT_BASELINE_PATH, "rb") as f:
            return pickle.load(f)
    return None


def save_baseline(docs: Dict[str, str]):
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    with open(DRIFT_BASELINE_PATH, "wb") as f:
        pickle.dump(docs, f)

    logger.info("Baseline saved/updated")


# ===============================
# Decision Logic
# ===============================
def detect_policy_change(tfidf, length, kl) -> str:
    if tfidf > 0.2 or length > 0.3 or kl > 0.5:
        return "YES"
    return "NO"


# ===============================
# CSV Logger
# ===============================
def save_drift_log(log_data: Dict):
    DRIFT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame([log_data])

    if DRIFT_LOG_PATH.exists():
        df.to_csv(DRIFT_LOG_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(DRIFT_LOG_PATH, index=False)

    logger.info(f"Drift log saved: {DRIFT_LOG_PATH}")


# ===============================
# MAIN FUNCTION
# ===============================
def run_drift_detection():
    logger.info("Running drift detection...")

    current_docs = load_processed_documents()
    baseline_docs = load_baseline()

    if baseline_docs is None:
        logger.info("No baseline found. Creating baseline.")
        save_baseline(current_docs)
        return

    old_docs = list(baseline_docs.values())
    new_docs = list(current_docs.values())

    tfidf_shift = compute_tfidf_shift(old_docs, new_docs)
    length_drift = compute_length_drift(old_docs, new_docs)
    kl_div = compute_kl_divergence(old_docs, new_docs)
    timestamp = get_latest_timestamp()

    policy_changed = detect_policy_change(tfidf_shift, length_drift, kl_div)

    log_data = {
        "timestamp": timestamp,
        "num_files": len(current_docs),
        "tfidf_shift": tfidf_shift,
        "length_drift": length_drift,
        "kl_divergence": kl_div,
        "policy_changed": policy_changed
    }

    save_drift_log(log_data)

    if policy_changed == "YES":
        logger.info("Policy changed → updating baseline")
        save_baseline(current_docs)

    logger.info("Drift detection completed")
