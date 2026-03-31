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
# LOAD DOCUMENTS
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
# TF-IDF DRIFT
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
# LENGTH DRIFT
# ===============================
def compute_length_drift(old_docs: List[str], new_docs: List[str]) -> float:
    old_len = np.mean([len(d) for d in old_docs])
    new_len = np.mean([len(d) for d in new_docs])

    return abs(new_len - old_len) / (old_len + 1e-5)


# ===============================
# KL DIVERGENCE
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
# TIMESTAMP
# ===============================
def get_latest_timestamp() -> str:
    times = [file.stat().st_mtime for file in PROCESSED_DIR.glob("*.txt")]
    return datetime.fromtimestamp(max(times)).isoformat() if times else None


# ===============================
# BASELINE
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
# GLOBAL DECISION
# ===============================
def detect_policy_change(tfidf, length, kl) -> str:
    if tfidf > 0.2 or length > 0.3 or kl > 0.5:
        return "YES"
    return "NO"


# ===============================
# FILE-LEVEL DRIFT
# ===============================
def detect_file_level_changes(
    baseline_docs: Dict[str, str],
    current_docs: Dict[str, str],
    threshold: float = 0.85
) -> List[Dict]:

    results = []
    vectorizer = TfidfVectorizer(max_features=1000)

    # NEW / MODIFIED / UNCHANGED
    for file_name, new_text in current_docs.items():

        if file_name not in baseline_docs:
            results.append({"file_name": file_name, "status": "NEW"})
            continue

        old_text = baseline_docs[file_name]

        tfidf = vectorizer.fit_transform([old_text, new_text])
        similarity = cosine_similarity(tfidf[0], tfidf[1])[0][0]

        if similarity < threshold:
            status = "MODIFIED"
        else:
            status = "UNCHANGED"

        results.append({"file_name": file_name, "status": status})

    # REMOVED
    for file_name in baseline_docs:
        if file_name not in current_docs:
            results.append({"file_name": file_name, "status": "REMOVED"})

    return results


# ===============================
# CSV LOG (GLOBAL)
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
# CSV LOG (FILE LEVEL)
# ===============================
def save_file_level_log(file_results: List[Dict], timestamp: str):
    file_log_path = DRIFT_DIR / "document_file_level_drift_log.csv"

    rows = []
    for item in file_results:
        rows.append({
            "timestamp": timestamp,
            "file_name": item["file_name"],
            "status": item["status"]
        })

    df = pd.DataFrame(rows)

    if file_log_path.exists():
        df.to_csv(file_log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_log_path, index=False)

    logger.info(f"File-level drift log saved: {file_log_path}")


# ===============================
# MAIN FUNCTION
# ===============================
def run_drift_detection():
    logger.info("Running drift detection...")

    try:
        current_docs = load_processed_documents()
        baseline_docs = load_baseline()

        if not current_docs:
            logger.warning("No processed documents found. Skipping drift detection.")
            return

        # ===============================
        # FIRST RUN
        # ===============================
        if baseline_docs is None:
            logger.info("No baseline found. Creating baseline.")

            save_baseline(current_docs)

            timestamp = get_latest_timestamp()

            log_data = {
                "timestamp": timestamp,
                "num_files": len(current_docs),
                "tfidf_shift": 0.0,
                "length_drift": 0.0,
                "kl_divergence": 0.0,
                "policy_changed": "NO",
                "changed_files": "None"
            }

            save_drift_log(log_data)

            logger.info("Initial baseline created and logged")
            return

        # ===============================
        # GLOBAL DRIFT
        # ===============================
        old_docs = list(baseline_docs.values())
        new_docs = list(current_docs.values())

        tfidf_shift = compute_tfidf_shift(old_docs, new_docs)
        length_drift = compute_length_drift(old_docs, new_docs)
        kl_div = compute_kl_divergence(old_docs, new_docs)
        timestamp = get_latest_timestamp()

        logger.info(
            "Drift Metrics → TF-IDF: %.4f | Length: %.4f | KL: %.4f",
            tfidf_shift,
            length_drift,
            kl_div
        )

        # ===============================
        # FILE-LEVEL DRIFT
        # ===============================
        file_results = detect_file_level_changes(baseline_docs, current_docs)

        changed_files = [
            f"{item['file_name']} ({item['status']})"
            for item in file_results
            if item["status"] != "UNCHANGED"
        ]

        logger.info(f"Changed files: {changed_files}")

        # ===============================
        # DECISION
        # ===============================
        policy_changed = detect_policy_change(tfidf_shift, length_drift, kl_div)

        # ===============================
        # LOGGING
        # ===============================
        log_data = {
            "timestamp": timestamp,
            "num_files": len(current_docs),
            "tfidf_shift": tfidf_shift,
            "length_drift": length_drift,
            "kl_divergence": kl_div,
            "policy_changed": policy_changed,
            "changed_files": ", ".join(changed_files) if changed_files else "None"
        }

        save_drift_log(log_data)
        save_file_level_log(file_results, timestamp)

        # ===============================
        # BASELINE UPDATE
        # ===============================
        if policy_changed == "YES":
            logger.info("Policy change detected → updating baseline")
            save_baseline(current_docs)
        else:
            logger.info("No significant policy change detected")

        logger.info("Drift detection completed successfully")

    except Exception as e:
        logger.exception("Drift detection failed: %s", str(e))
