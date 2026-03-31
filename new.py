def run_drift_detection():
    logger.info("Running drift detection...")

    try:
        # ===============================
        # LOAD DOCUMENTS
        # ===============================
        current_docs = load_processed_documents()
        baseline_docs = load_baseline()

        if not current_docs:
            logger.warning("No processed documents found. Skipping drift detection.")
            return

        timestamp = get_latest_timestamp()

        # ===============================
        # FIRST RUN → CREATE BASELINE
        # ===============================
        if baseline_docs is None:
            logger.info("No baseline found. Creating baseline.")

            save_baseline(current_docs)

            log_data = {
                "timestamp": timestamp,
                "num_files": len(current_docs),
                "tfidf_shift": 0.0,
                "length_drift": 0.0,
                "kl_divergence": 0.0,
                "old_policy_name": "NONE",
                "changed_policy_name": "NONE",
                "changed_status": "NO"
            }

            save_drift_log(log_data)

            logger.info("Initial baseline created and logged successfully")
            return

        # ===============================
        # DRIFT COMPUTATION
        # ===============================
        old_docs = list(baseline_docs.values())
        new_docs = list(current_docs.values())

        tfidf_shift = compute_tfidf_shift(old_docs, new_docs)
        length_drift = compute_length_drift(old_docs, new_docs)
        kl_div = compute_kl_divergence(old_docs, new_docs)

        logger.info(
            "Drift Metrics → TF-IDF: %.4f | Length: %.4f | KL: %.4f",
            tfidf_shift,
            length_drift,
            kl_div
        )

        # ===============================
        # FILE-LEVEL DRIFT  ✅ IMPORTANT
        # ===============================
        file_results = detect_file_level_changes(
            baseline_docs,
            current_docs
        )

        # ===============================
        # BUILD POLICY NAME FIELDS
        # ===============================

        # OLD POLICY NAMES (only affected ones)
        old_changed_files = [
            item["file_name"]
            for item in file_results
            if item["status"] in ["MODIFIED", "REMOVED"]
        ]
        old_policy_names = ", ".join(old_changed_files) if old_changed_files else "NONE"

        # CHANGED POLICY NAMES
        changed_files = [
            f"{item['file_name']} ({item['status']})"
            for item in file_results
            if item["status"] in ["NEW", "MODIFIED", "REMOVED"]
        ]
        changed_policy_names = ", ".join(changed_files) if changed_files else "NONE"

        # ===============================
        # GLOBAL DECISION
        # ===============================
        policy_changed = detect_policy_change(
            tfidf_shift,
            length_drift,
            kl_div
        )

        # ===============================
        # GLOBAL LOG
        # ===============================
        log_data = {
            "timestamp": timestamp,
            "num_files": len(current_docs),
            "tfidf_shift": tfidf_shift,
            "length_drift": length_drift,
            "kl_divergence": kl_div,
            "old_policy_name": old_policy_names,
            "changed_policy_name": changed_policy_names,
            "changed_status": policy_changed
        }

        save_drift_log(log_data)

        # ===============================
        # FILE-LEVEL LOG (per document) ✅
        # ===============================
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






def detect_file_level_changes(
    baseline_docs: Dict[str, str],
    current_docs: Dict[str, str],
    similarity_threshold: float = 0.90,
    length_threshold: float = 0.15
) -> List[Dict]:

    results = []

    # ===============================
    # FIT VECTORIZER ON ALL DOCS
    # ===============================
    all_docs = list(baseline_docs.values()) + list(current_docs.values())

    vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
    vectorizer.fit(all_docs)

    # ===============================
    # CURRENT FILES CHECK
    # ===============================
    for file_name, new_text in current_docs.items():

        # NEW FILE
        if file_name not in baseline_docs:
            results.append({
                "old_file_name": "NONE",
                "new_file_name": file_name,
                "status": "NEW",
                "similarity": None,
                "length_change": None
            })
            continue

        old_text = baseline_docs[file_name]

        vec_old = vectorizer.transform([old_text])
        vec_new = vectorizer.transform([new_text])

        similarity = cosine_similarity(vec_old, vec_new)[0][0]

        old_len = len(old_text)
        new_len = len(new_text)
        length_change = abs(new_len - old_len) / (old_len + 1e-5)

        # DECISION
        if similarity < similarity_threshold or length_change > length_threshold:
            status = "MODIFIED"
        else:
            status = "UNCHANGED"

        results.append({
            "old_file_name": file_name,
            "new_file_name": file_name,
            "status": status,
            "similarity": round(float(similarity), 4),
            "length_change": round(float(length_change), 4)
        })

    # ===============================
    # REMOVED FILES
    # ===============================
    for file_name in baseline_docs:
        if file_name not in current_docs:
            results.append({
                "old_file_name": file_name,
                "new_file_name": "NONE",
                "status": "REMOVED",
                "similarity": None,
                "length_change": None
            })

    return results






def save_file_level_log(file_results: List[Dict], timestamp: str):
    file_log_path = DRIFT_DIR / "document_file_level_drift_log.csv"

    rows = []
    for item in file_results:
        rows.append({
            "timestamp": timestamp,
            "old_file_name": item["old_file_name"],
            "new_file_name": item["new_file_name"],
            "status": item["status"],
            "similarity": item.get("similarity"),
            "length_change": item.get("length_change")
        })

    df = pd.DataFrame(rows)

    if file_log_path.exists():
        df.to_csv(file_log_path, mode="a", header=False, index=False)
    else:
        df.to_csv(file_log_path, index=False)







def build_drift_summary(results):
    changed_files = []
    old_files = []

    for r in results:
        status = r["status"]
        old_name = r["old_file_name"]
        new_name = r["new_file_name"]

        # Track old files (baseline snapshot)
        if old_name != "NONE":
            old_files.append(old_name)

        # Build changed file list
        if status == "NEW":
            changed_files.append(f"{new_name} (NEW)")

        elif status == "MODIFIED":
            changed_files.append(f"{new_name} (MODIFIED)")

        elif status == "REMOVED":
            changed_files.append(f"{old_name} (REMOVED)")

    return {
        "old_policy_name": ", ".join(sorted(set(old_files))) if old_files else None,
        "changed_policy_name": ", ".join(changed_files) if changed_files else None,
        "changed_status": "YES" if changed_files else "NO"
    }
