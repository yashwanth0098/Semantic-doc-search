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
