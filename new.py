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
                "old_policy_name": "None",
                "changed_policy_name": "None",
                "changed_status": "NO"
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

        # OLD POLICY FILES
        old_policy_names = ", ".join(baseline_docs.keys())

        # CHANGED FILES
        changed_files = [
            f"{item['file_name']} ({item['status']})"
            for item in file_results
            if item["status"] != "UNCHANGED"
        ]

        changed_policy_names = ", ".join(changed_files) if changed_files else "None"

        logger.info(f"Changed files: {changed_policy_names}")

        # ===============================
        # DECISION
        # ===============================
        policy_changed = detect_policy_change(
            tfidf_shift,
            length_drift,
            kl_div
        )

        # ===============================
        # LOGGING (FINAL FORMAT)
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
