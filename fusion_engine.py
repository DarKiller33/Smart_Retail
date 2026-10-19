class MultiModalFusionEngine:
    def __init__(self, w_vision: float = 0.45, w_rfid: float = 0.55, belief_threshold: float = 0.60):
        """
        Bayesian-inspired probabilistic sensor fusion engine for retail edge inventory.
        """
        self.w_v = w_vision
        self.w_r = w_rfid
        self.threshold = belief_threshold

    def synchronize_and_fuse(self, detected_items: list, rfid_counts: dict) -> dict:
        """
        Fuses visual detections with RFID streams using weighted confidence belief scoring.
        """
        # Aggregate visual counts and average confidence per class
        visual_summary = {}
        for item in detected_items:
            label = item["label"]
            conf = item["confidence"]
            if label not in visual_summary:
                visual_summary[label] = {"count": 0, "conf_sum": 0.0}
            visual_summary[label]["count"] += 1
            visual_summary[label]["conf_sum"] += conf

        all_labels = set(visual_summary.keys()).union(set(rfid_counts.keys()))
        fused_inventory = {}

        for label in all_labels:
            v_data = visual_summary.get(label, {"count": 0, "conf_sum": 0.0})
            v_count = v_data["count"]
            avg_v_conf = (v_data["conf_sum"] / v_count) if v_count > 0 else 0.0

            r_count = rfid_counts.get(label, 0)
            r_signal = 1.0 if r_count > 0 else 0.0

            # Calculate mathematical belief score
            belief_score = (self.w_v * avg_v_conf) + (self.w_r * r_signal)

            # Classify anomaly states
            if v_count == r_count and v_count > 0:
                status = "VERIFIED_MATCH"
            elif r_count > v_count:
                status = "FLAG_OCCLUDED_VISION"  # RFID sees it, vision missed/misclassified it
            elif v_count > r_count:
                status = "FLAG_UNREGISTERED_TAG" # Vision sees it, RFID missing
            else:
                status = "UNCERTAIN"

            fused_inventory[label] = {
                "visual_qty": v_count,
                "rfid_qty": r_count,
                "avg_vision_conf": round(avg_v_conf, 2),
                "belief_score": round(belief_score, 2),
                "status": status
            }

        return fused_inventory