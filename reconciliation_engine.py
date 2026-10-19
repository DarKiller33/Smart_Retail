# reconciliation_engine.py
# Triple Reconciliation Gate Logic (Vision, RFID, and HX711 Weight Sensor)

class TripleReconciliationGate:
    def __init__(self, weight_tolerance_g: float = 50.0):
        self.weight_tolerance_g = weight_tolerance_g

    def evaluate_gate(self, vision_counts: dict, rfid_counts: dict, measured_weight_g: float, catalog_lookup_fn) -> tuple:
        """
        Evaluates N_Vision == N_RFID == N_Weight
        Returns: (reconciliation_status: str, expected_weight_g: float, inventory_summary: dict)
        """
        all_labels = set(vision_counts.keys()).union(set(rfid_counts.keys()))
        inventory_summary = {}
        expected_total_weight_g = 0.0
        is_mismatch = False

        for label in all_labels:
            n_v = vision_counts.get(label, 0)
            n_r = rfid_counts.get(label, 0)

            meta = catalog_lookup_fn(label)
            item_weight = meta["unit_weight_g"]

            # Calculate expected weight contribution
            expected_qty = max(n_v, n_r)
            expected_total_weight_g += (expected_qty * item_weight)

            # Check Count Match
            if n_v == n_r and n_v > 0:
                item_status = "MATCH"
            elif n_r > n_v:
                item_status = "FLAG_VISION_OCCLUSION"
                is_mismatch = True
            elif n_v > n_r:
                item_status = "FLAG_UNREGISTERED_TAG"
                is_mismatch = True
            else:
                item_status = "EMPTY"

            inventory_summary[label] = {
                "name": meta["name"],
                "price": meta["price"],
                "n_v": n_v,
                "n_r": n_r,
                "status": item_status
            }

        # Check Weight Signal Match
        weight_diff = abs(measured_weight_g - expected_total_weight_g)
        weight_match = weight_diff <= self.weight_tolerance_g

        if not is_mismatch and weight_match:
            gate_status = "TRIPLE_MATCH_CHECKOUT_UNLOCKED"
        else:
            gate_status = "RECONCILIATION_MISMATCH_BLOCKED"

        return gate_status, expected_total_weight_g, inventory_summary