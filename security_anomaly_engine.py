# security_anomaly_engine.py
# Real-Time Security Intelligence for Retail Loss Prevention

class AntiTheftSecurityEngine:
    def __init__(self, weight_threshold_g=40.0):
        self.weight_threshold_g = weight_threshold_g

    def evaluate_security_risk(self, vision_item: dict, rfid_epc: str, measured_weight_g: float) -> dict:
        """
        Analyzes discrepancy vectors to flag tag-swapping, un-tagged placement, or concealment.
        """
        expected_weight = vision_item.get("unit_weight_g", 0.0)
        expected_epc = vision_item.get("epc", "")
        
        weight_delta = abs(measured_weight_g - expected_weight)
        
        # 1. Tag Swapping Attack Detection
        if rfid_epc != expected_epc and weight_delta > self.weight_threshold_g:
            return {
                "flag": "CRITICAL_SECURITY_ALERT",
                "risk_level": "HIGH",
                "cause": "TAG_SWAP_ATTEMPT_DETECTED",
                "action": "LOCK_TROLLEY_NOTIFY_SECURITY"
            }

        # 2. Concealment / Un-tagged Item Pick
        if measured_weight_g > (expected_weight + self.weight_threshold_g) and not rfid_epc:
            return {
                "flag": "UNTAGGED_ITEM_DETECTED",
                "risk_level": "MEDIUM",
                "cause": "WEIGHT_ADDED_WITHOUT_RFID_SCAN",
                "action": "PROMPT_USER_RESCAN"
            }

        return {"flag": "SECURE", "risk_level": "LOW", "cause": "NONE", "action": "CONTINUE"}