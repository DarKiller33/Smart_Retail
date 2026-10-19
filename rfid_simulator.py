import time
import random

class RFIDStreamSimulator:
    def __init__(self, tag_database: dict = None):
        """
        Simulates an RFID reader stream emitting EPC tag reads.
        """
        # Default EPC tag mapping: Tag ID -> Item Label
        self.tag_database = tag_database or {
            "EPC_001_BOTTLE": "bottle",
            "EPC_002_BOTTLE": "bottle",
            "EPC_003_APPLE": "apple",
            "EPC_004_CELLPHONE": "cell phone",
            "EPC_005_BOOK": "book"
        }

    def get_active_rfid_reads(self, active_epcs: list, drop_rate: float = 0.05) -> dict:
        """
        Returns scanned RFID tags with a small artificial drop rate to simulate signal noise.
        """
        scanned_counts = {}
        scanned_tags = []

        for epc in active_epcs:
            # Simulate real-world RFID attenuation / drop rate
            if random.random() > drop_rate:
                if epc in self.tag_database:
                    item_label = self.tag_database[epc]
                    scanned_counts[item_label] = scanned_counts.get(item_label, 0) + 1
                    scanned_tags.append({"epc": epc, "label": item_label, "timestamp": time.time()})

        return scanned_counts, scanned_tags