# rfid_engine.py
# Dual-Mode UHF RFID Reader Engine (Hardware Driver + Research Simulation)

import time
import random

try:
    import serial  # PySerial for physical USB/UART RFID readers
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

class RFIDReaderSystem:
    def __init__(self, mode="AUTO", port="COM3", baudrate=115200):
        """
        mode: 'AUTO', 'HARDWARE', or 'SIMULATION'
        """
        self.mode = mode
        self.port = port
        self.baudrate = baudrate
        self.hardware_connected = False
        self.serial_conn = None

        # Try connecting to physical RFID hardware
        if self.mode in ["AUTO", "HARDWARE"] and SERIAL_AVAILABLE:
            try:
                self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)
                self.hardware_connected = True
                print(f"[RFID ENGINE] Connected to Physical Hardware Reader on {self.port}")
            except Exception:
                self.hardware_connected = False
                print(f"[RFID ENGINE] Hardware not detected on {self.port}. Running in Research Simulation Mode.")

    def poll_rfid_tags(self, active_cart_items: dict) -> dict:
        """
        Polls RFID tags.
        Returns: {epc_id: {"qty": int, "rssi_dbm": float, "label": str}}
        """
        rfid_reads = {}

        # 1. Physical Hardware Execution
        if self.hardware_connected and self.serial_conn:
            try:
                if self.serial_conn.in_waiting > 0:
                    raw_line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                    if raw_line.startswith("EPC:"):
                        epc = raw_line.split("EPC:")[1].split(",")[0].strip()
                        rfid_reads[epc] = {"qty": 1, "rssi_dbm": -55.0, "source": "HARDWARE_READER"}
                return rfid_reads
            except Exception as e:
                print(f"[RFID ERROR] Serial Read Exception: {e}")

        # 2. Research Simulation Mode (For Laptop Benchmarking & Monte Carlo Testing)
        for label, item_info in active_cart_items.items():
            epc = item_info.get("epc", f"EPC_{label.upper()}")
            qty = item_info.get("qty", 1)

            # Model RF Path Loss & Tag Read Probabilities
            # Simulated 98.5% read success rate per active tag
            if random.random() < 0.985:
                # Simulated RSSI signal strength between -45 dBm and -65 dBm
                rssi = round(random.uniform(-45.0, -65.0), 2)
                rfid_reads[label] = {
                    "epc": epc,
                    "qty": qty,
                    "rssi_dbm": rssi,
                    "source": "SIMULATED_UHF_STREAM"
                }

        return rfid_reads

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()