import cv2
import numpy as np
import time
import os
import math
from ultralytics import YOLO

from product_db import get_product_metadata, ALLOWED_RETAIL_CLASSES
from reconciliation_engine import TripleReconciliationGate
from rfid_engine import RFIDReaderSystem

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea <= 0:
        return 0.0
    
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

def get_centroid(box):
    return int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)

class MasterSmartTrolleyApp:
    def __init__(self, model_path="yolov8n.pt"):
        if not os.path.exists(model_path):
            model_path = "yolov8n.pt"
            
        self.model = YOLO(model_path)
        self.rfid_system = RFIDReaderSystem(mode="AUTO")
        self.reconciler = TripleReconciliationGate(weight_tolerance_g=50.0)

        # System Parameters
        self.CONF_THRESH = 0.38
        self.VERIFY_SECS = 0.4
        self.REMOVE_TIMEOUT_SECS = 3.0
        
        # State Management
        self.active_tracks = {}
        self.next_track_id = 101
        self.paid_ledger = set()            # EPC tags successfully paid for
        self.unpaid_stolen_epcs = set()      # Unpaid tags injected during theft simulation
        
        # View Modes
        self.show_navigation = False
        self.show_eco_health = False
        self.exit_gate_alarm = False
        self.exit_gate_msg = ""
        
        self.toast_msg = ""
        self.toast_time = 0

    def process_frame(self, frame):
        h, w, _ = frame.shape
        now = time.time()

        # 1. Computer Vision Processing
        results = self.model.predict(frame, conf=self.CONF_THRESH, verbose=False)[0]
        matched_track_ids = set()

        if results.boxes is not None and len(results.boxes) > 0:
            raw_boxes = results.boxes.xyxy.cpu().numpy().astype(int)
            raw_cls_ids = results.boxes.cls.cpu().numpy().astype(int)
            raw_confs = results.boxes.conf.cpu().numpy()

            nms_boxes = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in raw_boxes]
            indices = cv2.dnn.NMSBoxes(nms_boxes, raw_confs.tolist(), self.CONF_THRESH, 0.45)

            if len(indices) > 0:
                indices = indices.flatten()
                boxes = raw_boxes[indices]
                cls_ids = raw_cls_ids[indices]
                confs = raw_confs[indices]

                for box, cls_id, conf in zip(boxes, cls_ids, confs):
                    raw_label = self.model.names[cls_id].lower()

                    if raw_label not in ALLOWED_RETAIL_CLASSES:
                        continue

                    best_match_id = None
                    best_score = 0.0
                    c_curr = get_centroid(box)

                    for tid, tdata in self.active_tracks.items():
                        iou = calculate_iou(box, tdata["last_box"])
                        c_prev = get_centroid(tdata["last_box"])
                        dist = math.hypot(c_curr[0] - c_prev[0], c_curr[1] - c_prev[1])

                        if (iou > 0.25 or (dist < 150 and tdata["label"] == raw_label)) and iou >= best_score:
                            best_score = iou
                            best_match_id = tid

                    if best_match_id is not None:
                        track_id = best_match_id
                        tdata = self.active_tracks[track_id]
                        tdata["last_seen"] = now
                        tdata["last_box"] = box
                    else:
                        track_id = self.next_track_id
                        self.next_track_id += 1
                        self.active_tracks[track_id] = {
                            "label": raw_label,
                            "first_seen": now,
                            "last_seen": now,
                            "last_box": box,
                            "verified": False
                        }

                    matched_track_ids.add(track_id)
                    tdata = self.active_tracks[track_id]

                    elapsed = now - tdata["first_seen"]
                    if elapsed >= self.VERIFY_SECS and not tdata["verified"]:
                        tdata["verified"] = True
                        meta = get_product_metadata(tdata["label"])
                        self.toast_msg = f"+ Verified {meta['name']} in Cart"
                        self.toast_time = now

                    meta = get_product_metadata(tdata["label"])
                    color = (0, 255, 0) if tdata["verified"] else (0, 255, 255)

                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                    cv2.putText(frame, f"{meta['name']} ({conf:.2f})", 
                                (box[0], max(25, box[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # 2. Persistence Cleanup
        vision_counts = {}
        expired_ids = []

        for tid, tdata in list(self.active_tracks.items()):
            time_absent = now - tdata["last_seen"]
            if time_absent >= self.REMOVE_TIMEOUT_SECS:
                expired_ids.append(tid)
            else:
                if tdata["verified"]:
                    lbl = tdata["label"]
                    vision_counts[lbl] = vision_counts.get(lbl, 0) + 1

        for tid in expired_ids:
            lbl = self.active_tracks[tid]["label"]
            if self.active_tracks[tid]["verified"]:
                meta = get_product_metadata(lbl)
                self.toast_msg = f"- Removed {meta['name']} from Cart"
                self.toast_time = now
            del self.active_tracks[tid]

        # 3. Synchronize with Primary RFID Engine Stream
        active_cart_state = {
            lbl: {"qty": qty, "epc": get_product_metadata(lbl)["epc"]} 
            for lbl, qty in vision_counts.items()
        }
        
        rfid_raw_reads = self.rfid_system.poll_rfid_tags(active_cart_state)
        rfid_counts = {lbl: data["qty"] for lbl, data in rfid_raw_reads.items()}

        # 4. Mass Profiling Weight Calculation
        simulated_weight = sum(
            get_product_metadata(lbl)["unit_weight_g"] * qty 
            for lbl, qty in rfid_counts.items()
        )

        # 5. Triple Reconciliation Gate
        gate_status, exp_weight, _ = self.reconciler.evaluate_gate(
            vision_counts=vision_counts,
            rfid_counts=rfid_counts,
            measured_weight_g=simulated_weight,
            catalog_lookup_fn=get_product_metadata
        )

        # Render Indoor Navigation or Eco Overlay onto camera frame if toggled
        if self.show_navigation:
            self.overlay_navigation_map(frame)
        elif self.show_eco_health:
            self.overlay_eco_analytics(frame, vision_counts)

        return self.render_pos_terminal(frame, rfid_counts, vision_counts, gate_status)

    def trigger_payment_checkout(self):
        """Simulates payment completion and marks current cart tags as PAID."""
        for tid, tdata in self.active_tracks.items():
            if tdata["verified"]:
                meta = get_product_metadata(tdata["label"])
                self.paid_ledger.add(meta["epc"])
        self.toast_msg = "PAYMENT SUCCESSFUL! All RFID Tags Marked Paid."
        self.toast_time = time.time()

    def simulate_unpaid_theft_injection(self):
        """Injects an unscanned RFID tag into the cart without vision detection."""
        stolen_epc = "EPC_UNPAID_BOOSTER_PACK_007"
        self.unpaid_stolen_epcs.add(stolen_epc)
        self.toast_msg = "THEFT INJECTED: Unscanned RFID Tag placed in Cart!"
        self.toast_time = time.time()

    def pass_through_rfid_exit_gate(self):
        """Simulates passage through the RFID Exit Gate (EAS Portal)."""
        current_cart_epcs = set()
        
        # Scanned vision items
        for tid, tdata in self.active_tracks.items():
            if tdata["verified"]:
                meta = get_product_metadata(tdata["label"])
                current_cart_epcs.add(meta["epc"])
                
        # Include injected stolen tags
        current_cart_epcs.update(self.unpaid_stolen_epcs)

        # Check for unpaid tags
        unpaid_found = current_cart_epcs - self.paid_ledger

        if len(unpaid_found) > 0:
            self.exit_gate_alarm = True
            self.exit_gate_msg = f"SECURITY ALARM! Unpaid RFID Tag Detected: {list(unpaid_found)[0]}"
            self.toast_msg = "EXIT GATE: THEFT DETECTED - ALARM TRIGGERED!"
        else:
            self.exit_gate_alarm = False
            self.exit_gate_msg = "EXIT GATE CLEAR: Thank you for shopping!"
            self.toast_msg = "EXIT GATE: PASS APPROVED"
        self.toast_time = time.time()

    def overlay_navigation_map(self, frame):
        """Draws a 2D store map showing aisle routes."""
        h, w, _ = frame.shape
        map_w, map_h = 220, 160
        map_box = np.zeros((map_h, map_w, 3), dtype=np.uint8)
        map_box[:] = (40, 40, 40)

        # Map Layout
        cv2.putText(map_box, "STORE MAP (Aisle Nav)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 215, 0), 1)
        cv2.rectangle(map_box, (20, 35), (80, 80), (100, 100, 100), -1)  # Aisle 1
        cv2.putText(map_box, "Produce", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        cv2.rectangle(map_box, (120, 35), (180, 80), (100, 100, 100), -1) # Aisle 2
        cv2.putText(map_box, "Drinks", (125, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

        # Path Vector (Green Line)
        cv2.arrowedLine(map_box, (50, 130), (50, 85), (0, 255, 0), 2)
        cv2.circle(map_box, (50, 130), 5, (0, 0, 255), -1) # Cart Position
        cv2.putText(map_box, "You are here", (65, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)

        frame[h - map_h - 20:h - 20, 20:20 + map_w] = map_box

    def overlay_eco_analytics(self, frame, vision_counts):
        """Displays carbon footprint and nutritional summary."""
        h, w, _ = frame.shape
        panel_w, panel_h = 240, 130
        panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        panel[:] = (35, 45, 30)

        total_items = sum(vision_counts.values())
        est_calories = total_items * 95
        est_co2_kg = round(total_items * 0.18, 2)

        cv2.putText(panel, "ECO-HEALTH ANALYTICS", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1)
        cv2.putText(panel, f"Nutri-Grade Score: A", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(panel, f"Total Est. Energy: {est_calories} kcal", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(panel, f"Carbon Impact: {est_co2_kg} kg CO2e", (10, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

        frame[h - panel_h - 20:h - 20, 20:20 + panel_w] = panel

    def render_pos_terminal(self, frame, rfid_counts, vision_counts, gate_status):
        h, w, _ = frame.shape
        sidebar_w = 460
        canvas = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
        canvas[:, :w] = frame

        sidebar = canvas[:, w:]
        sidebar[:] = (28, 24, 20)

        # POS Header
        cv2.putText(sidebar, "SMART TROLLEY POS", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 215, 0), 2)
        cv2.putText(sidebar, "Primary RFID + Dual Verification Terminal", (20, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        cv2.line(sidebar, (20, 68), (sidebar_w - 20, 68), (100, 100, 100), 1)

        # Table Header
        cv2.putText(sidebar, "ITEM SKU", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(sidebar, "RFID", (190, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(sidebar, "VISION", (250, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.putText(sidebar, "PRICE", (320, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
        cv2.line(sidebar, (20, 100), (sidebar_w - 20, 100), (70, 70, 70), 1)

        y_pos = 125
        subtotal = 0.0

        all_item_keys = set(rfid_counts.keys()).union(set(vision_counts.keys()))

        if not all_item_keys:
            cv2.putText(sidebar, "Cart is Empty (0 Items Scanned)", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (120, 120, 120), 1)
        else:
            for label in all_item_keys:
                meta = get_product_metadata(label)
                r_qty = rfid_counts.get(label, 0)
                v_qty = vision_counts.get(label, 0)
                
                item_price = meta["price"] * max(r_qty, v_qty)
                subtotal += item_price

                color = (0, 255, 0) if r_qty == v_qty else (0, 0, 255)

                cv2.putText(sidebar, meta["name"][:16], (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                cv2.putText(sidebar, f"x{r_qty}", (195, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
                cv2.putText(sidebar, f"x{v_qty}", (255, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)
                cv2.putText(sidebar, f"Rs.{item_price:.1f}", (320, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                y_pos += 26

        # Financial Summary
        gst = subtotal * 0.05
        grand_total = subtotal + gst

        summary_y = h - 160
        cv2.line(sidebar, (20, summary_y), (sidebar_w - 20, summary_y), (100, 100, 100), 1)
        cv2.putText(sidebar, f"Subtotal: Rs. {subtotal:.2f}", (20, summary_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(sidebar, f"GST (5%): Rs. {gst:.2f}", (20, summary_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(sidebar, f"GRAND TOTAL: Rs. {grand_total:.2f}", (20, summary_y + 70), cv2.FONT_HERSHEY_DUPLEX, 0.65, (0, 255, 0), 2)

        # Interactive Controls Legend
        cv2.line(sidebar, (20, summary_y + 82), (sidebar_w - 20, summary_y + 82), (60, 60, 60), 1)
        cv2.putText(sidebar, "[P] Pay | [E] Pass Exit Gate | [T] Inject Theft", (20, summary_y + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)
        cv2.putText(sidebar, "[N] Map Nav | [H] Eco-Health | [Q] Exit App", (20, summary_y + 115), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)

        # RFID Exit Gate Status Display
        if self.exit_gate_alarm:
            cv2.rectangle(sidebar, (20, h - 35), (sidebar_w - 20, h - 10), (0, 0, 220), -1)
            cv2.putText(sidebar, "SECURITY ALARM: UNPAID RFID ITEM!", (28, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        elif self.exit_gate_msg:
            cv2.rectangle(sidebar, (20, h - 35), (sidebar_w - 20, h - 10), (0, 120, 0), -1)
            cv2.putText(sidebar, "EXIT GATE: CLEAR TO LEAVE", (50, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        elif gate_status == "TRIPLE_MATCH_CHECKOUT_UNLOCKED":
            cv2.rectangle(sidebar, (20, h - 35), (sidebar_w - 20, h - 10), (0, 100, 0), -1)
            cv2.putText(sidebar, "RECONCILIATION MATCH: READY", (30, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        else:
            cv2.rectangle(sidebar, (20, h - 35), (sidebar_w - 20, h - 10), (0, 0, 180), -1)
            cv2.putText(sidebar, "GATE MISMATCH DETECTED", (40, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # Toast Bar
        if time.time() - self.toast_time < 2.5 and self.toast_msg:
            cv2.rectangle(canvas, (20, h - 45), (420, h - 15), (0, 0, 0), -1)
            cv2.putText(canvas, self.toast_msg, (30, h - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1)

        return canvas

    def close(self):
        self.rfid_system.close()

def main():
    trolley = MasterSmartTrolleyApp(model_path="yolov8n.pt")
    cap = cv2.VideoCapture(0)

    print("\n--- Master Smart Trolley System Active ---")
    print("Demo Keybindings:")
    print("  [P] : Process Payment (Mark RFID tags as PAID)")
    print("  [E] : Pass through RFID Exit Gate Portal")
    print("  [T] : Inject Unpaid Theft Tag into Cart")
    print("  [N] : Toggle Indoor Aisle Navigation Map")
    print("  [H] : Toggle Eco-Health & Carbon Footprint Panel")
    print("  [Q] : Quit Application\n")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        ui_canvas = trolley.process_frame(frame)
        cv2.imshow("Smart Shopping Trolley - Integrated Terminal", ui_canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            trolley.trigger_payment_checkout()
        elif key == ord('e'):
            trolley.pass_through_rfid_exit_gate()
        elif key == ord('t'):
            trolley.simulate_unpaid_theft_injection()
        elif key == ord('n'):
            trolley.show_navigation = not trolley.show_navigation
        elif key == ord('h'):
            trolley.show_eco_health = not trolley.show_eco_health

    trolley.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()