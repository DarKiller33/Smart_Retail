import cv2
import numpy as np
import time
from yolo_detector import YOLOVisualDetector
from rfid_simulator import RFIDStreamSimulator
from fusion_engine import MultiModalFusionEngine
from product_db import lookup_barcode, lookup_yolo_class
from cart_manager import ShoppingCartManager

def run_advanced_smart_trolley(source=0):
    detector = YOLOVisualDetector(model_path="yolov8n.pt", conf_threshold=0.35)
    rfid_sim = RFIDStreamSimulator()
    fusion = MultiModalFusionEngine(w_vision=0.45, w_rfid=0.55, belief_threshold=0.60)
    cart = ShoppingCartManager(debounce_frames=3)

    # Built-in OpenCV Barcode Detector
    barcode_detector = cv2.barcode.BarcodeDetector()

    # Active RFID tags in cart
    active_cart_epcs = ["EPC_001_BOTTLE", "EPC_004_CELLPHONE"]

    cap = cv2.VideoCapture(source)
    print("\n--- Advanced Industrial Smart Trolley Terminal ---")
    print("Controls:")
    print("  - Point product barcode to camera for instant scan")
    print("  - Press 'c' to clear/reset cart to 0")
    print("  - Press 'q' to quit\n")

    last_barcode_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Barcode Detection Fallback Step
        try:
            ok, decoded_info, decoded_type = barcode_detector.detectAndDecode(frame)
            if ok and decoded_info:
                for barcode_val in decoded_info:
                    if barcode_val and (time.time() - last_barcode_time > 2.0): # 2s cooldown
                        product_match = lookup_barcode(barcode_val)
                        if product_match:
                            cart.add_item_by_barcode(product_match)
                            last_barcode_time = time.time()
                        else:
                            print(f"[BARCODE] Scanned Code: {barcode_val} (Unregistered SKU)")
        except Exception:
            pass

        # 2. Vision Pipeline Execution
        visual_counts, detections, annotated_frame = detector.process_frame(frame)

        # 3. RFID Simulation Read
        rfid_counts, _ = rfid_sim.get_active_rfid_reads(active_cart_epcs, drop_rate=0.0)

        # 4. Multi-Modal Fusion Engine
        fused_inventory = fusion.synchronize_and_fuse(detections, rfid_counts)

        # Enrich fused items with product database metadata
        for label, info in fused_inventory.items():
            meta = lookup_yolo_class(label)
            info["name"] = meta["name"]
            info["price"] = meta["price"]

        # Update Dynamic Cart Ledger
        cart.update_fused_state(fused_inventory)

        # 5. Render POS UI Terminal
        trolley_ui = render_pos_ui(annotated_frame, cart)

        cv2.imshow("Smart Shopping Cart - Edge AI Terminal", trolley_ui)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            cart.clear_cart()
            print("[CART EVENT] Shopping cart reset to empty (0 items).")

    cap.release()
    cv2.destroyAllWindows()

def render_pos_ui(annotated_frame, cart_manager):
    h, w, _ = annotated_frame.shape
    sidebar_w = 480
    ui = np.zeros((h, w + sidebar_w, 3), dtype=np.uint8)
    ui[:, :w] = annotated_frame

    sidebar = ui[:, w:]
    sidebar[:] = (25, 25, 30) # Dark theme

    # Header
    cv2.putText(sidebar, "SMART TROLLEY POS", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 215, 0), 2)
    cv2.putText(sidebar, "Multi-Modal Edge AI & Barcode Terminal", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)
    cv2.line(sidebar, (20, 70), (sidebar_w - 20, 70), (100, 100, 100), 1)

    # Table Header
    cv2.putText(sidebar, "ITEM SKU", (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(sidebar, "QTY", (210, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(sidebar, "PRICE", (270, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(sidebar, "SOURCE", (360, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.line(sidebar, (20, 105), (sidebar_w - 20, 105), (70, 70, 70), 1)

    y_pos = 130
    has_discrepancy = False

    if not cart_manager.cart_ledger:
        cv2.putText(sidebar, "Cart is Empty (0 Items)", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    else:
        for item_key, item in cart_manager.cart_ledger.items():
            qty = item["qty"]
            item_price = item["price"] * qty
            status = item.get("status", "VERIFIED_MATCH")
            source = item.get("source", "FUSION")

            if status != "VERIFIED_MATCH" and source != "BARCODE_SCAN":
                has_discrepancy = True

            name_disp = item["name"][:18]
            src_disp = "BARCODE" if source == "BARCODE_SCAN" else "FUSION"
            src_color = (0, 255, 255) if source == "BARCODE_SCAN" else (0, 255, 0)

            cv2.putText(sidebar, name_disp, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(sidebar, f"x{qty}", (210, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(sidebar, f"Rs.{item_price:.1f}", (270, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.putText(sidebar, src_disp, (360, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.4, src_color, 1)

            y_pos += 28

    # Totals Section
    subtotal, gst, grand_total = cart_manager.calculate_bill()
    summary_y = h - 130
    cv2.line(sidebar, (20, summary_y), (sidebar_w - 20, summary_y), (100, 100, 100), 1)

    cv2.putText(sidebar, f"Subtotal: Rs. {subtotal:.2f}", (20, summary_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    cv2.putText(sidebar, f"GST (18%): Rs. {gst:.2f}", (20, summary_y + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    cv2.putText(sidebar, f"GRAND TOTAL: Rs. {grand_total:.2f}", (20, summary_y + 85), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 255, 0), 2)

    # Status Banner
    if has_discrepancy:
        cv2.rectangle(sidebar, (20, h - 35), (sidebar_w - 20, h - 10), (0, 0, 180), -1)
        cv2.putText(sidebar, "FLAG: UNVERIFIED ITEM IN CART", (30, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
    else:
        cv2.rectangle(sidebar, (20, h - 35), (sidebar_w - 20, h - 10), (0, 120, 0), -1)
        cv2.putText(sidebar, "CART VERIFIED: SYSTEM READY", (35, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    return ui

if __name__ == "__main__":
    run_advanced_smart_trolley(source=0)