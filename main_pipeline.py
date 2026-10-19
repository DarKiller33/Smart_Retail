import cv2
import time
import psutil
import csv
from yolo_detector import YOLOVisualDetector
from rfid_simulator import RFIDStreamSimulator
from fusion_engine import MultiModalFusionEngine

def run_framework(source=0, log_csv="edge_metrics_log.csv"):
    detector = YOLOVisualDetector(model_path="yolov8n.pt", conf_threshold=0.35)
    rfid_sim = RFIDStreamSimulator()
    fusion = MultiModalFusionEngine(w_vision=0.45, w_rfid=0.55, belief_threshold=0.60)

    # Active RFID tags simulated inside cart
    active_cart_epcs = ["EPC_001_BOTTLE", "EPC_003_APPLE", "EPC_004_CELLPHONE"]

    cap = cv2.VideoCapture(source)

    with open(log_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "Inference_Time_ms", "Fusion_Time_ms", "FPS", "CPU_Util_Percent", "RAM_Util_Percent"])

        frame_count = 0
        print("\n--- Starting Smart Shopping Framework Pipeline ---")
        print("Press 'q' in the video window to quit execution.\n")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            start_total_time = time.time()

            # 1. Vision Detection Step
            t_vision_start = time.time()
            visual_counts, detections, annotated_frame = detector.process_frame(frame)
            t_vision_end = time.time()
            inference_time_ms = (t_vision_end - t_vision_start) * 1000

            # 2. RFID Scanning Simulation
            rfid_counts, _ = rfid_sim.get_active_rfid_reads(active_cart_epcs, drop_rate=0.0)

            # 3. Multi-Modal Data Fusion (FIXED: passing 'detections' list)
            t_fusion_start = time.time()
            fused_inventory = fusion.synchronize_and_fuse(detections, rfid_counts)
            t_fusion_end = time.time()
            fusion_time_ms = (t_fusion_end - t_fusion_start) * 1000

            # Telemetry Metrics
            total_duration = time.time() - start_total_time
            fps = 1.0 / total_duration if total_duration > 0 else 0
            cpu_usage = psutil.cpu_percent()
            ram_usage = psutil.virtual_memory().percent

            writer.writerow([frame_count, f"{inference_time_ms:.2f}", f"{fusion_time_ms:.4f}", f"{fps:.2f}", cpu_usage, ram_usage])

            # Draw Telemetry Overlay
            y_offset = 30
            cv2.putText(annotated_frame, f"FPS: {fps:.1f} | CPU: {cpu_usage}% | RAM: {ram_usage}%", 
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            for item, info in fused_inventory.items():
                y_offset += 25
                text = f"{item.upper()}: V_Qty={info['visual_qty']} | RFID_Qty={info['rfid_qty']} | Belief={info['belief_score']} | Status={info['status']}"
                color = (0, 255, 0) if info['status'] == "VERIFIED_MATCH" else (0, 0, 255)
                cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            cv2.imshow("Multi-Modal Edge Smart Cart Pipeline", annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_framework(source=0)