import time
import numpy as np
from ultralytics import YOLO

def run_edge_benchmark():
    print("--- Starting Edge AI Model Optimization & Export ---")
    
    # Load base PyTorch model
    pt_model = YOLO("yolov8n.pt")
    
    # Export to OpenVINO format (Optimized specifically for Intel/Windows CPUs)
    print("\n[1/3] Exporting YOLOv8n to OpenVINO format...")
    openvino_path = pt_model.export(format="openvino")
    print(f"OpenVINO Model generated at: {openvino_path}")

    openvino_model = YOLO(openvino_path, task="detect")
    dummy_frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Warmup runs
    for _ in range(10):
        _ = pt_model(dummy_frame, verbose=False)
        _ = openvino_model(dummy_frame, verbose=False)

    # Benchmark PyTorch FP32
    print("\n[2/3] Benchmarking PyTorch FP32 Model (100 iterations)...")
    start_pt = time.time()
    for _ in range(100):
        _ = pt_model(dummy_frame, verbose=False)
    pt_latency = ((time.time() - start_pt) / 100) * 1000

    # Benchmark OpenVINO Edge Format
    print("[3/3] Benchmarking OpenVINO Engine Model (100 iterations)...")
    start_ov = time.time()
    for _ in range(100):
        _ = openvino_model(dummy_frame, verbose=False)
    ov_latency = ((time.time() - start_ov) / 100) * 1000

    print("\n================ BENCHMARK RESULTS ================")
    print(f"PyTorch FP32 Average Latency : {pt_latency:.2f} ms")
    print(f"OpenVINO Edge Average Latency: {ov_latency:.2f} ms")
    speedup = ((pt_latency - ov_latency) / pt_latency) * 100
    print(f"Latency Reduction (Speedup)  : {speedup:.2f}%")
    print("===================================================")

if __name__ == "__main__":
    run_edge_benchmark()