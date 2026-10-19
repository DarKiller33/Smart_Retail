import cv2
from ultralytics import YOLO

class YOLOVisualDetector:
    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.45):
        """
        Lightweight YOLO detector with ByteTrack persistent tracking for retail inventory.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        
        # COCO Class IDs for typical retail/desk testing items:
        # 39: bottle, 41: cup, 63: laptop, 64: mouse, 67: cell phone, 73: book, 47: apple
        self.target_classes = [39, 41, 63, 64, 67, 73, 47]

    def process_frame(self, frame):
        """
        Runs object detection and persistent tracking to stabilize frame-to-frame predictions.
        """
        # Run ByteTrack tracking across frames using persist=True
        results = self.model.track(
            frame, 
            conf=self.conf_threshold, 
            classes=self.target_classes, 
            persist=True, 
            verbose=False
        )[0]

        detected_items = []
        visual_counts = {}

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                class_name = self.model.names[cls_id]
                confidence = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else None
                xyxy = box.xyxy[0].cpu().numpy().tolist()

                detected_items.append({
                    "track_id": track_id,
                    "label": class_name,
                    "confidence": confidence,
                    "bbox": xyxy
                })

                visual_counts[class_name] = visual_counts.get(class_name, 0) + 1

        annotated_frame = results.plot()
        return visual_counts, detected_items, annotated_frame