from ultralytics import YOLO

# Load your custom weights file
model = YOLO("best.pt")

# Print all class names with their internal class IDs
print("\n--- Allowed Products in best.pt ---")
for class_id, class_name in model.names.items():
    print(f"Class ID {class_id}: {class_name}")