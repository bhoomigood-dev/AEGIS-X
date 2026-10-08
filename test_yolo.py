from ultralytics import YOLO

# Load the lightweight YOLO26 model
model = YOLO("yolo26n.pt")

print("YOLO26n loaded successfully!")