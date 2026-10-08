import cv2
from ultralytics import YOLO

# Load YOLO26n
model = YOLO("yolo26n.pt")

# Vehicle classes used by AEGIS-X
VEHICLES = {"car", "truck", "bus", "motorcycle"}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    intersection = inter_w * inter_h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union == 0:
        return 0

    return intersection / union


while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    result = results[0]
    annotated = result.plot()

    vehicle_boxes = []

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            name = result.names[cls_id]

            if name in VEHICLES:
                coords = box.xyxy[0].tolist()
                vehicle_boxes.append(coords)

    vehicle_count = len(vehicle_boxes)

    # Default status
    status = "NORMAL"

    # Check possible vehicle collision/overlap
    possible_collision = False

    for i in range(vehicle_count):
        for j in range(i + 1, vehicle_count):
            iou = calculate_iou(vehicle_boxes[i], vehicle_boxes[j])

            if iou > 0.25:
                possible_collision = True

    if possible_collision:
        status = "POSSIBLE COLLISION"
    elif vehicle_count >= 4:
        status = "TRAFFIC BUILD-UP"

    # Display AEGIS-X status
    cv2.putText(
        annotated,
        f"Vehicles: {vehicle_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        annotated,
        f"AEGIS-X STATUS: {status}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("AEGIS-X - Incident Intelligence", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()+