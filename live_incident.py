import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import math

# -----------------------------
# Load YOLO model
# -----------------------------
model = YOLO("yolo26n.pt")

# -----------------------------
# Open live webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permission.")

# Vehicle classes
VEHICLES = {"car", "truck", "bus", "motorcycle"}

# Store recent center positions for each tracked vehicle
history = defaultdict(lambda: deque(maxlen=8))

# Store collision evidence for vehicle pairs
overlap_frames = defaultdict(int)


def center_of(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)

    intersection = w * h

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read camera frame.")
        break

    # -----------------------------------------
    # YOLO + ByteTrack
    # -----------------------------------------
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    result = results[0]
    output = result.plot()

    vehicles = []

    # -----------------------------------------
    # Read tracked vehicles
    # -----------------------------------------
    if result.boxes is not None and result.boxes.id is not None:

        ids = result.boxes.id.int().cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        boxes = result.boxes.xyxy.cpu().tolist()

        for track_id, class_id, box in zip(ids, classes, boxes):

            class_name = result.names[class_id]

            if class_name not in VEHICLES:
                continue

            centre = center_of(box)

            history[track_id].append(centre)

            vehicles.append({
                "id": track_id,
                "name": class_name,
                "box": box,
                "center": centre
            })

    vehicle_count = len(vehicles)

    # -----------------------------------------
    # Detect stationary vehicles
    # -----------------------------------------
    stationary_count = 0

    for vehicle in vehicles:
        track_id = vehicle["id"]
        positions = history[track_id]

        if len(positions) >= 6:
            movement = 0

            for i in range(1, len(positions)):
                movement += distance(
                    positions[i - 1],
                    positions[i]
                )

            # Very small movement = possible stopped vehicle
            if movement < 25:
                stationary_count += 1

    # -----------------------------------------
    # Detect persistent overlap
    # -----------------------------------------
    collision_detected = False

    for i in range(len(vehicles)):
        for j in range(i + 1, len(vehicles)):

            id1 = vehicles[i]["id"]
            id2 = vehicles[j]["id"]

            pair = tuple(sorted((id1, id2)))

            iou = calculate_iou(
                vehicles[i]["box"],
                vehicles[j]["box"]
            )

            if iou > 0.15:
                overlap_frames[pair] += 1
            else:
                overlap_frames[pair] = 0

            # Persistent overlap = possible collision
            if overlap_frames[pair] >= 5:
                collision_detected = True

    # -----------------------------------------
    # AEGIS-X Risk Intelligence
    # -----------------------------------------
    risk_score = 0
    status = "NORMAL"

    if vehicle_count >= 4:
        risk_score += 20

    if vehicle_count >= 6:
        risk_score += 15

    risk_score += stationary_count * 10

    if collision_detected:
        risk_score += 40

    risk_score = min(risk_score, 100)

    if collision_detected:
        status = "POSSIBLE INCIDENT"
    elif risk_score >= 50:
        status = "WARNING"
    else:
        status = "NORMAL"

    # -----------------------------------------
    # Display AEGIS-X information
    # -----------------------------------------
    cv2.putText(
        output,
        f"Vehicles: {vehicle_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Stopped Vehicles: {stationary_count}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Risk Score: {risk_score}/100",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"AEGIS-X: {status}",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2
    )

    cv2.imshow("AEGIS-X - Live Incident Intelligence", output)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()