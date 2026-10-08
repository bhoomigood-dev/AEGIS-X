import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
import math

# -----------------------------
# Load YOLO26n
# -----------------------------
model = YOLO("yolo26n.pt")

# -----------------------------
# Open LIVE CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

VEHICLES = {"car", "truck", "bus", "motorcycle"}

# Store recent positions of tracked vehicles
history = defaultdict(lambda: deque(maxlen=8))

# Store persistent overlap between vehicle pairs
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

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    intersection = width * height

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

    # -----------------------------
    # YOLO + ByteTrack
    # -----------------------------
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

    # -----------------------------
    # Collect tracked vehicles
    # -----------------------------
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

    # -----------------------------
    # Detect stopped vehicles
    # -----------------------------
    stopped_count = 0

    for vehicle in vehicles:

        positions = history[vehicle["id"]]

        if len(positions) >= 6:

            movement = 0

            for i in range(1, len(positions)):
                movement += distance(
                    positions[i - 1],
                    positions[i]
                )

            if movement < 25:
                stopped_count += 1

    # -----------------------------
    # Detect persistent overlap
    # -----------------------------
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

            if overlap_frames[pair] >= 5:
                collision_detected = True

    # -----------------------------
    # Risk calculation
    # -----------------------------
    risk = 0
    reasons = []

    if vehicle_count >= 4:
        risk += 20
        reasons.append("High vehicle density")

    if vehicle_count >= 6:
        risk += 15
        reasons.append("Heavy traffic")

    if stopped_count > 0:
        risk += stopped_count * 10
        reasons.append("Stopped vehicle detected")

    if collision_detected:
        risk += 40
        reasons.append("Persistent vehicle overlap")

    risk = min(risk, 100)

    # -----------------------------
    # Severity
    # -----------------------------
    if risk >= 70:
        severity = "HIGH"
    elif risk >= 40:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    # -----------------------------
    # Incident status
    # -----------------------------
    if collision_detected:
        status = "POSSIBLE INCIDENT"
    elif risk >= 40:
        status = "WARNING"
    else:
        status = "NORMAL"

    # -----------------------------
    # Display
    # -----------------------------
    cv2.putText(
        output,
        f"AEGIS-X",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Vehicles: {vehicle_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Stopped: {stopped_count}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Risk: {risk}/100",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Severity: {severity}",
        (20, 160),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Status: {status}",
        (20, 195),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2
    )

    # Show first reason
    if reasons:
        cv2.putText(
            output,
            f"Reason: {reasons[0]}",
            (20, 230),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    cv2.imshow("AEGIS-X - Live Incident Intelligence", output)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()