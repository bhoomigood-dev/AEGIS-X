import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture("demo/road.mp4")
cap = cv2.VideoCapture("demo/road1.mp4")

if not cap.isOpened():
    raise RuntimeError("Could not open road.mp4")

VEHICLES = {"car", "truck", "bus", "motorcycle"}

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
    output = result.plot()

    vehicle_count = 0
    person_count = 0

    if result.boxes is not None:
        for box in result.boxes:
            name = result.names[int(box.cls[0])]

            if name in VEHICLES:
                vehicle_count += 1

            elif name == "person":
                person_count += 1

    # Simple deterministic risk score
    risk = 0

    if vehicle_count >= 2:
        risk += 20

    if vehicle_count >= 4:
        risk += 25

    if vehicle_count >= 6:
        risk += 20

    if person_count >= 3:
        risk += 10

    risk = min(risk, 100)

    if risk >= 70:
        level = "HIGH"
    elif risk >= 40:
        level = "MEDIUM"
    else:
        level = "LOW"

    cv2.putText(
        output,
        f"Vehicles: {vehicle_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"Risk Score: {risk}/100",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"AEGIS-X Risk: {level}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("AEGIS-X - Risk Intelligence", output)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()