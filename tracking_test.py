import cv2
from ultralytics import YOLO

# Load YOLO26n
model = YOLO("yolo26n.pt")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detection + ByteTrack tracking
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    # Draw boxes and tracking IDs
    annotated = results[0].plot()

    # Count detected objects
    counts = {}

    if results[0].boxes is not None:
        for cls in results[0].boxes.cls:
            name = results[0].names[int(cls)]
            counts[name] = counts.get(name, 0) + 1

    # Display counts
    y = 30
    for name, count in counts.items():
        cv2.putText(
            annotated,
            f"{name}: {count}",
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        y += 30

    cv2.imshow("AEGIS-X - AI Tracking", annotated)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()