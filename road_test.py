import cv2
from ultralytics import YOLO

# Load YOLO26n
model = YOLO("yolo26n.pt")

# Road video
video_path = "demo/road.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Could not open demo/road.mp4")

VEHICLES = {"car", "truck", "bus", "motorcycle"}

while True:
    ret, frame = cap.read()

    if not ret:
        print("Video finished.")
        break

    # YOLO detection + ByteTrack
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.35,
        verbose=False
    )

    result = results[0]
    annotated = result.plot()

    # Count vehicles
    vehicle_count = 0

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            if class_name in VEHICLES:
                vehicle_count += 1

    # AEGIS-X status
    if vehicle_count >= 4:
        status = "TRAFFIC BUILD-UP"
    else:
        status = "NORMAL"

    # Display information
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
        f"AEGIS-X: {status}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow("AEGIS-X - Road Intelligence", annotated)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()