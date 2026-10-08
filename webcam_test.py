import cv2
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permission.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read webcam frame.")
        break

    results = model(frame, conf=0.35, verbose=False)
    frame = results[0].plot()

    cv2.imshow("AEGIS-X - Live AI Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()