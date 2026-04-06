from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

video_path = r"C:\Users\asus\OneDrive\Desktop\DL Project\13731740_1080_2448_60fps.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Cannot open video")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    "output.mp4",
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Detection", 800, 600)

print("Processing video...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.3)

    annotated_frame = results[0].plot()

    annotated_frame = cv2.resize(annotated_frame, (width, height))

    out.write(annotated_frame)

    cv2.imshow("Detection", annotated_frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("Done! Check output.mp4")