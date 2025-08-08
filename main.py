import torch
from ultralytics import YOLO
import cv2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
model = YOLO('yolov8l.pt').to(device)
def detect_objects_from_video(video_source=0):
    cap = cv2.VideoCapture(video_source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
        frame_resized = cv2.resize(frame, (640, 640))
        frame_gpu = torch.from_numpy(frame_resized).permute(2, 0, 1).unsqueeze(0).float().to(device)
        results = model(frame_gpu)
        annotated_frame = results[0].plot()
        cv2.imshow("Object detaction", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
detect_objects_from_video(0)