import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO


WANTED_CLASSES = {"car", "truck", "bus", "motorcycle", "person"}
BOX_COLOUR     = (0, 200, 255)   # BGR yellow-orange
FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE     = 0.5
THICKNESS      = 2
CONF_THRESHOLD = 0.40


def run_detection(video_path: str, output_path: str, model_path: str = "models/yolov8n.pt"):
    """
    Run YOLOv8 on a video file, draw boxes for WANTED_CLASSES,
    write annotated output video.
    """
    model = YOLO(model_path)   # downloads yolov8n.pt on first run if not found
    cap   = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_count = 0
    print(f"[ARGUS] Detecting on {video_path} ...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model(frame, verbose=False, conf=CONF_THRESHOLD)[0]

        for box in results.boxes:
            cls_name = model.names[int(box.cls[0])]
            if cls_name not in WANTED_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            label = f"{cls_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOUR, THICKNESS)
            cv2.putText(frame, label, (x1, y1 - 6), FONT, FONT_SCALE, BOX_COLOUR, THICKNESS)

        cv2.putText(frame, f"Frame {frame_count}", (10, 30), FONT, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        frame_count += 1

        cv2.imshow("ARGUS Phase 1", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[ARGUS] Done. {frame_count} frames written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARGUS Phase 1 - Detection")
    parser.add_argument("--video",  required=True,               help="Input video path")
    parser.add_argument("--output", default="demos/phase1.mp4",  help="Output video path")
    parser.add_argument("--model",  default="models/yolov8n.pt", help="YOLO model path")
    args = parser.parse_args()
    run_detection(args.video, args.output, args.model)