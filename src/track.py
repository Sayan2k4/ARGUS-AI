import cv2
import argparse
import numpy as np
from collections import defaultdict
from pathlib import Path
from ultralytics import YOLO

WANTED_CLASSES = {"car", "truck", "bus", "motorcycle", "person"}
CONF_THRESHOLD = 0.40
TRAIL_LENGTH   = 30     
FONT           = cv2.FONT_HERSHEY_SIMPLEX


def track_colour(track_id: int):
    rng = np.random.default_rng(track_id * 137)
    return tuple(int(x) for x in rng.integers(80, 255, 3))


def run_tracking(video_path: str, output_path: str, model_path: str = "models/yolov8n.pt"):
   
    model = YOLO(model_path)
    cap   = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    track_history: dict[int, list] = defaultdict(list)
    track_class:   dict[int, str]  = {}

    frame_count = 0
    print(f"[ARGUS] Tracking on {video_path} ...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",   # built-in Ultralytics config
            conf=CONF_THRESHOLD,
            verbose=False,
        )[0]

        if results.boxes.id is not None:
            for box in results.boxes:
                if box.id is None:
                    continue

                track_id = int(box.id[0])
                cls_name = model.names[int(box.cls[0])]

                if cls_name not in WANTED_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                history = track_history[track_id]
                history.append((cx, cy))
                if len(history) > TRAIL_LENGTH:
                    history.pop(0)
                track_class[track_id] = cls_name

                colour = track_colour(track_id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                label = f"{cls_name} #{track_id}"
                cv2.putText(frame, label, (x1, y1 - 7), FONT, 0.45, colour, 2)

                pts = np.array(history, dtype=np.int32)
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    c = tuple(int(v * alpha) for v in colour)
                    cv2.line(frame, tuple(pts[i - 1]), tuple(pts[i]), c, 2)

        cv2.putText(frame, f"Frame {frame_count} | Tracks: {len(track_history)}",
                    (10, 30), FONT, 0.7, (255, 255, 255), 2)
        writer.write(frame)
        frame_count += 1

        cv2.imshow("ARGUS Phase 2", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[ARGUS] Tracking done. Total unique tracks: {len(track_history)}")
    return track_history, track_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARGUS Phase 2 - Tracking")
    parser.add_argument("--video",  required=True)
    parser.add_argument("--output", default="demos/phase2.mp4")
    parser.add_argument("--model",  default="models/yolov8n.pt")
    args = parser.parse_args()
    run_tracking(args.video, args.output, args.model)