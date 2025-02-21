"""
LangChain YOLO Agent
---------------------

This project provides a YOLO-based object detection tool integrated with LangChain.
Users can upload any video to analyze its contents, generate object detection logs,
and visualize detections with bounding boxes.

Steps:
1) Install dependencies: `pip install langchain openai ultralytics opencv-python`
2) Add this file (`yolo_agent.py`) to your project.
3) Ensure that the YOLO model file (`last.pt`) is available in the working directory.
4) Use the provided functions to analyze uploaded videos dynamically.
"""

import os
import cv2
import shutil
from langchain.agents import Tool, tool
from ultralytics import YOLO

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Update the model path to point to the "yolo" directory
MODEL_PATH = os.path.join(os.path.dirname(__file__), "last.pt")


def detect_with_yolo(
    video_path: str,
    model_path: str = os.path.join(os.path.dirname(__file__), "last.pt"),
    output_dir: str = "detections",
    frame_skip: int = 10,
    conf: float = 0.7
) -> str:
    """
    Runs YOLO detection on the given video.
    - Detects only class_id 0..5 (Danger / Handgun / Knife, etc.)
    - Draws red bounding boxes
    - Saves logs to a text file
    - Saves detected frames as images
    """

    if not os.path.exists(video_path):
        return f"Video not found: {video_path}"

    try:
        model = YOLO(model_path)
    except Exception as e:
        return f"Failed to load model: {e}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return f"Cannot open video: {video_path}"

    os.makedirs(output_dir, exist_ok=True)
    output_txt = os.path.join(output_dir, "detections.txt")
    frame_count = 0

    with open(output_txt, "w") as ftxt:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.8)
            detections = (
                results[0].boxes.data.cpu().numpy() if len(results) > 0 else []
            )

            valid_detections = [det for det in detections if int(det[5]) in [0, 1, 2, 3, 4, 5]]

            if len(valid_detections) > 0:
                for det in valid_detections:
                    x1, y1, x2, y2, conf, cls_ = det
                    class_id = int(cls_)

                    if class_id in [0, 1, 2]:
                        class_label = "Danger"
                    elif class_id in [3, 4, 5]:
                        class_label = model.names.get(class_id, f"Class {class_id}")
                    else:
                        class_label = f"Class {class_id}"

                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 0, 255),
                        3,
                    )

                    (w, h), _ = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_COMPLEX, 0.8, 2)
                    label_x1 = int(x1)
                    label_y2 = int(y1)
                    label_y1 = label_y2 - h - 10
                    label_x2 = label_x1 + w + 10

                    cv2.rectangle(
                        frame,
                        (label_x1, label_y1),
                        (label_x2, label_y2),
                        (0, 0, 255),
                        cv2.FILLED,
                    )

                    cv2.putText(
                        frame,
                        class_label,
                        (label_x1 + 5, label_y1 + h + 5),
                        cv2.FONT_HERSHEY_COMPLEX,
                        0.85,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    ftxt.write(
                        f"Frame {frame_count}: {class_label} at ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})\n"
                    )

                output_frame_path = os.path.join(output_dir, f"frame_{frame_count}.jpg")
                cv2.imwrite(output_frame_path, frame)

            frame_count += frame_skip
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

    cap.release()
    cv2.destroyAllWindows()

    return f"Processing complete. Outputs saved in '{output_dir}' and '{output_txt}'."

@tool("video_detection_tool", return_direct=True)
def video_detection_tool(video) -> str:
    """
    Handles video uploads dynamically and runs YOLO detection.
    Saves results in the "detections" folder and logs to detections.txt.
    """
    video_path = os.path.join(UPLOAD_FOLDER, os.path.basename(video.name))
    # Update the model path used here too
    MODEL_PATH = os.path.join(os.path.dirname(__file__), "last.pt")

    shutil.copy(video.name, video_path)
    try:
        result = detect_with_yolo(video_path)
    except Exception as e:
        return f"Error during detection: {e}"

    return result

if __name__ == "__main__":
    print("LangChain YOLO Agent Ready!")

__all__ = ["video_detection_tool"]
