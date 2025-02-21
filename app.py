import gradio as gr
from yolo_agent import video_detection_tool
import os
import time

def detect_objects(video):
    """Handles video upload and runs YOLO detection, displaying detections in real-time."""
    result = video_detection_tool.invoke(video, conf=0.8)  # Explicitly setting confidence threshold
    detected_images = "detections"  # Folder where detected images are stored

    
    image_paths = []
    if os.path.exists(detected_images):
        for _ in range(20):  # Limit the loop to avoid infinite execution
            new_images = sorted(
                [os.path.join(detected_images, img) for img in os.listdir(detected_images) if img.endswith(".jpg")],
                key=os.path.getmtime  # Sort images by modification time for real-time order
            )
            if new_images != image_paths:
                image_paths = new_images
                yield result, image_paths
            time.sleep(1)  # Update images in real-time
    
    return result, []

# Gradio Interface
demo = gr.Blocks()
with demo:
    gr.Markdown("# 🎥 YOLO Object Detection with LangChain - Real-time Display")
    video_input = gr.File(label="📤 Upload a Video", type="filepath")
    output_text = gr.Textbox(label="📄 Detection Results")
    output_gallery = gr.Gallery(label="📸 Detected Objects", show_label=True, interactive=False, columns=4)
    detect_button = gr.Button("🚀 Run Detection")
    detect_button.click(fn=detect_objects, inputs=video_input, outputs=[output_text, output_gallery])  # Removed `live=True`

demo.launch(share=True)
