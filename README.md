# Gun Detection Agent

This project integrates a YOLO object detection model into a LangChain tool. It allows you to process videos (either file-based or real-time), detect objects, and generate both image and text outputs.

## 📌 Features
- Detects objects using YOLO (class IDs: 0-5)
- Draws **red bounding boxes** and labels (e.g. "Handgun", "Knife")
- Saves detected frames as images in the `detections` directory
- Logs detection results in `detections/detections.txt`
- Can be used as a **LangChain Tool** for easy integration into other projects

## 🚀 Installation
### 1️⃣ Install dependencies:
```bash
pip install -r requirements.txt
```
### 2️⃣ Set up the project:
```bash
git clone https://github.com/your-repository/langchain-yolo-agent.git
cd langchain-yolo-agent
```
### 3️⃣ Run the detection script:
```bash
python langchain_yolo_agent.py
```

## 🔧 Usage
### As a Standalone Script
```python
from langchain_yolo_agent import video_detection_tool

response = video_detection_tool("path/to/video.mp4")
print(response)
```
### As a LangChain Tool
```python
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain_yolo_agent import video_detection_tool

llm = OpenAI(temperature=0)
tools = [video_detection_tool]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

user_input = "Analyze the objects in the video: path/to/video.mp4"
response = agent.run(user_input)
print(response)
```

## 📂 Output Files
- **Detected frames**: Saved in `detections/` (e.g., `detections/frame_1240.jpg`)
- **Detection log**: `detections/detections.txt` (text-based log of detected objects)

## 🔥 Future Improvements
- Add support for real-time video detection
- Deploy as an API for remote processing

## 📝 License
This project is open-source and available under the MIT license.
