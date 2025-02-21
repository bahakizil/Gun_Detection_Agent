from setuptools import setup, find_packages

setup(
    name="yolo_agent",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "openai",
        "ultralytics",
        "opencv-python",
        "opencv-contrib-python"
    ],
    include_package_data=True,
    package_data={
        "": ["best.pt"]  # Ensure the YOLO model file is included in the package
    },
)
