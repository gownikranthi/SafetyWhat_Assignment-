# Object and Sub-Object Detection with YOLOv5

This project implements object detection and sub-object detection using the YOLOv5 deep learning model. It processes video frames to detect objects such as "person" and its associated sub-objects like "chair," "flower pot," etc. Detected objects and sub-objects are saved with bounding boxes, and the object hierarchy is exported as a JSON file for further analysis.

## Features

- **YOLOv5 Object Detection**: Detects primary objects and their associated sub-objects.
- **Bounding Box Visualization**: Draws bounding boxes around detected objects and sub-objects.
- **Export to JSON**: Saves the detected object hierarchy (with primary and sub-objects) to a JSON file.
- **Cropped Image Saving**: Saves cropped images of detected sub-objects to disk.
- **FPS Display**: Displays the frames per second (FPS) of the video during processing.

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- YOLOv5 (from Ultralytics)
- JSON

### Install dependencies

You can install the required libraries using 
pip install torch torchvision torchaudio
pip install opencv-python-headless
pip install matplotlib
Setup
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/object-subobject-detection.git
cd object-subobject-detection
Download the YOLOv5 model: The script automatically loads the YOLOv5 model from the Ultralytics repository.

Prepare your video file: Make sure you have a video file (e.g., .mp4) for object detection. Replace the path in the code where it reads:

python
Copy code
video_path = r"C:\path\to\your\video.mp4"
Create config.json (Optional): If you want to define specific sub-objects to detect (like "chair," "bowl," "flower pot," etc.), create a config.json file in the following format:

json
Copy code
{
    "objects": {
        "person": ["chair", "bowl", "flower pot", "tv", "suitcase"]
    }
}
If the file is not found, a default list will be used.

Usage
Run the detection script: Run the following command to start object and sub-object detection on the video:

bash
Copy code
python detect_objects.py
View the results:

The bounding boxes for detected objects and sub-objects will be displayed in real-time on the video feed.
Sub-objects are cropped and saved as individual images (subobject_{id}.jpg).
The object hierarchy, including parent and child objects, will be saved in a JSON file (detection_results.json).
Quit the detection process: Press q to stop the video processing and exit.

Example Output
detection_results.json: Contains the hierarchical data for detected objects and sub-objects.
Cropped images for each detected sub-object saved as subobject_{id}.jpg.
json
Copy code
{
    "1": {
        "label": "person",
        "id": 1,
        "bbox": [100, 150, 200, 300],
        "sub_objects": [
            {
                "label": "chair",
                "id": 2,
                "bbox": [120, 160, 180, 240]
            },
            {
                "label": "bowl",
                "id": 3,
                "bbox": [130, 170, 160, 210]
            }
        ]
    }
}
Notes
The script uses the default YOLOv5 small model (yolov5s). You can switch to a faster model like yolov5n for better performance, or use a more accurate model like yolov5m or yolov5l for improved detection accuracy.
Ensure that your video file is accessible by the script and that the path is correctly set.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
The project utilizes the YOLOv5 model from Ultralytics.
OpenCV is used for image processing and video handling.
PyTorch is used for deep learning tasks.
markdown
Copy code

### Explanation of the Sections

- **Project Overview**: Describes the functionality of the project.
- **Requirements**: Lists the libraries and dependencies needed to run the project.
- **Setup**: Explains how to clone the repository, install dependencies, and configure the project.
- **Usage**: Guides users on how to run the script and interact with the detection process.
- **Example Output**: Shows what the expected output will look like, including a sample JSON structure for object hierarchy.
- **License**: Placeholder for project license details.
- **Acknowledgments**: Credits to the libraries and frameworks used in the project.
