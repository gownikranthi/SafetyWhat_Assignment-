import torch
import cv2
import numpy as np
import json
import time

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Change to 'yolov5n' for faster inference

# Video path
video_path = r"C:\Users\Lenovo\Desktop\SaftyWhatAssessment\WhatsApp Video 2025-01-10 at 19.49.01_bc3eb6d9.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Object hierarchy and configuration
object_hierarchy = {}
object_id = 0

# Load configuration for sub-object types
try:
    with open('config.json') as config_file:
        config = json.load(config_file)
    subobject_list = config.get("objects", {}).get("person", [])
except FileNotFoundError:
    print("Warning: 'config.json' not found. Using default sub-object list.")
    subobject_list = ['chair', 'bowl', 'flower pot', 'tv', 'suitcase']

# Function to save cropped images
def save_cropped_image(frame, bbox, save_path):
    x1, y1, x2, y2 = bbox
    cropped_image = frame[max(0, y1):y2, max(0, x1):x2]
    if cropped_image.size > 0:
        cv2.imwrite(save_path, cropped_image)
        print(f"Cropped image saved: {save_path}")
    else:
        print(f"Error: Empty cropped image for {save_path}")

# Function to export object hierarchy to JSON
def export_to_json(object_hierarchy, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(object_hierarchy, json_file, indent=4)
    print(f"Detection results saved to {output_path}")

# Processing frames
start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame = cv2.resize(frame, (640, 480))  # Resize for faster inference
    results = model(frame)
    primary_boxes = results.xyxy[0]  # Get bounding boxes (x1, y1, x2, y2, confidence, class)

    for box in primary_boxes:
       x1, y1, x2, y2 = map(int, box[:4].tolist())
       conf = box[4].item()
       cls = int(box[5].item())

    primary_label = results.names[cls]

    if primary_label == 'person':
            object_id += 1
            object_hierarchy[object_id] = {
                "label": primary_label,
                "id": object_id,
                "bbox": [x1, y1, x2, y2],
                "sub_objects": []
            }

            cropped_region = frame[y1:y2, x1:x2]

            if cropped_region.size > 0:
                sub_results = model(cropped_region)
                sub_boxes = sub_results.xyxy[0]

                for sub_box in sub_boxes:
                    sub_x1, sub_y1, sub_x2, sub_y2 = map(int, sub_box[:4].tolist())
                    sub_conf = sub_box[4].item()
                    sub_cls = int(sub_box[5].item())

                    sub_label = sub_results.names[sub_cls]

                    if sub_label in subobject_list:
                        sub_object_id = object_id + 1
                        sub_bbox = [sub_x1 + x1, sub_y1 + y1, sub_x2 + x1, sub_y2 + y1]

                        object_hierarchy[object_id]["sub_objects"].append({
                            "label": sub_label,
                            "id": sub_object_id,
                            "bbox": sub_bbox
                        })

                        save_cropped_image(frame, sub_bbox, f"subobject_{sub_object_id}.jpg")

                        cv2.rectangle(frame, (sub_bbox[0], sub_bbox[1]), (sub_bbox[2], sub_bbox[3]), (0, 255, 0), 2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('Object and Sub-Object Detection', frame)

    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    print(f"FPS: {fps:.2f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Export object hierarchy to JSON
export_to_json(object_hierarchy, 'detection_results.json')

cap.release()
cv2.destroyAllWindows()
