People Counting YOLO

A real-time People Counting System using YOLO and OpenCV that detects and counts people in both images and video files. The project uses YOLO object detection to identify people and OpenCV to process, display, and analyze visual data.

Features
👤 Detect people in images using YOLO.
🎥 Detect and count people in video files.
📊 Display the total number of detected people.
⚡ Real-time video frame processing with OpenCV.
🖼️ Supports image-based detection.
🔍 Uses YOLO's pre-trained COCO model for object detection.
📦 Simple Python implementation that is easy to understand and extend.

Technologies Used
Python
YOLO
OpenCV
NumPy
COCO Dataset

How It Works
Load the YOLO configuration, weights, and COCO class names.
Read an image or video frame using OpenCV.
Pass the frame to the YOLO model for object detection.
Filter detections for the person class.
Draw bounding boxes around detected people.
Display the total number of people detected.

Use Cases
This project can be extended for:
Crowd monitoring
Retail store analytics
Office occupancy monitoring
Security and surveillance
Traffic and public-area monitoring
Smart-city applications
Real-time people analytics

The application will display the processed image/video with bounding boxes around detected people and the current people count.

Future Improvements
Add object tracking using DeepSORT or ByteTrack.
Improve counting accuracy in crowded environments.
Add entry/exit line-based counting.
Support live webcam and CCTV streams.
Add people-counting analytics and reports.
Build a web dashboard using Flask or FastAPI.
