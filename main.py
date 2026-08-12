import cv2
import numpy as np
import os

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")

layer_names = net.getLayerNames()
output_layers = [
    layer_names[i - 1]
    for i in net.getUnconnectedOutLayers()
]

input_path = input("Enter file path : ")

image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
video_extensions = [
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".flv"
]
extension = os.path.splitext(input_path)[1].lower()

cv2.namedWindow("People Counter", cv2.WINDOW_NORMAL)
cv2.resizeWindow("People Counter", 800, 600)

def detect_people(frame):

    height, width = frame.shape[:2]

    # Convert image/frame to blob
    blob = cv2.dnn.blobFromImage(
        frame,
        1 / 255.0,
        (416, 416),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)

    outputs = net.forward(output_layers)

    boxes = []
    confidences = []

    # Detect persons
    for output in outputs:

        for detection in output:

            scores = detection[5:]

            class_id = np.argmax(scores)

            confidence = scores[class_id]

            # COCO class ID 0 = person
            if class_id == 0 and confidence > 0.5:

                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)

                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Remove duplicate boxes
    indexes = cv2.dnn.NMSBoxes(
        boxes,
        confidences,
        0.5,
        0.4
    )

    people_count = 0

    if len(indexes) > 0:

        for i in indexes.flatten():

            x, y, w, h = boxes[i]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Draw Person label
            cv2.putText(
                frame,
                "Person",
                (x, max(y - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )

            people_count += 1

    # Draw people count
    cv2.putText(
        frame,
        f"People Count: {people_count}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 0, 255),
        2
    )

    return frame, people_count

if extension in image_extensions:

    frame = cv2.imread(input_path)

    if frame is None:
        print("ERROR: Could not open image")
        exit()

    frame, people_count = detect_people(frame)

    print("People Count:", people_count)

    cv2.imshow("People Counter", frame)

    # Wait until user presses q
    while True:

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

elif extension in video_extensions:

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print("ERROR: Could not open video")
        exit()

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Video ended")
            break

        frame, people_count = detect_people(frame)

        cv2.imshow("People Counter", frame)

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()

else:
    print("Unsupported file format")

cv2.destroyAllWindows()