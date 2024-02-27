import cv2
import numpy as np
import tensorflow as tf
import pytesseract

# Load the EAST text detection model
net = cv2.dnn.readNet('frozen_east_text_detection.pb')

# Function to perform text detection using the EAST model
def detect_text(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)
    orig = image.copy()
    (H, W) = image.shape[:2]

    # Preprocess the image for text detection
    blob = cv2.dnn.blobFromImage(image, 1.0, (320, 320), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"])

    # Decode the predictions and apply non-maxima suppression
    rects, confidences = decode_predictions(scores, geometry)
    boxes, probs = non_max_suppression(np.array(rects), probs=confidences)

    # Initialize an empty list to store detected text
    detected_text = []

    # Loop over the bounding boxes
    for box in boxes:
        # Extract the bounding box coordinates and ensure they are integers
        startX, startY, endX, endY = [int(coord) for coord in box]

        # Ensure the bounding box coordinates are within the image dimensions
        startX = max(0, startX)
        startY = max(0, startY)
        endX = min(W, endX)
        endY = min(H, endY)

        # Extract the region of interest (ROI) containing the text
        roi = orig[startY:endY, startX:endX]

        # Use Tesseract OCR to perform text recognition on the ROI
        text = pytesseract.image_to_string(roi)

        # Append the detected text to the list
        detected_text.append(text)

    return detected_text


# Function to decode predictions from the EAST model
def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):
            if scoresData[x] < 0.5:
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

# Function to apply non-maxima suppression to bounding boxes
def non_max_suppression(boxes, probs=None, overlapThresh=0.3):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last],
                                                np.where(overlap > overlapThresh)[0])))

    if probs is not None:
        pick = [int(i) for i in pick]  # Convert to integers
        return boxes[pick], [probs[i] for i in pick]

    return boxes[pick]

# Function to search for the word 'Software' in the detected text
def search_word(detected_text, target_word):
    for text in detected_text:
        if target_word.lower() in text.lower():
            return True
    return False

# Example usage
image_path = r"C:\Users\seank\Downloads\Software 2024 Album Cover.jpg"
target_word = 'Software'

# Detect text in the image
detected_text = detect_text(image_path)

# Search for the word 'Software' in the detected text
found = search_word(detected_text, target_word)

if found:
    print(f"The word '{target_word}' was found in the image.")
else:
    print(f"The word '{target_word}' was not found in the image.")
