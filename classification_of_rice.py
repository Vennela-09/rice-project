import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_classification(ratio):
    ratio = round(ratio, 1)
    if ratio >= 3:
        return "(Slender)"
    elif 2.1 <= ratio < 3:
        return "(Medium)"
    elif 1.1 <= ratio < 2.1:
        return "(Bold)"
    elif ratio <= 1:
        return "(Round)"
    return "(Unknown)"

print("Starting")

# Initialize the camera (0 for the default camera)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture a single frame from the camera
ret, frame = cap.read()

# Release the camera immediately after capturing the image
cap.release()

# Check if the frame was captured successfully
if not ret:
    print("Error: Could not read frame.")
    exit()

# Convert the captured frame to grayscale
img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Convert the grayscale image to binary
ret, binary = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)

# Apply an averaging filter
kernel = np.ones((5, 5), np.float32) / 25
dst = cv2.filter2D(binary, -1, kernel)

# Structuring element for morphological operations
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Perform erosion
erosion = cv2.erode(dst, kernel2, iterations=1)

# Perform dilation
dilation = cv2.dilate(erosion, kernel2, iterations=1)

# Detect edges
edges = cv2.Canny(dilation, 100, 200)

# Find contours
contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("No. of rice grains =", len(contours))
total_ar = 0
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    if aspect_ratio < 1:
        aspect_ratio = 1 / aspect_ratio
    print(round(aspect_ratio, 2), get_classification(aspect_ratio))
    total_ar += aspect_ratio

if len(contours) > 0:
    avg_ar = total_ar / len(contours)
    print("Average Aspect Ratio =", round(avg_ar, 2), get_classification(avg_ar))
else:
    print("No contours found")

# Plot the images
imgs_row = 2
imgs_col = 3
plt.subplot(imgs_row, imgs_col, 1), plt.imshow(img, cmap='gray')
plt.title("Original image")

plt.subplot(imgs_row, imgs_col, 2), plt.imshow(binary, cmap='gray')
plt.title("Binary image")

plt.subplot(imgs_row, imgs_col, 3), plt.imshow(dst, cmap='gray')
plt.title("Filtered image")

plt.subplot(imgs_row, imgs_col, 4), plt.imshow(erosion, cmap='gray')
plt.title("Eroded image")

plt.subplot(imgs_row, imgs_col, 5), plt.imshow(dilation, cmap='gray')
plt.title("Dilated image")

plt.subplot(imgs_row, imgs_col, 6), plt.imshow(edges, cmap='gray')
plt.title("Edge detect")

plt.show()
