import cv2
import time
import numpy
# Load the image from file
img = cv2.imread('src/img/animals.jpeg')

# Display the image in a window
cv2.imshow('window', img)
cv2.waitKey(1)

# Load the pre-trained YOLOv3 object detection model
net = cv2.dnn.readNetFromDarknet('src/yolov/yolov3.cfg', 'src/yolov/yolov3.weights')

# Set the OpenCV backend for the neural network
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

# Get the names of the output layers in the neural network
ln = net.getLayerNames()
print(len(ln), ln)

# Convert the image to a blob suitable for input to the neural network
blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416), swapRB=True, crop=False)

# Display the blob image in a window
r = blob[0, 0, :, :]
cv2.imshow('blob', r)
text = f'Blob shape={blob.shape}'
cv2.displayOverlay('blob', text)
cv2.waitKey(1)

# Set the input to the neural network to be the blob image
net.setInput(blob)

# Time how long it takes to perform the object detection
t0 = time.time()
outputs = net.forward(ln)
t = time.time()

# Print the time taken to perform the object detection
print(f'Time taken: {t - t0:.3f} seconds')
