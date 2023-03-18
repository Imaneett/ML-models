import cv2
from matplotlib
import pyplot

# Load the image and display it
img = cv2.imread("src/img/test.jpg")
cv2.imshow('image', img)
cv2.waitKey(0)

# Convert the image to RGB format for better object detection
tst_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load the Haar classifier for object detection
my_detector = cv2.CascadeClassifier("myClassifier.xml")

# Detect objects in the image
detected_img = my_detector.detectMultiScale(tst_converted, minSize=(20, 20))

# Draw rectangles around the detected objects
if len(detected_img) != 0:  # Check if any objects were detected
    for (x, y, w, h) in detected_img:
        cv2.rectangle(tst_converted, (x, y), (x + w, y + h), (0, 255, 0), 5)

# Display the final image with the detected objects
pyplot.subplot(1, 1, 1)
pyplot.imshow(tst_converted)
pyplot.show()

# Define exponential of a list of numbers
exponential = [math.exp(-x) for x in [2, 3, 0.0497871, 0.135335]]

