
from ultralytics import YOLO
import cv2

# Load the pretrained model
model = YOLO("yolo11n.pt")

# Detect objects in the image
results = model.predict(
    source="bus.jpg",
    conf=0.25
)

# Save the image with detection boxes
annotated_image = results[0].plot()
cv2.imwrite("detected_bus.jpg", annotated_image)

print("Object detection completed successfully!")
print("Open detected_bus.jpg to see the results.")
