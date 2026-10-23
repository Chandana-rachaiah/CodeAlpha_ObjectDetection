from ultralytics import YOLO

# Load the pretrained YOLO model
model = YOLO("yolo11n.pt")

# Track objects in a video
results = model.track(
    source="sample.mp4",
    show=True,
    save=True,
    conf=0.25,
    tracker="bytetrack.yaml"
)

print("Video object tracking completed successfully!")
