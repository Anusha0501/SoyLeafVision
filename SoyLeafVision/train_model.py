from ultralytics import YOLO

def train_yolo():
    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        name="soyleaf_swin",
        pretrained=True
    )

if __name__ == "__main__":
    train_yolo()
