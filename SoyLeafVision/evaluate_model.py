from ultralytics import YOLO

def evaluate():
    model = YOLO("runs/detect/soyleaf_swin/weights/best.pt")
    results = model.val()
    print("✅ Evaluation Complete")
    print(results)

if __name__ == "__main__":
    evaluate()
