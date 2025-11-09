import cv2
import matplotlib.pyplot as plt

def show_image_with_boxes(image_path, boxes):
    image = cv2.imread(image_path)
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0,255,0), 2)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
