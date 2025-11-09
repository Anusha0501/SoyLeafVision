from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import cv2
import numpy as np

def generate_gradcam(model, image_path, target_layer, output_path="outputs/gradcam.jpg"):
    model.eval()
    image = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    input_tensor = torch.tensor(rgb_img.transpose(2, 0, 1)).unsqueeze(0).float()

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    cv2.imwrite(output_path, cam_image)
    print(f"✅ GradCAM saved at {output_path}")
