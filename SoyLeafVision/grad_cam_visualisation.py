import torch
from models.model_utils import SoyLeafModel
from utils.explainability import generate_gradcam

model = SoyLeafModel(num_classes=6)
target_layer = model.backbone.backbone[-1]  # final Swin feature map

generate_gradcam(
    model=model,
    image_path="sample_leaf.jpg",
    target_layer=target_layer,
    output_path="outputs/gradcam_result.jpg"
)
