from models import MODEL_REGISTRY
from .dinov2 import DINOv2
from .dinov2_grayscale import DINOv2Grayscale

MODEL_REGISTRY["dinov2_base"] = {
    "class": DINOv2, "model_id_mapping": "DINOV2-BASE"}
MODEL_REGISTRY["dinov2_large"] = {
    "class": DINOv2, "model_id_mapping": "DINOV2-LARGE"}
MODEL_REGISTRY["dinov2_giant"] = {
    "class": DINOv2, "model_id_mapping": "DINOV2-GIANT"}
MODEL_REGISTRY["dinov2_large_sub"] = {
    "class": DINOv2, "model_id_mapping": "DINOV2-SUBLARGE"}

MODEL_REGISTRY["dinov2_base_grayscale"] = {
    "class": DINOv2Grayscale, "model_id_mapping": "DINOV2-BASE"}
