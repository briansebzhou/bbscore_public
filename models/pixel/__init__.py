from models import MODEL_REGISTRY
from .pixel import PIXEL, GrayscalePixel

MODEL_REGISTRY["pixel"] = {
    "class": PIXEL, "model_id_mapping": "PIXEL"}
MODEL_REGISTRY["grayscale_pixel"] = {
    "class": GrayscalePixel, "model_id_mapping": "identity_layer"}
