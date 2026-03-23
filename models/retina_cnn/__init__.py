from models import MODEL_REGISTRY
from .retina_cnn import RetinaCNN

MODEL_REGISTRY["retina_cnn"] = {"class": RetinaCNN, "model_id_mapping": "CNN"}
MODEL_REGISTRY["retina_ln"] = {"class": RetinaCNN, "model_id_mapping": "LN"}
