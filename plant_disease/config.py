import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(
    os.environ.get(
        "PLANT_DISEASE_DATA_DIR",
        PROJECT_ROOT / "data" / "plantvillage dataset" / "color",
    )
)
ARTIFACTS_DIR = Path(
    os.environ.get("PLANT_DISEASE_ARTIFACTS_DIR", PROJECT_ROOT / "artifacts")
)
MODEL_PATH = ARTIFACTS_DIR / "plant_disease_prediction_model.keras"
CLASS_INDICES_PATH = ARTIFACTS_DIR / "class_indices.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
DEFAULT_EPOCHS = 5
MIN_PREDICTION_CONFIDENCE = 0.75
