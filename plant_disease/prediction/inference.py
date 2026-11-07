import json
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image
from tensorflow.keras import Model, models

from plant_disease.config import CLASS_INDICES_PATH, IMAGE_SIZE, MODEL_PATH


def load_prediction_assets(
    model_path: Path = MODEL_PATH,
    class_indices_path: Path = CLASS_INDICES_PATH,
) -> tuple[Model, dict[int, str]]:
    model_path = Path(model_path)
    class_indices_path = Path(class_indices_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model not found: {model_path}")
    if not class_indices_path.is_file():
        raise FileNotFoundError(f"Class mapping not found: {class_indices_path}")

    with class_indices_path.open(encoding="utf-8") as file:
        raw_indices = json.load(file)
    try:
        class_indices = {int(index): name for index, name in raw_indices.items()}
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError(
            f"Invalid class mapping in {class_indices_path}; expected an index-to-name object."
        ) from error
    if not class_indices or not all(isinstance(name, str) for name in class_indices.values()):
        raise ValueError(f"Class mapping is empty or invalid: {class_indices_path}")

    return models.load_model(model_path), class_indices


def preprocess_image(
    image: Image.Image,
    target_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    resized_image = image.convert("RGB").resize(target_size)
    image_array = np.asarray(resized_image, dtype=np.float32) / 255.0
    return np.expand_dims(image_array, axis=0)


def predict_image(
    model: Model,
    image: Image.Image,
    class_indices: Mapping[int, str],
) -> tuple[str, float]:
    predictions = model.predict(preprocess_image(image), verbose=0)
    if predictions.ndim != 2 or predictions.shape[0] != 1 or predictions.shape[1] == 0:
        raise ValueError("The model returned predictions in an unexpected shape.")

    class_index = int(np.argmax(predictions[0]))
    if class_index not in class_indices:
        raise ValueError(f"Predicted class index {class_index} is missing from the class mapping.")
    return class_indices[class_index], float(np.max(predictions[0]))
