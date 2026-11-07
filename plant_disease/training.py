import argparse
import json
from pathlib import Path

from plant_disease.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    DATA_DIR,
    DEFAULT_EPOCHS,
    IMAGE_SIZE,
    MODEL_PATH,
)
from plant_disease.data.loader import create_data_generators
from plant_disease.models.cnn import build_model


def train_model(
    data_dir: Path = DATA_DIR,
    artifacts_dir: Path = ARTIFACTS_DIR,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = BATCH_SIZE,
):
    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    train_generator, validation_generator = create_data_generators(
        data_dir=data_dir,
        batch_size=batch_size,
    )
    model = build_model(
        num_classes=train_generator.num_classes,
        image_size=IMAGE_SIZE,
    )
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
    )
    model.evaluate(validation_generator, verbose=1)

    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model.save(artifacts_dir / MODEL_PATH.name)
    index_to_class = {
        str(index): class_name
        for class_name, index in train_generator.class_indices.items()
    }
    with (artifacts_dir / "class_indices.json").open("w", encoding="utf-8") as file:
        json.dump(index_to_class, file, indent=2)
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the plant disease classifier.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--artifacts-dir", type=Path, default=ARTIFACTS_DIR)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        artifacts_dir=args.artifacts_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
