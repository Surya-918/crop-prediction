from tensorflow.keras import Model, layers, models


def build_model(num_classes: int, image_size: tuple[int, int] = (224, 224)) -> Model:
    if num_classes < 1:
        raise ValueError("num_classes must be at least 1.")

    model = models.Sequential(
        [
            layers.Input(shape=(*image_size, 3)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(256, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
