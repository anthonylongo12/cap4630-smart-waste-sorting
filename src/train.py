import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(description="Train waste classification model.")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to class folders.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--output-model", type=str, default="models/waste_classifier.keras")
    parser.add_argument("--output-class-map", type=str, default="models/class_names.json")
    return parser.parse_args()


def make_datasets(data_dir, img_size, batch_size, val_split, test_split):
    val_plus_test = val_split + test_split
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_plus_test,
        subset="training",
        seed=SEED,
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )

    holdout_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_plus_test,
        subset="validation",
        seed=SEED,
        label_mode="int",
        image_size=(img_size, img_size),
        batch_size=batch_size,
    )

    holdout_batches = tf.data.experimental.cardinality(holdout_ds).numpy()
    if holdout_batches < 2:
        raise ValueError("Not enough holdout batches. Add more data or reduce batch size.")

    val_ratio_inside_holdout = val_split / (val_split + test_split)
    val_batches = int(np.floor(holdout_batches * val_ratio_inside_holdout))
    val_batches = max(1, min(val_batches, holdout_batches - 1))

    val_ds = holdout_ds.take(val_batches)
    test_ds = holdout_ds.skip(val_batches)

    class_names = train_ds.class_names

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    test_ds = test_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names


def compute_weights(train_ds, num_classes):
    y = []
    for _, labels in train_ds.unbatch():
        y.append(int(labels.numpy()))
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(y))
    return {i: float(w) for i, w in enumerate(weights)}


def build_model(num_classes, img_size, learning_rate):
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    base_model = MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.25)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(SEED)

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    train_ds, val_ds, test_ds, class_names = make_datasets(
        data_dir=data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
    )

    num_classes = len(class_names)
    class_weights = compute_weights(train_ds, num_classes)
    model = build_model(num_classes, args.img_size, args.learning_rate)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Optional quick check on test set so training run provides immediate signal.
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model)

    output_class_map = Path(args.output_class_map)
    output_class_map.parent.mkdir(parents=True, exist_ok=True)
    output_class_map.write_text(json.dumps(class_names, indent=2), encoding="utf-8")

    print(f"Saved model: {output_model}")
    print(f"Saved class map: {output_class_map}")


if __name__ == "__main__":
    main()
