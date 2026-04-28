import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


SEED = 42
AUTOTUNE = tf.data.AUTOTUNE


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate waste classification model.")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--class-map", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--metrics-out", type=str, default="reports/metrics.json")
    parser.add_argument("--cm-out", type=str, default="reports/confusion_matrix.png")
    return parser.parse_args()


def load_test_set(data_dir, img_size, batch_size, val_split, test_split):
    val_plus_test = val_split + test_split
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
    val_ratio_inside_holdout = val_split / (val_split + test_split)
    val_batches = int(np.floor(holdout_batches * val_ratio_inside_holdout))
    val_batches = max(1, min(val_batches, holdout_batches - 1))
    test_ds = holdout_ds.skip(val_batches).prefetch(AUTOTUNE)
    return test_ds


def save_confusion_matrix(cm, class_names, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(SEED)

    data_dir = Path(args.data_dir)
    model_path = Path(args.model_path)
    class_map_path = Path(args.class_map)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not class_map_path.exists():
        raise FileNotFoundError(f"Class map not found: {class_map_path}")

    class_names = json.loads(class_map_path.read_text(encoding="utf-8"))
    test_ds = load_test_set(
        data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
    )

    model = tf.keras.models.load_model(model_path)

    y_true = []
    y_pred = []
    for images, labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": report,
    }

    metrics_out = Path(args.metrics_out)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    save_confusion_matrix(cm, class_names, Path(args.cm_out))

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics: {metrics_out}")
    print(f"Saved confusion matrix: {args.cm_out}")


if __name__ == "__main__":
    main()
