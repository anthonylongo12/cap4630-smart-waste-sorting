import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path("models/waste_classifier.keras")
CLASS_MAP_PATH = Path("models/class_names.json")
IMG_SIZE = 224
LOW_CONFIDENCE_THRESHOLD = 0.60

DISPOSAL_GUIDANCE = {
    "plastic": "Place in plastics recycling bin if clean and dry.",
    "paper": "Place in paper recycling bin if not food-soiled.",
    "glass": "Place in glass recycling container; avoid broken unsafe handling.",
    "metal": "Place in metal recycling bin (cans/foil cleaned).",
    "organic": "Place in compost/organic waste bin.",
    "other": "Place in general waste unless local rules specify otherwise.",
}


@st.cache_resource
def load_model_and_classes():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    if not CLASS_MAP_PATH.exists():
        raise FileNotFoundError(f"Class map not found: {CLASS_MAP_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = json.loads(CLASS_MAP_PATH.read_text(encoding="utf-8"))
    return model, class_names


def preprocess_image(pil_image):
    image = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image).astype(np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)
    return arr


def main():
    st.set_page_config(page_title="Smart Waste Sorting Assistant", layout="centered")
    st.title("Smart Waste Sorting Assistant")
    st.write(
        "Upload an image to classify waste into one of six categories: "
        "plastic, paper, glass, metal, organic, or other."
    )

    try:
        model, class_names = load_model_and_classes()
    except FileNotFoundError as err:
        st.error(str(err))
        st.info("Train the model first using: `python src/train.py --data-dir data/raw`")
        return

    file = st.file_uploader("Upload a waste image", type=["jpg", "jpeg", "png", "webp"])
    if not file:
        return

    image = Image.open(file)
    st.image(image, caption="Input Image", use_container_width=True)

    x = preprocess_image(image)
    probs = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_class = class_names[top_idx]
    confidence = float(probs[top_idx])

    st.subheader("Prediction")
    st.write(f"**Class:** `{top_class}`")
    st.write(f"**Confidence:** `{confidence:.2%}`")

    if confidence < LOW_CONFIDENCE_THRESHOLD:
        st.warning(
            "Low-confidence prediction. Recommended action: send this image for manual review."
        )

    guidance = DISPOSAL_GUIDANCE.get(top_class, "Follow local disposal guidelines.")
    st.info(f"Disposal guidance: {guidance}")

    st.subheader("Class Probabilities")
    probs_df = pd.DataFrame(
        {
            "class": class_names,
            "probability": probs,
        }
    ).sort_values(by="probability", ascending=False)
    st.bar_chart(probs_df.set_index("class"))

    with st.expander("Responsible AI Notes"):
        st.markdown(
            "- Predictions are data-dependent and can fail on unusual lighting/backgrounds.\n"
            "- Confidence is not certainty; low-confidence outputs should be human-reviewed.\n"
            "- Model performance may vary across class imbalance and image quality."
        )


if __name__ == "__main__":
    main()
