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

DISPOSAL_GUIDANCE = {
    "plastic": "Place in plastics recycling bin if clean and dry.",
    "paper": "Place in paper recycling bin if not food-soiled.",
    "glass": "Place in glass recycling container; avoid broken unsafe handling.",
    "metal": "Place in metal recycling bin (cans/foil cleaned).",
    "organic": "Place in compost/organic waste bin.",
    "other": "Place in general waste unless local rules specify otherwise.",
}

CLASS_COLORS = {
    "plastic": "#4f8df7",
    "paper": "#8e7dff",
    "glass": "#2fbf71",
    "metal": "#6f6f75",
    "organic": "#57a646",
    "other": "#9c6b38",
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
    arr = np.expand_dims(arr, axis=0)
    return arr


def apply_styles():
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #f7f9fc 0%, #eef2f9 100%);
            }
            .block-container {
                padding-top: 1.5rem;
                padding-bottom: 1.5rem;
                max-width: 900px;
            }
            .hero-card {
                background: white;
                border-radius: 16px;
                border: 1px solid #e6e9f0;
                padding: 1.25rem 1.25rem 0.8rem 1.25rem;
                margin-bottom: 1rem;
                box-shadow: 0 8px 24px rgba(15, 23, 42, 0.05);
            }
            .prediction-card {
                background: #ffffff;
                border-radius: 14px;
                border: 1px solid #e6e9f0;
                padding: 0.9rem 1rem;
                box-shadow: 0 6px 18px rgba(15, 23, 42, 0.05);
            }
            .badge {
                display: inline-block;
                padding: 0.25rem 0.6rem;
                border-radius: 999px;
                color: white;
                font-size: 0.85rem;
                font-weight: 600;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Smart Waste Sorting Assistant", layout="centered")
    apply_styles()
    st.markdown(
        """
        <div class="hero-card">
            <h1 style="margin:0; font-size: 2rem;">Smart Waste Sorting Assistant</h1>
            <p style="margin:0.45rem 0 0.8rem 0; color:#4b5563;">
                Upload an image to classify waste into one of six categories and get disposal guidance instantly.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
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

    x = preprocess_image(image)
    probs = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_class = class_names[top_idx]
    confidence = float(probs[top_idx])

    left_col, right_col = st.columns([1.05, 1.15], gap="large")
    with left_col:
        st.image(image, caption="Uploaded image", use_container_width=True)
    with right_col:
        color = CLASS_COLORS.get(top_class, "#4b5563")
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.subheader("Prediction")
        st.markdown(
            f'<span class="badge" style="background:{color};">{top_class.upper()}</span>',
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{confidence:.2%}")
        guidance = DISPOSAL_GUIDANCE.get(top_class, "Follow local disposal guidelines.")
        st.info(f"Disposal guidance: {guidance}")
        st.markdown("</div>", unsafe_allow_html=True)

    probs_df = pd.DataFrame(
        {
            "class": class_names,
            "probability": probs,
        }
    ).sort_values(by="probability", ascending=False)
    probs_df["probability"] = probs_df["probability"] * 100

    st.subheader("Class Probabilities")
    st.dataframe(
        probs_df.rename(columns={"probability": "probability (%)"}),
        use_container_width=True,
        hide_index=True,
    )
    st.bar_chart(probs_df.set_index("class"))

    with st.expander("Responsible AI Notes"):
        st.markdown(
            "- Predictions are data-dependent and can fail on unusual lighting/backgrounds.\n"
            "- Confidence is not certainty; low-confidence outputs should be human-reviewed.\n"
            "- Model performance may vary across class imbalance and image quality."
        )


if __name__ == "__main__":
    main()
