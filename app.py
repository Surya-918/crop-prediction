import streamlit as st
from dotenv import load_dotenv
from PIL import Image

from plant_disease.config import MIN_PREDICTION_CONFIDENCE
from plant_disease.prediction.inference import load_prediction_assets, predict_image
from plant_disease.services.treatment import (
    TreatmentServiceError,
    get_disease_solution,
)

load_dotenv()


@st.cache_resource
def get_prediction_assets():
    return load_prediction_assets()


st.set_page_config(page_title="Plant Disease Classifier", page_icon="🌿")
st.title("🌿 Crop Disease Prediction")
st.write(
    "Upload an image of a plant leaf, and the model will predict whether it is "
    "healthy or diseased."
)

try:
    model, class_indices = get_prediction_assets()
except (FileNotFoundError, ValueError) as error:
    st.error(f"{error} Train the classifier first with `python -m plant_disease.training`.")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a leaf image...",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    with Image.open(uploaded_file) as uploaded_image:
        image = uploaded_image.convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        with st.spinner("Analyzing leaf image..."):
            predicted_class, confidence = predict_image(model, image, class_indices)

        confidence_percent = confidence * 100
        is_not_a_leaf = predicted_class.casefold() == "not_a_leaf"
        if confidence < MIN_PREDICTION_CONFIDENCE:
            st.error("⚠️ Invalid image (not confidently recognized as a leaf).")
        elif is_not_a_leaf:
            st.error("⚠️ This is not a plant leaf.")
        else:
            st.success(f"🌿 Predicted disease: {predicted_class}")

        st.metric(label="Confidence score", value=f"{confidence_percent:.2f}%")

        if (
            confidence >= MIN_PREDICTION_CONFIDENCE
            and not is_not_a_leaf
            and "healthy" not in predicted_class.casefold()
        ):
            st.markdown("---")
            st.subheader("💊 Recommended treatment & cure")
            try:
                st.markdown(get_disease_solution(predicted_class))
            except TreatmentServiceError as error:
                st.warning(str(error))
else:
    st.info("Train the model and upload a leaf image to get a prediction.")
