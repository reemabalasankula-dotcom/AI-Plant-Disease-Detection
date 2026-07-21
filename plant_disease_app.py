import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("🌱 Multi Plant Disease Detection")

plant = st.selectbox(
    "Select Plant",
    ["Tomato", "Rice", "Potato"]
)

if plant == "Tomato":
    model = YOLO("tomato_best.pt")

elif plant == "Rice":
    model = YOLO("rice_best.pt")

else:
    model = YOLO("potato_best.pt")

option = st.radio(
    "Choose Input Method",
    ["📷 Take Photo", "📁 Upload Image"]
)

image = None

if option == "📷 Take Photo":
    camera_image = st.camera_input("Take Plant Photo")
    if camera_image:
        image = Image.open(camera_image)

else:
    uploaded_file = st.file_uploader(
        "Upload Plant Image",
        type=["jpg", "jpeg", "png", "webp"]
    )

    if uploaded_file:
        image = Image.open(uploaded_file)

if image:
    st.image(image, caption="Input Image")

    results = model.predict(image)

    for r in results:
        disease = r.names[r.probs.top1]
        confidence = float(r.probs.top1conf)

        st.success(f"Disease: {disease}")
        st.info(f"Confidence: {confidence:.2%}")