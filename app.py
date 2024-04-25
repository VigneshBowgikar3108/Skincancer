import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
model = tf.keras.models.load_model("C:\\pbl 3\\skincancer\\models\\best_model.h5")
classes = {
    0: ("akiec", "Actinic Keratoses and Intraepithelial Carcinomae", "Commonly found on sun-exposed skin, these lesions are considered precancerous and can develop into squamous cell carcinoma if left untreated."),
    1: ("bcc", "Basal Cell Carcinoma", "The most common type of skin cancer, BCCs usually appear as pearly or waxy bumps or sores that may bleed or crust."),
    2: ("bkl", "Benign Keratosis-like Lesions", "These lesions are not cancerous but can sometimes resemble skin cancer. They include seborrheic keratoses, solar lentigines, and lichen planus-like keratoses."),
    3: ("df", "Dermatofibroma", "Common benign skin growths that are often firm and reddish-brown in color."),
    4: ("nv", "Melanocytic Nevi", "Commonly known as moles, these are usually harmless but can sometimes develop into melanoma."),
    5: ("vasc", "Pyogenic Granulomas and Hemorrhage", "Benign vascular lesions that often appear as red, raised bumps that may bleed easily."),
    6: ("mel", "Melanoma", "The most serious type of skin cancer, melanoma can be deadly if not caught early. It often appears as a new or changing mole with irregular borders and colors."),
}
def predict_skin_cancer(image):
    image = image.resize((28, 28)) 
    img_array = np.array(image) / 255.0  
    img_array = img_array.reshape(-1, 28, 28, 3)  

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    return classes[predicted_class]

st.title("Skin Cancer Prediction App")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    prediction = predict_skin_cancer(image)
    st.write("Predicted Class:", prediction[0])
    st.write("Description:", prediction[1])
    st.write("Description:", prediction[2])
    st.write("Please visit the nearest hospital")
    link_text = "To visit the Nearest hospitals click me"
    link_url = "https://www.google.com/search?q=skin+cancer+hospital+near+me&rlz=1C1CHBF_enIN1092IN1092&oq=skin+cancer+hospital+&gs_lcrp=EgZjaHJvbWUqBwgBEAAYgAQyBggAEEUYOTIHCAEQABiABDIHCAIQABiABDIHCAMQABiABDIHCAQQABiABDIICAUQABgHGB4yCAgGEAAYBxgeMggIBxAAGAcYHjIICAgQABgHGB4yCggJEAAYBxgPGB7SAQkxODI1OWowajeoAgCwAgA&sourceid=chrome&ie=UTF-8"
    st.markdown(f"[{link_text}]({link_url})")