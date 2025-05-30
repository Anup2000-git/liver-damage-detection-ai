import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load trained model
model = load_model("E:/Mandeep/360 DigiTMG/PROJECTS/LIVER DAMAGE DETECTION PROJECT/Deployment/DenseNet121_best_model.h5")

# Class names
class_names = ["CC", "HCC", "NORMAL LIVER"]
colors = {"CC": "#FF5733", "HCC": "#C70039", "NORMAL LIVER": "#2ECC71"}

# Image preprocessing
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Page Config
st.set_page_config(page_title="Liver Damage Detection", page_icon="🧬", layout="centered")

# Custom CSS
st.markdown("""
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            color: #4CAF50;
            text-align: center;
        }
        .subtitle {
            font-size: 18px;
            text-align: center;
            color: #666;
            margin-bottom: 20px;
        }
        .prediction {
            font-size: 28px;
            font-weight: bold;
            text-align: center;
        }
        .confidence {
            font-size: 20px;
            text-align: center;
            color: #888;
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: #888;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Instructions
st.sidebar.header("📝 How to Use")
st.sidebar.markdown("""
1. Upload your organization logo (optional)  
2. Upload a histopathology image (`.jpg`, `.jpeg`, `.png`)  
3. View the prediction and confidence levels  
""")

# Optional Logo Upload
logo_file = st.sidebar.file_uploader("📌 Upload AISpry Logo", type=["jpg", "jpeg", "png"])
if logo_file:
    logo_img = Image.open(logo_file)
    st.image(logo_img, width=150)

# Main UI Title
st.markdown('<div class="main-title">🧬 Liver Damage Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a histopathology image to detect liver condition (HCC, CC, or Normal Liver)</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload Histopathology Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="📷 Uploaded Image", use_column_width=True)

    with st.spinner("🧠 Analyzing image with AI model..."):
        try:
            processed_img = preprocess_image(img)
            predictions = model.predict(processed_img)

            if predictions is not None and len(predictions) > 0:
                predicted_class = class_names[np.argmax(predictions[0])]
                confidence = np.max(predictions[0]) * 100

                # Display Prediction
                st.markdown(f'<div class="prediction">✅ Prediction: <span style="color:{colors[predicted_class]};">{predicted_class}</span></div>', unsafe_allow_html=True)
                st.markdown(f'<div class="confidence">🔢 Confidence: {confidence:.2f}%</div>', unsafe_allow_html=True)

                # Display Probabilities
                st.subheader("📊 Prediction Probabilities")
                for i, cls in enumerate(class_names):
                    prob = float(predictions[0][i])
                    st.progress(prob)
                    st.markdown(f"**{cls}**: {prob * 100:.2f}%")

            else:
                st.error("🚫 Model returned no predictions. Please check the image or try another.")

        except Exception as e:
            st.error(f"🚫 Error during prediction: {str(e)}")

else:
    st.info("🖼️ Please upload a valid histopathology image to get started.")

# Footer
st.markdown(
    '<div class="footer">🔗 Developed by <b>Mandeep Singh Thakur</b> | Powered by TensorFlow & Streamlit</div>',
    unsafe_allow_html=True
)

