# 🧬 AI-Driven Liver Damage Detection

A deep-learning system that classifies **liver histopathology images** into three classes — **HCC** (Hepatocellular Carcinoma), **CC** (Cholangiocarcinoma) and **Normal Liver** — to help hepatologists make faster, more consistent diagnoses.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![Model](https://img.shields.io/badge/Best%20Model-DenseNet121-blue)
![Streamlit](https://img.shields.io/badge/App-Streamlit-FF4B4B?logo=streamlit&logoColor=white)

## 🎯 Business Problem

Liver damage detection today relies heavily on manual analysis of tissue slides by hepatologists, which is time-consuming and prone to human error. This project uses computer vision to analyse histopathology images and provide a second opinion that supports earlier and more accurate diagnosis.

## 🔬 Approach

1. **Exploratory Data Analysis** — class distribution, RGB intensity histograms, per-class colour mean/std, image size and aspect-ratio analysis
2. **Class balancing** — undersampling majority classes to avoid a biased model
3. **Preprocessing** — resizing to 224×224, normalisation and data augmentation
4. **Train / Validation / Test split** — 70 / 20 / 10
5. **Transfer learning** — benchmarked pre-trained CNNs (**DenseNet121, ResNet50, ResNet101, ResNet152, VGG16** and more) across 10, 15, 30, 50 and 70 epochs with EarlyStopping and ReduceLROnPlateau
6. **Model selection** — **DenseNet121** chosen as the best-performing model and exported as `.h5`
7. **Deployment** — Streamlit web app for image upload and real-time prediction with confidence scores

## 📁 Project Structure

```
├── data preprocessing code/   # EDA + preprocessing notebook
├── model building/            # Training & comparison of CNN architectures
├── model deployment/LDD.py    # Streamlit prediction app
└── LIVER DAMAGE DETECTION PROJECT.drawio.pdf   # Project architecture diagram
```

## 🚀 Run the App

```bash
pip install streamlit tensorflow pillow numpy

# Place the trained model next to the app, or point to it:
export MODEL_PATH=path/to/DenseNet121_best_model.h5

streamlit run "model deployment/LDD.py"
```

Upload a histopathology image (`.jpg`/`.png`) and the app shows the predicted class with confidence levels.

## 🛠️ Tech Stack

**Python** · **TensorFlow / Keras** · **Transfer Learning (DenseNet, ResNet, VGG)** · **OpenCV** · **Pandas** · **Seaborn** · **Streamlit** · **Google Colab**

> ⚠️ For research and educational purposes only — not a substitute for professional medical diagnosis.
