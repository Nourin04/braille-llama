# Import necessary libraries
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import requests
import json
import os
from dotenv import load_dotenv
from spellchecker import SpellChecker
from PIL import Image
import numpy as np
import cv2

# ✅ Load API Key from .env File
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ ERROR: Groq API key not found! Ensure you have set it in the .env file.")
    st.stop()

# ✅ Define Character Recognition Model
class CharacterRecognitionModel(nn.Module):
    def __init__(self, num_classes=26):
        super(CharacterRecognitionModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(1024 * 1 * 1, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ✅ Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharacterRecognitionModel(num_classes=26).to(device)

# Load trained model weights
model_path = "C:/Users/USER/Desktop/braille_llama/braille_recognition_model_final.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# ✅ Braille-to-English Dictionary
braille_to_english = {i: chr(97 + i) for i in range(26)}

# ✅ Image Processing Functions
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

def preprocess_image(image):
    try:
        image_np = np.array(image)
        if len(image_np.shape) == 3:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
        return Image.fromarray(image_np)
    except Exception as e:
        st.error(f"⚠️ Error during preprocessing: {e}")
        return None

def resize_image(image, new_size):
    return image.resize(new_size, Image.Resampling.LANCZOS)

def segment_braille_characters(image, num_characters):
    """Split the image into individual characters based on num_characters."""
    width, height = image.size
    segment_width = width // num_characters
    braille_images = [image.crop((i * segment_width, 0, (i + 1) * segment_width, height)) for i in range(num_characters)]
    return braille_images

# ✅ Braille to Text Conversion
def braille_to_text(braille_images):
    text = ""
    for img in braille_images:
        transformed_img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(transformed_img)
        predicted_class = torch.argmax(output, dim=1).item()
        text += braille_to_english[predicted_class]
    return text

# ✅ Correct Spelling using Groq LLaMA API
def correct_spelling(text, num_letters):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    prompt = (
        f"The given text is '{text}', and it contains {num_letters} letters. "
        f"Correct its spelling while ensuring the output has exactly {num_letters} letters."
    )

    payload = {
        "model": "llama3-8b-8192",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
    }

    response = requests.post(url, headers=headers, json=payload)
    try:
        response_json = response.json()
        if "choices" in response_json:
            corrected_text = response_json["choices"][0]["message"]["content"].strip()
            return corrected_text
        else:
            return text  # Fallback
    except json.JSONDecodeError:
        return text  # Fallback

# ✅ Streamlit UI
st.title("🔠 Braille-to-English Translator")
st.markdown("Convert Braille characters from images into English text using AI!")

uploaded_file = st.file_uploader("📤 Upload a Braille Image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Braille Image", use_column_width=True)

    # Get user input for number of characters
    num_letters = st.number_input("🔢 Enter Number of Braille Characters", min_value=1, max_value=10, value=5)

    if st.button("🛠 Convert Braille to Text"):
        resized_image = resize_image(image, (num_letters * 28, 28))
        preprocessed_image = preprocess_image(resized_image)

        if preprocessed_image:
            braille_images = segment_braille_characters(preprocessed_image, num_letters)
            predicted_text = braille_to_text(braille_images)
            corrected_text = correct_spelling(predicted_text, num_letters)

            st.subheader("📜 Recognized Text:")
            st.write(f"🔹 **Original:** `{predicted_text}`")
            st.write(f"✅ **Corrected:** `{corrected_text}`")
        else:
            st.error("⚠️ Preprocessing failed. Check the image!")

st.info("Upload a Braille image and specify the number of characters for recognition.")  
