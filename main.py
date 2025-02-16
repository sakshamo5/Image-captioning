import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import pickle
import gdown
import os

def download_file(url, output_path):
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)

def generate_and_display_caption(image_path, model_path, tokenizer_path, feature_extractor_path, max_length=34, img_size=224):
    caption_model = load_model(model_path)
    feature_extractor = load_model(feature_extractor_path)
    tokenizer = pickle.load(open(tokenizer_path, "rb"))

    img = load_img(image_path, target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    image_features = feature_extractor.predict(img, verbose=0)

    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([image_features, sequence], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index, None)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break

    caption = in_text.replace('startseq', "").replace("endseq", "").strip()

    img = load_img(image_path, target_size=(img_size, img_size))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(caption, fontsize=16, color='blue')
    st.pyplot(fig)

def main():
    st.title("Image Caption Generator")
    st.write("Upload an image and generate a caption using the trained model")

    # Download required files
    download_file("https://drive.google.com/uc?id=1it1byjlCF0poO1vzQcGi758lK2vrnV89", "models/feature_extractor.keras")
    download_file("https://drive.google.com/uc?id=1DEOU0r7qf76B9CBs69MZuTg53_vRlykH", "models/model.keras")
    download_file("https://drive.google.com/uc?id=1RK4N4v5tL7qOKbb77UYzNUTlmxwsXITJ", "models/tokenizer.pkl")

    uploaded_image = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    if uploaded_image is not None:
        with open("uploaded_img.jpg", "wb") as f:
            f.write(uploaded_image.getbuffer())

        model_path = "models/model.keras"
        tokenizer_path = "models/tokenizer.pkl"
        feature_extractor_path = "models/feature_extractor.keras"

        generate_and_display_caption("uploaded_img.jpg", model_path, tokenizer_path, feature_extractor_path)

if __name__ == "__main__":
    main()
