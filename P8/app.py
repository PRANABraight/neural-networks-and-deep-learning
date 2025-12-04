import streamlit as st
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import time

# Mocking TensorFlow/Keras for users without it
# import tensorflow as tf
# from tensorflow.keras import layers, models

# Set page config
st.set_page_config(page_title="Convolutional Autoencoder (Demo)", layout="wide")

st.title("Convolutional Autoencoder for Image Reconstruction")
st.write("This app demonstrates a Convolutional Autoencoder trained on insect images.")
st.info("Running in Demo Mode (Hardcoded) - TensorFlow not required.")

# Sidebar configuration
st.sidebar.header("Configuration")
epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=10)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=2)
test_size = st.sidebar.slider("Test Split Size", 0.1, 0.5, 0.2)

# Load Data
@st.cache_data
def load_images(base_path, target_size=(128, 128)):
    images = []
    labels = []
    categories = ['Butterfly', 'Dragonfly', 'Grasshopper', 'Ladybird', 'Mosquito']
    
    if not os.path.exists(base_path):
        st.error(f"Base path {base_path} not found!")
        return np.array([])

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_categories = len(categories)
    
    for i, category in enumerate(categories):
        status_text.text(f"Loading {category}...")
        path = os.path.join(base_path, category)
        if not os.path.exists(path):
            continue

        for filename in os.listdir(path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(path, filename)
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(target_size)
                    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                    images.append(img_array)
                    labels.append(category)
                except Exception as e:
                    pass
        progress_bar.progress((i + 1) / total_categories)
    
    status_text.text("Images loaded successfully!")
    progress_bar.empty()
    return np.array(images)

# Mock Model Class
class MockAutoencoder:
    def predict(self, data):
        # In a perfect autoencoder, output equals input. 
        # We'll just return the input to simulate a "good" reconstruction.
        return data

# Main execution
base_dir = os.path.join(os.path.dirname(__file__), 'assets')
images = load_images(base_dir)

if len(images) > 0:
    st.write(f"**Total images loaded:** {images.shape[0]}")
    
    X_train, X_test = train_test_split(images, test_size=test_size, random_state=42)
    st.write(f"**Training samples:** {X_train.shape[0]} | **Testing samples:** {X_test.shape[0]}")

    if st.button("Train Autoencoder"):
        with st.spinner("Training model... (Simulated)"):
            # Simulate training delay
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.02)
                progress_bar.progress(i + 1)
            
            # Mock history data
            epochs_range = range(1, epochs + 1)
            loss = [0.5 * (0.9 ** i) for i in epochs_range]
            val_loss = [0.55 * (0.9 ** i) + 0.02 for i in epochs_range]
            
            history = {'loss': loss, 'val_loss': val_loss}
            model = MockAutoencoder()
            
            st.session_state['model'] = model
            st.session_state['history'] = history
            st.success("Training complete!")

    if 'model' in st.session_state:
        model = st.session_state['model']
        history = st.session_state['history']

        # Plot Loss
        st.subheader("Training History")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(history['loss'], label='Training Loss')
        ax.plot(history['val_loss'], label='Validation Loss')
        ax.set_title('Model Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss (MSE)')
        ax.legend()
        st.pyplot(fig)

        # Reconstruction Visualization
        st.subheader("Reconstruction Results (Test Set)")
        decoded_imgs = model.predict(X_test)
        
        n = st.slider("Number of images to display", 1, 10, 5)
        
        fig2, axes = plt.subplots(2, n, figsize=(20, 8))
        for i in range(n):
            if i < len(X_test):
                # Display original
                axes[0, i].imshow(X_test[i])
                axes[0, i].set_title("Original")
                axes[0, i].axis("off")

                # Display reconstruction (Mocked as original)
                axes[1, i].imshow(decoded_imgs[i])
                axes[1, i].set_title("Reconstructed")
                axes[1, i].axis("off")
        st.pyplot(fig2)
        
        # User Inference
        st.subheader("Test with your own image")
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file).convert('RGB')
                st.image(img, caption='Uploaded Image', width=200)
                
                img_resized = img.resize((128, 128))
                img_array = np.array(img_resized) / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                prediction = model.predict(img_array)
                
                st.write("Reconstructed Image:")
                st.image(prediction[0], width=200)
            except Exception as e:
                st.error(f"Error processing image: {e}")

else:
    st.warning("No images found. Please check the 'assets' folder.")
