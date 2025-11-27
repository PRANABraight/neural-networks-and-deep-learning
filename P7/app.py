import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Set page config
st.set_page_config(page_title="Autoencoder Lab", layout="wide")

# Title and Introduction
st.title("Autoencoder Feature Extraction & Dimensionality Reduction")
st.markdown("""
This application demonstrates the use of **CNN** and **LSTM** Autoencoders for feature extraction and dimensionality reduction on the MNIST dataset.
You can train models, visualize reconstructions, and explore the latent space.
""")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "CNN Autoencoder", "LSTM Autoencoder", "Comparison"])

# Data Loading and Preprocessing
@st.cache_resource
def load_data():
    """Loads and preprocesses the MNIST dataset."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape for CNN (28, 28, 1)
    x_train_cnn = np.expand_dims(x_train, axis=-1)
    x_test_cnn = np.expand_dims(x_test, axis=-1)
    
    # Reshape for LSTM (28, 28) - treated as sequence of 28 rows
    # Actually MNIST is already (N, 28, 28), so it's ready for LSTM if we consider rows as timesteps
    x_train_lstm = x_train
    x_test_lstm = x_test
    
    return (x_train_cnn, x_test_cnn, x_train_lstm, x_test_lstm, y_train, y_test)

# Load data
with st.spinner("Loading MNIST dataset..."):
    x_train_cnn, x_test_cnn, x_train_lstm, x_test_lstm, y_train, y_test = load_data()

if page == "Introduction":
    st.header("Dataset Overview")
    st.write(f"**Dataset:** MNIST (Modified National Institute of Standards and Technology database)")
    st.write(f"**Training Samples:** {x_train_cnn.shape[0]}")
    st.write(f"**Test Samples:** {x_test_cnn.shape[0]}")
    st.write(f"**Image Size:** 28x28 pixels")
    
    st.subheader("Sample Images")
    fig, axes = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        axes[i].imshow(x_train_cnn[i].reshape(28, 28), cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"Label: {y_train[i]}")
    st.pyplot(fig)

elif page == "CNN Autoencoder":
    st.header("CNN Autoencoder")
    st.markdown("A Convolutional Autoencoder learns spatial hierarchies of features.")
    
    # Model Parameters
    latent_dim = st.sidebar.slider("Latent Dimension (CNN)", 2, 64, 32)
    epochs = st.sidebar.slider("Epochs (CNN)", 1, 50, 10)
    
    # Model Definition
    def build_cnn_autoencoder(latent_dim):
        input_img = layers.Input(shape=(28, 28, 1))
        
        # Encoder
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Flatten for latent space visualization if needed, but here we keep spatial structure or flatten?
        # The notebook used a Dense layer for latent space. Let's follow the notebook structure if possible.
        # Notebook structure: Conv -> MaxPool -> Dense (Latent) -> Dense -> Reshape -> UpSampling -> Conv
        # Let's verify notebook structure.
        # Notebook:
        # Encoder: Conv2D(32), MaxPool, Conv2D(64), MaxPool, Flatten, Dense(latent_dim)
        # Decoder: Dense(7*7*64), Reshape, Conv2DTranspose(64), UpSampling, Conv2DTranspose(32), UpSampling, Conv2D(1)
        
        # Let's implement a standard CNN AE for now, or try to match notebook.
        # Matching notebook structure is better for consistency.
        
        # Re-implementing based on typical structure for now to ensure it works, 
        # but I will try to align with the notebook's logic which likely used a Dense bottleneck.
        
        x = layers.Flatten()(encoded)
        latent = layers.Dense(latent_dim, activation='relu')(x)
        
        # Decoder
        x = layers.Dense(4 * 4 * 8, activation='relu')(latent) # 4x4x8 = 128
        x = layers.Reshape((4, 4, 8))(x)
        x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        x = layers.UpSampling2D((2, 2))(x)
        x = layers.Conv2D(32, (3, 3), activation='relu')(x) # Valid padding to get to 28? No, let's stick to 'same' and crop if needed or use specific padding.
        # 4x4 -> 8x8 -> 16x16. We need 28x28.
        # Let's simplify: 28 -> 14 -> 7.
        # Encoder: 28x28 -> 14x14 -> 7x7.
        
        # Revised Encoder
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = layers.MaxPooling2D((2, 2), padding='same')(x) # 14x14
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        encoded_spatial = layers.MaxPooling2D((2, 2), padding='same')(x) # 7x7
        
        x = layers.Flatten()(encoded_spatial)
        latent = layers.Dense(latent_dim, name='latent_layer')(x)
        
        # Revised Decoder
        x = layers.Dense(7 * 7 * 64, activation='relu')(latent)
        x = layers.Reshape((7, 7, 64))(x)
        x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x) # 14x14
        x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x) # 28x28
        decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        autoencoder = models.Model(input_img, decoded)
        encoder = models.Model(input_img, latent)
        
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder, encoder

    # Train Model
    if st.button("Train CNN Autoencoder"):
        with st.spinner("Training CNN Autoencoder..."):
            autoencoder, encoder = build_cnn_autoencoder(latent_dim)
            history = autoencoder.fit(
                x_train_cnn, x_train_cnn,
                epochs=epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_cnn, x_test_cnn),
                verbose=0
            )
            
            # Save models to session state to persist
            st.session_state['cnn_autoencoder'] = autoencoder
            st.session_state['cnn_encoder'] = encoder
            st.session_state['cnn_history'] = history.history
            st.success("Training Complete!")

    # Display Results if model exists
    if 'cnn_autoencoder' in st.session_state:
        autoencoder = st.session_state['cnn_autoencoder']
        encoder = st.session_state['cnn_encoder']
        history = st.session_state['cnn_history']
        
        # Plot Loss
        st.subheader("Training History")
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(history['loss'], label='Train Loss')
        ax_loss.plot(history['val_loss'], label='Val Loss')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss (MSE)')
        ax_loss.legend()
        st.pyplot(fig_loss)
        
        # Reconstructions
        st.subheader("Reconstructions")
        decoded_imgs = autoencoder.predict(x_test_cnn[:10])
        
        fig_rec, axes = plt.subplots(2, 10, figsize=(20, 4))
        for i in range(10):
            # Original
            axes[0, i].imshow(x_test_cnn[i].reshape(28, 28), cmap='gray')
            axes[0, i].axis('off')
            if i == 5: axes[0, i].set_title("Original")
            
            # Reconstructed
            axes[1, i].imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
            axes[1, i].axis('off')
            if i == 5: axes[1, i].set_title("Reconstructed")
        st.pyplot(fig_rec)
        
        # Latent Space Visualization
        st.subheader("Latent Space Visualization (t-SNE)")
        if st.button("Generate t-SNE"):
            with st.spinner("Generating t-SNE projection..."):
                # Encode test data
                latent_features = encoder.predict(x_test_cnn)
                
                # t-SNE
                tsne = TSNE(n_components=2, random_state=42)
                tsne_results = tsne.fit_transform(latent_features[:1000]) # Use subset for speed
                
                # Plot
                fig_tsne, ax_tsne = plt.subplots(figsize=(10, 8))
                scatter = ax_tsne.scatter(
                    tsne_results[:, 0], tsne_results[:, 1],
                    c=y_test[:1000], cmap='tab10', alpha=0.6
                )
                plt.colorbar(scatter, label='Digit Label')
                ax_tsne.set_title("t-SNE of CNN Latent Space")
                st.pyplot(fig_tsne)

elif page == "LSTM Autoencoder":
    st.header("LSTM Autoencoder")
    st.markdown("An LSTM Autoencoder captures temporal dependencies in sequential data.")
    
    # Model Parameters
    lstm_latent_dim = st.sidebar.slider("Latent Dimension (LSTM)", 2, 64, 16)
    lstm_epochs = st.sidebar.slider("Epochs (LSTM)", 1, 50, 10)
    
    # Model Definition
    def build_lstm_autoencoder(latent_dim):
        # Input: (28, 28) -> Sequence of 28 vectors of size 28
        input_seq = layers.Input(shape=(28, 28))
        
        # Encoder
        x = layers.LSTM(64, return_sequences=True)(input_seq)
        x = layers.LSTM(32, return_sequences=False)(x)
        latent = layers.Dense(latent_dim, name='lstm_latent')(x)
        
        # Decoder
        x = layers.RepeatVector(28)(latent)
        x = layers.LSTM(32, return_sequences=True)(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        decoded = layers.TimeDistributed(layers.Dense(28))(x) # Output: (28, 28)
        
        autoencoder = models.Model(input_seq, decoded)
        encoder = models.Model(input_seq, latent)
        
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder, encoder

    # Train Model
    if st.button("Train LSTM Autoencoder"):
        with st.spinner("Training LSTM Autoencoder..."):
            lstm_ae, lstm_enc = build_lstm_autoencoder(lstm_latent_dim)
            history = lstm_ae.fit(
                x_train_lstm, x_train_lstm,
                epochs=lstm_epochs,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test_lstm, x_test_lstm),
                verbose=0
            )
            
            st.session_state['lstm_autoencoder'] = lstm_ae
            st.session_state['lstm_encoder'] = lstm_enc
            st.session_state['lstm_history'] = history.history
            st.success("Training Complete!")

    # Display Results
    if 'lstm_autoencoder' in st.session_state:
        lstm_ae = st.session_state['lstm_autoencoder']
        history = st.session_state['lstm_history']
        
        # Plot Loss
        st.subheader("Training History")
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(history['loss'], label='Train Loss')
        ax_loss.plot(history['val_loss'], label='Val Loss')
        ax_loss.set_xlabel('Epochs')
        ax_loss.set_ylabel('Loss (MSE)')
        ax_loss.legend()
        st.pyplot(fig_loss)
        
        # Reconstructions
        st.subheader("Reconstructions")
        decoded_seqs = lstm_ae.predict(x_test_lstm[:6])
        
        fig_seq, axes = plt.subplots(2, 3, figsize=(15, 8))
        for i, ax in enumerate(axes.flat):
            # Plotting the sequence as a 1D signal (flattened) or row-by-row?
            # The notebook plotted sequences. Let's plot the image rows as time series.
            # Actually, for visual comparison, plotting them as images is still valid for MNIST,
            # but to show "sequence" reconstruction, maybe plot a few rows?
            # Let's stick to image reconstruction for visual clarity, as it's MNIST.
            
            ax.imshow(decoded_seqs[i], cmap='gray')
            ax.set_title(f"Reconstructed {i}")
            ax.axis('off')
            
        st.pyplot(fig_seq)
        
        st.write("Original Images for comparison:")
        fig_orig, axes_orig = plt.subplots(1, 6, figsize=(15, 3))
        for i in range(6):
            axes_orig[i].imshow(x_test_lstm[i], cmap='gray')
            axes_orig[i].axis('off')
        st.pyplot(fig_orig)

elif page == "Comparison":
    st.header("Model Comparison")
    
    if 'cnn_autoencoder' in st.session_state and 'lstm_autoencoder' in st.session_state:
        cnn_ae = st.session_state['cnn_autoencoder']
        lstm_ae = st.session_state['lstm_autoencoder']
        
        # Evaluate MSE
        cnn_mse = cnn_ae.evaluate(x_test_cnn, x_test_cnn, verbose=0)
        lstm_mse = lstm_ae.evaluate(x_test_lstm, x_test_lstm, verbose=0)
        
        st.subheader("Performance Metrics")
        col1, col2 = st.columns(2)
        col1.metric("CNN MSE", f"{cnn_mse:.4f}")
        col2.metric("LSTM MSE", f"{lstm_mse:.4f}")
        
        # Parameter Count
        cnn_params = cnn_ae.count_params()
        lstm_params = lstm_ae.count_params()
        
        st.subheader("Model Complexity")
        st.bar_chart(pd.DataFrame({
            'Parameters': [cnn_params, lstm_params]
        }, index=['CNN', 'LSTM']))
        
    else:
        st.warning("Please train both models first to see the comparison.")

