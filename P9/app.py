import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os

# Set page config
st.set_page_config(page_title="Recurrent Autoencoder (PyTorch)", layout="wide")

st.title("Recurrent Autoencoder for Time Series Reconstruction")
st.markdown("""
This application demonstrates a **Recurrent Autoencoder** using LSTM layers (implemented in **PyTorch**) to reconstruct temperature time-series data.
The model learns to compress 30-day temperature sequences into a latent representation and then reconstruct them.
""")

# 1. Data Loading
st.header("1. Data Loading & Preprocessing")

@st.cache_data
def load_and_preprocess_data():
    # Check if files exist
    if not os.path.exists('training.csv') or not os.path.exists('testing.csv'):
        return None, None, None, None, None, None

    # Load
    train_df = pd.read_csv('training.csv', sep='\t')
    test_df = pd.read_csv('testing.csv', sep=',')
    
    # Clean
    train_df = train_df.dropna()
    test_df = test_df.dropna()
    
    # Normalize
    scaler = MinMaxScaler()
    train_data = scaler.fit_transform(train_df[['Temperature']])
    test_data = scaler.transform(test_df[['Temperature']])
    
    return train_df, test_df, train_data, test_data, scaler

train_df, test_df, train_data, test_data, scaler = load_and_preprocess_data()

if train_df is None:
    st.error("Data files (training.csv, testing.csv) not found in the current directory.")
else:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Data")
        st.dataframe(train_df.head())
    with col2:
        st.subheader("Testing Data")
        st.dataframe(test_df.head())

    # Create Sequences
    SEQ_LENGTH = 30
    
    def create_sequences(data, seq_length):
        X = []
        for i in range(len(data) - seq_length):
            X.append(data[i:i+seq_length])
        return np.array(X)

    X_train = create_sequences(train_data, SEQ_LENGTH)
    X_test = create_sequences(test_data, SEQ_LENGTH)

    # Convert to PyTorch Tensors
    X_train_tensor = torch.FloatTensor(X_train)
    X_test_tensor = torch.FloatTensor(X_test)

    st.info(f"**Sequence Length:** {SEQ_LENGTH}")
    st.write(f"**Training Sequences Shape:** {X_train.shape}")
    st.write(f"**Testing Sequences Shape:** {X_test.shape}")

    # 2. Model Architecture
    st.header("2. Model Architecture")

    class RecurrentAutoencoder(nn.Module):
        def __init__(self, seq_length, n_features, embedding_dim=128):
            super(RecurrentAutoencoder, self).__init__()
            self.seq_length = seq_length
            self.n_features = n_features
            self.embedding_dim = embedding_dim

            # Encoder
            self.encoder_lstm = nn.LSTM(
                input_size=n_features,
                hidden_size=embedding_dim,
                batch_first=True
            )

            # Decoder
            self.decoder_lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=embedding_dim,
                batch_first=True
            )
            
            self.output_layer = nn.Linear(embedding_dim, n_features)

        def forward(self, x):
            # Encoder
            # x shape: (batch, seq_len, n_features)
            _, (hidden, _) = self.encoder_lstm(x)
            # hidden shape: (1, batch, embedding_dim)
            
            # Latent vector (squeeze the layer dimension)
            latent = hidden.squeeze(0) # (batch, embedding_dim)
            
            # Repeat latent vector for decoder input
            # We need to repeat it seq_length times to match the input structure for reconstruction
            # Shape: (batch, seq_len, embedding_dim)
            decoder_input = latent.unsqueeze(1).repeat(1, self.seq_length, 1)
            
            # Decoder
            decoder_output, _ = self.decoder_lstm(decoder_input)
            # decoder_output shape: (batch, seq_len, embedding_dim)
            
            # Final output layer
            prediction = self.output_layer(decoder_output)
            # prediction shape: (batch, seq_len, n_features)
            
            return prediction

    model = RecurrentAutoencoder(seq_length=SEQ_LENGTH, n_features=1, embedding_dim=128)
    
    st.code(str(model))

    # 3. Training
    st.header("3. Training")
    
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
        st.session_state.train_losses = []
        st.session_state.trained_model = None

    train_btn = st.button("Train Model (10 Epochs)")

    if train_btn:
        with st.spinner("Training model..."):
            # Setup
            model = RecurrentAutoencoder(seq_length=SEQ_LENGTH, n_features=1, embedding_dim=128)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            
            # DataLoader
            dataset = TensorDataset(X_train_tensor, X_train_tensor) # Autoencoder target is input
            dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
            
            epochs = 10
            train_losses = []
            
            progress_bar = st.progress(0)
            
            for epoch in range(epochs):
                model.train()
                epoch_loss = 0
                for batch_x, batch_y in dataloader:
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                train_losses.append(avg_loss)
                progress_bar.progress((epoch + 1) / epochs)
            
            st.session_state.trained_model = model
            st.session_state.train_losses = train_losses
            st.session_state.model_trained = True
            st.success("Training Complete!")

    if st.session_state.model_trained:
        # Plot Loss
        st.subheader("Training Performance")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(st.session_state.train_losses, label='Training Loss')
        ax.set_title('Model Loss over Epochs')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss (MSE)')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # 4. Evaluation
        st.header("4. Evaluation & Reconstruction")
        
        # Predict
        model.eval()
        with torch.no_grad():
            X_test_pred_tensor = st.session_state.trained_model(X_test_tensor)
            X_test_pred = X_test_pred_tensor.numpy()
        
        # Visualization
        st.subheader("Original vs Reconstructed Sequence")
        
        # Slider for index
        idx = st.slider("Select a Test Sequence Index to Visualize", 0, len(X_test)-1, 0)
        
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot(X_test[idx].flatten(), label='Original Sequence', marker='o', color='blue')
        ax2.plot(X_test_pred[idx].flatten(), label='Reconstructed Sequence', marker='x', color='red', linestyle='--')
        ax2.set_title(f'Sequence Reconstruction (Index {idx})')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Normalized Temperature')
        ax2.legend()
        ax2.grid(True)
        st.pyplot(fig2)
        
        # Calculate MSE for this sequence
        mse = np.mean(np.square(X_test[idx] - X_test_pred[idx]))
        st.metric("Reconstruction MSE for this sequence", f"{mse:.6f}")

    else:
        st.info("Click 'Train Model' to start training and see results.")
