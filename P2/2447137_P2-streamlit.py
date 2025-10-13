import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score

# Set page configuration
st.set_page_config(page_title="Activation Functions Explorer", layout="wide", page_icon="🧠")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        font-weight: bold;
        margin-top: 1.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🧠 Activation Functions in Neural Networks</p>', unsafe_allow_html=True)
st.write("Interactive exploration of activation functions and their impact on neural network performance")

# Activation Functions
def step_function(x):
    return np.where(x >= 0, 1, 0)

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def bipolar_sigmoid(x):
    return (2 / (1 + np.exp(-np.clip(x, -500, 500)))) - 1

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

# Derivatives
def sigmoid_derivative(x):
    return x * (1 - x)

def tanh_derivative(x):
    return 1 - np.power(x, 2)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

# Neural Network Class
class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size) * 0.5
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.5
        self.b2 = np.zeros((1, output_size))
        
        self.activation_name = activation
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_derivative
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_derivative = tanh_derivative
        elif activation == 'relu':
            self.activation = relu
            self.activation_derivative = relu_derivative
        
        self.loss_history = []
    
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.activation(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, learning_rate):
        m = X.shape[0]
        
        dz2 = self.a2 - y
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        da1 = np.dot(dz2, self.W2.T)
        dz1 = da1 * self.activation_derivative(self.a1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            loss = -np.mean(y * np.log(output + 1e-8) + (1 - y) * np.log(1 - output + 1e-8))
            self.loss_history.append(loss)
            self.backward(X, y, learning_rate)
    
    def predict(self, X):
        output = self.forward(X)
        return (output > 0.5).astype(int)

# Sidebar for navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio("Choose a section:", 
                        ["Activation Functions", "Neural Network Training", "Performance Comparison"])

# XOR Dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Page 1: Activation Functions Visualization
if page == "Activation Functions":
    st.markdown('<p class="sub-header">📊 Activation Functions Visualization</p>', unsafe_allow_html=True)
    
    st.write("""
    Activation functions introduce non-linearity into neural networks, enabling them to learn complex patterns.
    Select different functions below to visualize their behavior.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_functions = st.multiselect(
            "Select Activation Functions to Visualize:",
            ["Step", "Binary Sigmoid", "Bipolar Sigmoid", "Tanh", "ReLU"],
            default=["Binary Sigmoid", "Tanh", "ReLU"]
        )
    
    with col2:
        x_range = st.slider("Input Range:", -20.0, 20.0, (-10.0, 10.0))
    
    if selected_functions:
        x = np.linspace(x_range[0], x_range[1], 400)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {'Step': 'blue', 'Binary Sigmoid': 'green', 'Bipolar Sigmoid': 'red', 
                  'Tanh': 'magenta', 'ReLU': 'cyan'}
        
        for func_name in selected_functions:
            if func_name == "Step":
                ax.plot(x, step_function(x), color=colors[func_name], linewidth=2, label=func_name)
            elif func_name == "Binary Sigmoid":
                ax.plot(x, sigmoid(x), color=colors[func_name], linewidth=2, label=func_name)
            elif func_name == "Bipolar Sigmoid":
                ax.plot(x, bipolar_sigmoid(x), color=colors[func_name], linewidth=2, label=func_name)
            elif func_name == "Tanh":
                ax.plot(x, tanh(x), color=colors[func_name], linewidth=2, label=func_name)
            elif func_name == "ReLU":
                ax.plot(x, relu(x), color=colors[func_name], linewidth=2, label=func_name)
        
        ax.set_xlabel('Input', fontsize=12)
        ax.set_ylabel('Output', fontsize=12)
        ax.set_title('Activation Functions Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    # Information about each function
    st.markdown('<p class="sub-header">📝 Function Characteristics</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **Step Function**
        - Output: {0, 1}
        - Not differentiable
        - Used in perceptrons
        """)
        
        st.success("""
        **Binary Sigmoid**
        - Output: [0, 1]
        - Smooth gradient
        - Good for probabilities
        - Vanishing gradient issue
        """)
    
    with col2:
        st.warning("""
        **Bipolar Sigmoid**
        - Output: [-1, 1]
        - Centered at zero
        - Better than binary sigmoid
        """)
        
        st.info("""
        **Tanh**
        - Output: [-1, 1]
        - Zero-centered
        - Faster convergence
        - Still has vanishing gradient
        """)
    
    with col3:
        st.success("""
        **ReLU**
        - Output: [0, ∞)
        - Fast computation
        - No vanishing gradient
        - 'Dying ReLU' problem
        - Most popular for hidden layers
        """)

# Page 2: Neural Network Training
elif page == "Neural Network Training":
    st.markdown('<p class="sub-header">🎯 Neural Network Training on XOR Problem</p>', unsafe_allow_html=True)
    
    st.write("""
    The XOR problem is a classic non-linearly separable problem that requires at least one hidden layer to solve.
    Train a neural network with different activation functions and observe the results.
    """)
    
    # Show XOR dataset
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**XOR Truth Table:**")
        st.dataframe({
            'Input 1': X[:, 0],
            'Input 2': X[:, 1],
            'Output': y.flatten()
        })
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 6))
        colors = ['red' if label == 0 else 'blue' for label in y.flatten()]
        ax.scatter(X[:, 0], X[:, 1], c=colors, s=200, edgecolors='black', linewidth=2)
        ax.set_xlabel('Input 1', fontsize=12)
        ax.set_ylabel('Input 2', fontsize=12)
        ax.set_title('XOR Problem (Red=0, Blue=1)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        st.pyplot(fig)
    
    # Training parameters
    st.markdown('<p class="sub-header">⚙️ Training Configuration</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        activation = st.selectbox("Activation Function:", ["sigmoid", "tanh", "relu"])
    
    with col2:
        epochs = st.slider("Training Epochs:", 1000, 10000, 5000, 1000)
    
    with col3:
        learning_rate = st.slider("Learning Rate:", 0.01, 1.0, 0.5 if activation != 'relu' else 0.1, 0.01)
    
    if st.button("🚀 Train Neural Network", type="primary"):
        with st.spinner(f"Training with {activation.upper()} activation..."):
            nn = SimpleNeuralNetwork(input_size=2, hidden_size=4, output_size=1, activation=activation)
            nn.train(X, y, epochs=epochs, learning_rate=learning_rate)
            
            predictions = nn.predict(X)
            accuracy = accuracy_score(y, predictions)
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Final Accuracy", f"{accuracy*100:.2f}%")
                
                st.write("**Predictions vs Actual:**")
                results_df = {
                    'Input 1': X[:, 0],
                    'Input 2': X[:, 1],
                    'Predicted': predictions.flatten(),
                    'Actual': y.flatten(),
                    'Correct': ['✅' if p == a else '❌' for p, a in zip(predictions.flatten(), y.flatten())]
                }
                st.dataframe(results_df)
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.plot(nn.loss_history, linewidth=2, color='#1f77b4')
                ax.set_xlabel('Epochs', fontsize=12)
                ax.set_ylabel('Loss', fontsize=12)
                ax.set_title(f'Training Loss ({activation.upper()})', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

# Page 3: Performance Comparison
elif page == "Performance Comparison":
    st.markdown('<p class="sub-header">📈 Performance Comparison</p>', unsafe_allow_html=True)
    
    st.write("""
    Compare the performance of neural networks with different activation functions on the XOR problem.
    This helps understand which activation function converges faster and achieves better accuracy.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        epochs = st.slider("Training Epochs:", 1000, 10000, 5000, 1000)
    
    with col2:
        st.write("**Training Configuration:**")
        st.write("- Hidden Layer: 4 neurons")
        st.write("- Loss: Binary Cross-Entropy")
        st.write("- Optimizer: Gradient Descent")
    
    if st.button("🔬 Compare All Activation Functions", type="primary"):
        with st.spinner("Training networks with different activation functions..."):
            # Train with Sigmoid
            nn_sigmoid = SimpleNeuralNetwork(2, 4, 1, 'sigmoid')
            nn_sigmoid.train(X, y, epochs, 0.5)
            acc_sigmoid = accuracy_score(y, nn_sigmoid.predict(X))
            
            # Train with Tanh
            nn_tanh = SimpleNeuralNetwork(2, 4, 1, 'tanh')
            nn_tanh.train(X, y, epochs, 0.5)
            acc_tanh = accuracy_score(y, nn_tanh.predict(X))
            
            # Train with ReLU
            nn_relu = SimpleNeuralNetwork(2, 4, 1, 'relu')
            nn_relu.train(X, y, epochs, 0.1)
            acc_relu = accuracy_score(y, nn_relu.predict(X))
            
            # Display metrics
            st.markdown('<p class="sub-header">🎯 Accuracy Results</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sigmoid", f"{acc_sigmoid*100:.2f}%")
            with col2:
                st.metric("Tanh", f"{acc_tanh*100:.2f}%")
            with col3:
                st.metric("ReLU", f"{acc_relu*100:.2f}%")
            
            # Loss curves
            st.markdown('<p class="sub-header">📉 Training Loss Curves</p>', unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(nn_sigmoid.loss_history, label='Sigmoid', linewidth=2, alpha=0.8)
            ax.plot(nn_tanh.loss_history, label='Tanh', linewidth=2, alpha=0.8)
            ax.plot(nn_relu.loss_history, label='ReLU', linewidth=2, alpha=0.8)
            ax.set_xlabel('Epochs', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            # Accuracy bar chart
            st.markdown('<p class="sub-header">📊 Accuracy Comparison</p>', unsafe_allow_html=True)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            activations = ['Sigmoid', 'Tanh', 'ReLU']
            accuracies = [acc_sigmoid * 100, acc_tanh * 100, acc_relu * 100]
            colors = ['#3498db', '#e74c3c', '#2ecc71']
            
            bars = ax.bar(activations, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
            ax.set_ylabel('Accuracy (%)', fontsize=12)
            ax.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
            ax.set_ylim([0, 110])
            ax.grid(axis='y', alpha=0.3)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
            
            st.pyplot(fig)
            
            # Summary
            st.markdown('<p class="sub-header">💡 Key Insights</p>', unsafe_allow_html=True)
            
            st.success("""
            **Observations:**
            
            1. **Convergence Speed:** Different activation functions converge at different rates
            2. **Final Accuracy:** All three can solve XOR when properly tuned
            3. **Stability:** ReLU often requires lower learning rates
            4. **Modern Practice:** ReLU is preferred for hidden layers in deep networks
            5. **Output Layer:** Sigmoid is standard for binary classification
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>🧠 Neural Networks Lab - Activation Functions Explorer</p>
    <p>Built with Streamlit | For Educational Purposes</p>
</div>
""", unsafe_allow_html=True)
