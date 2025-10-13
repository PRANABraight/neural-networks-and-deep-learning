import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Perceptron:
    """Single Layer Perceptron"""
    
    def __init__(self, n_features, learning_rate=0.1, random_weights=True):
        self.lr = learning_rate
        if random_weights:
            self.weights = np.random.randn(n_features)
            self.bias = np.random.randn()
        else:
            self.weights = np.zeros(n_features)
            self.bias = 0.0
    
    def _step_activation(self, z):
        return 1 if z >= 0 else 0
    
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        return np.array([self._step_activation(val) for val in z])
    
    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            error_count = 0
            for xi, target in zip(X, y):
                prediction = self.predict(xi.reshape(1, -1))[0]
                error = target - prediction
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    error_count += 1
            if error_count == 0:
                break

# Pre-train models
@st.cache_resource
def load_models():
    models = {}
    
    # AND Gate
    X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and = np.array([0, 0, 0, 1])
    np.random.seed(42)
    and_model = Perceptron(n_features=2, learning_rate=0.1)
    and_model.fit(X_and, y_and)
    models['AND'] = and_model
    
    # OR Gate
    X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_or = np.array([0, 1, 1, 1])
    np.random.seed(42)
    or_model = Perceptron(n_features=2, learning_rate=0.1)
    or_model.fit(X_or, y_or)
    models['OR'] = or_model
    
    # AND-NOT Gate
    X_and_not = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_and_not = np.array([0, 0, 1, 0])
    np.random.seed(42)
    and_not_model = Perceptron(n_features=2, learning_rate=0.1)
    and_not_model.fit(X_and_not, y_and_not)
    models['AND-NOT'] = and_not_model
    
    # XOR Gate (Note: Single perceptron cannot learn XOR perfectly)
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])
    np.random.seed(42)
    xor_model = Perceptron(n_features=2, learning_rate=0.1)
    xor_model.fit(X_xor, y_xor)  # This won't converge perfectly
    models['XOR'] = xor_model
    
    return models

# Streamlit UI
st.set_page_config(page_title="Logic Gate Perceptron", page_icon="🧠", layout="centered")

st.title("🧠 Single Layer Perceptron: Logic Gates")
st.markdown("### Interactive demonstration of learned logic gates")

# Load pre-trained models
models = load_models()

# Gate selection
col1, col2 = st.columns([2, 1])
with col1:
    selected_gate = st.selectbox(
        "Select Logic Gate:",
        options=["AND", "OR", "AND-NOT", "XOR"],
        help="Choose a gate to test"
    )
    
    if selected_gate == "XOR":
        st.warning("⚠️ Note: A single perceptron cannot perfectly learn the XOR function as it is not linearly separable. This demonstrates a limitation of single-layer perceptrons.")

# Input toggles
st.markdown("---")
st.markdown("#### Input Values")
col_a, col_b = st.columns(2)

with col_a:
    input_a = st.toggle("Input A", value=False)
    st.markdown(f"**Value: {1 if input_a else 0}**")

with col_b:
    input_b = st.toggle("Input B", value=False)
    st.markdown(f"**Value: {1 if input_b else 0}**")

# Visualization functions
def plot_decision_boundary(model, gate_name):
    x_min, x_max = -0.5, 1.5
    y_min, y_max = -0.5, 1.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter([0, 0, 1, 1], [0, 1, 0, 1], 
              c=['red', 'red', 'red', 'green'] if gate_name == 'AND' else
                ['red', 'green', 'green', 'green'] if gate_name == 'OR' else
                ['red', 'red', 'green', 'red'] if gate_name == 'AND-NOT' else
                ['red', 'green', 'green', 'red'],  # XOR
              marker='o', s=100)
    
    ax.set_xlabel('Input A')
    ax.set_ylabel('Input B')
    ax.set_title(f'Decision Boundary for {gate_name} Gate')
    ax.grid(True)
    return fig

def create_truth_table(gate_name):
    data = {
        'Input A': [0, 0, 1, 1],
        'Input B': [0, 1, 0, 1],
        'Output': [0, 0, 0, 1] if gate_name == 'AND' else
                 [0, 1, 1, 1] if gate_name == 'OR' else
                 [0, 0, 1, 0] if gate_name == 'AND-NOT' else
                 [0, 1, 1, 0]  # XOR
    }
    df = pd.DataFrame(data)
    return df

def plot_weights(model, gate_name):
    weights = np.append(model.weights, model.bias)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=['Weight 1', 'Weight 2', 'Bias'], y=weights, ax=ax)
    ax.set_title(f'Weights and Bias for {gate_name} Gate')
    ax.set_ylabel('Value')
    return fig

# Visualizations
st.markdown("---")
st.markdown("### Visualizations")

col_viz1, col_viz2 = st.columns(2)

with col_viz1:
    st.markdown("#### Truth Table")
    truth_table = create_truth_table(selected_gate)
    st.dataframe(truth_table, use_container_width=True)
    
    st.markdown("#### Current Weights")
    weights_fig = plot_weights(models[selected_gate], selected_gate)
    st.pyplot(weights_fig)

with col_viz2:
    st.markdown("#### Decision Boundary")
    decision_fig = plot_decision_boundary(models[selected_gate], selected_gate)
    st.pyplot(decision_fig)

st.markdown("#### Circuit Representation")
st.markdown(f"""
```
     Input A ────┐
                 ├─[{selected_gate}]─── Output
     Input B ────┘
```
""")

# Prediction
st.markdown("---")
st.markdown("### Prediction")
X_input = np.array([[1 if input_a else 0, 1 if input_b else 0]])
prediction = models[selected_gate].predict(X_input)[0]

# Display result
st.markdown("### Prediction")
if prediction == 1:
    st.success(f"### {selected_gate} Output: **{prediction}** ✓", icon="✅")
else:
    st.info(f"### {selected_gate} Output: **{prediction}**", icon="ℹ️")

# Truth table reference
st.markdown("---")
st.markdown("#### Truth Table Reference")

truth_tables = {
    'AND': [(0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1)],
    'OR': [(0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1)],
    'AND-NOT': [(0, 0, 0), (0, 1, 0), (1, 0, 1), (1, 1, 0)]
}

table_data = truth_tables[selected_gate]
st.markdown(f"""
| Input A | Input B | Output |
|---------|---------|--------|
| {table_data[0][0]} | {table_data[0][1]} | {table_data[0][2]} |
| {table_data[1][0]} | {table_data[1][1]} | {table_data[1][2]} |
| {table_data[2][0]} | {table_data[2][1]} | {table_data[2][2]} |
| {table_data[3][0]} | {table_data[3][1]} | {table_data[3][2]} |
""")

# Model info
with st.expander("📊 Model Details"):
    model = models[selected_gate]
    st.markdown(f"""
    **Weights:** `{model.weights}`  
    **Bias:** `{model.bias:.4f}`  
    **Learning Rate:** `{model.lr}`
    """)
    st.info("This perceptron successfully learned the linearly separable pattern!")
