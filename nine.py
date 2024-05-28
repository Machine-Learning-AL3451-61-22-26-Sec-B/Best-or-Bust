import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def gaussian_kernel(x, xi, tau):
    return np.exp(- np.sum((x - xi) ** 2) / (2 * tau ** 2))

def locally_weighted_regression(x_query, X, y, tau):
    m, n = X.shape
    W = np.zeros((m, m))
    for i in range(m):
        W[i, i] = gaussian_kernel(X[i], x_query, tau)
    
    XTWX = X.T @ W @ X
    if np.linalg.det(XTWX) == 0.0:
        return np.linalg.pinv(XTWX) @ X.T @ W @ y
    theta = np.linalg.inv(XTWX) @ X.T @ W @ y
    return x_query @ theta

def predict(X_test, X_train, y_train, tau):
    y_pred = np.zeros(len(X_test))
    for i in range(len(X_test)):
        y_pred[i] = locally_weighted_regression(X_test[i], X_train, y_train, tau)
    return y_pred

# Streamlit app
st.title('Locally Weighted Regression')

# Parameters
tau = st.slider('Tau (bandwidth parameter)', min_value=0.01, max_value=1.0, value=0.5)

# Generate sample data
np.random.seed(0)
X = np.linspace(0, 10, 100)
y = np.sin(X) + np.random.normal(0, 0.1, 100)

# Add a bias term
X_bias = np.vstack([np.ones(len(X)), X]).T

# Test points
X_test = np.linspace(0, 10, 100)
X_test_bias = np.vstack([np.ones(len(X_test)), X_test]).T

# Fit and predict
y_pred = predict(X_test_bias, X_bias, y, tau)

# Plotting the results
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='Data')
ax.plot(X_test, y_pred, color='red', label='LWR Prediction')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Locally Weighted Regression')
ax.legend()

# Display the plot
st.pyplot(fig)

# Main execution for running in VS Code
if __name__ == "__main__":
    # This will run the Streamlit app when executing the script directly
    import os
    os.system('streamlit run ' + __file__)
