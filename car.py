# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
data = {
    'Mileage': [15000, 25000, 30000, 35000, 45000, 50000, 60000, 75000, 80000, 95000],
    'Age': [1, 2, 2, 3, 3, 4, 5, 5, 6, 7],
    'Price': [20000, 18000, 16000, 15000, 14000, 13000, 12000, 10000, 9000, 8000]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define features and target
X = df[['Mileage', 'Age']]
y = df['Price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)

# Streamlit app
st.title("Car Price Predictor")

# Display dataset
st.write("### Sample Dataset")
st.write(df)

# Input fields for features
mileage = st.number_input("Enter Mileage:", min_value=0)
age = st.number_input("Enter Age of the car:", min_value=0)

# Predict button
if st.button("Predict Price"):
    # Create a dataframe for the input
    input_data = pd.DataFrame([[mileage, age]], columns=['Mileage', 'Age'])
    
    # Make prediction
    prediction = model.predict(input_data)[0]
    
    # Display the prediction
    st.write(f"The predicted price of the car is: ${prediction:.2f}")

# Display model evaluation
st.write("### Model Evaluation")
st.write(f"Mean Squared Error: {mse:.2f}")

