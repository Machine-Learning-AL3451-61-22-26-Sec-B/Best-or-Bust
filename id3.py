pip install scikit-learn streamlit
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Define function to predict class labels for new samples
def predict_class(sample):
    prediction = clf.predict([sample])
    return iris.target_names[prediction[0]]

# Streamlit web application
def main():
    st.title("ID3 Decision Tree Classifier Demo")

    st.write("""
    ## Iris Flower Classification
    This app predicts the Iris flower species based on its features using the ID3 decision tree algorithm.
    """)

    # User input for new sample features
    st.sidebar.header('Input Features')
    sepal_length = st.sidebar.slider('Sepal Length', float(X[:, 0].min()), float(X[:, 0].max()), float(X[:, 0].mean()))
    sepal_width = st.sidebar.slider('Sepal Width', float(X[:, 1].min()), float(X[:, 1].max()), float(X[:, 1].mean()))
    petal_length = st.sidebar.slider('Petal Length', float(X[:, 2].min()), float(X[:, 2].max()), float(X[:, 2].mean()))
    petal_width = st.sidebar.slider('Petal Width', float(X[:, 3].min()), float(X[:, 3].max()), float(X[:, 3].mean()))

    # Display the input features
    st.write('Input Features:')
    st.write('Sepal Length:', sepal_length)
    st.write('Sepal Width:', sepal_width)
    st.write('Petal Length:', petal_length)
    st.write('Petal Width:', petal_width)

    # Predict the class label for the new sample
    sample = [sepal_length, sepal_width, petal_length, petal_width]
    prediction = predict_class(sample)

    # Display the predicted class label
    st.write('Predicted Class Label:', prediction)

if __name__ == '__main__':
    main()
