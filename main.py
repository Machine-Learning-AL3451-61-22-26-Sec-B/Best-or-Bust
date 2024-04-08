import streamlit as st
import pandas as pd
import numpy as np

class Node:
    def __init__(self, attribute):
        self.attribute = attribute
        self.children = {}
        self.prediction = None

def entropy(y):
    classes, counts = np.unique(y, return_counts=True)
    entropy = 0
    total_samples = len(y)
    for count in counts:
        p = count / total_samples
        entropy -= p * np.log2(p)
    return entropy

def information_gain(X, y, attribute):
    attribute_values = np.unique(X[attribute])
    total_entropy = entropy(y)
    weighted_entropy = 0
    for value in attribute_values:
        subset_indices = X[attribute] == value
        subset_entropy = entropy(y[subset_indices])
        weight = np.sum(subset_indices) / len(y)
        weighted_entropy += weight * subset_entropy
    return total_entropy - weighted_entropy

def id3(X, y, attributes):
    if len(np.unique(y)) == 1:
        node = Node(None)
        node.prediction = y[0]
        return node
    if len(attributes) == 0:
        node = Node(None)
        node.prediction = np.argmax(np.bincount(y))
        return node
    gains = [information_gain(X, y, attribute) for attribute in attributes]
    best_attribute_index = np.argmax(gains)
    best_attribute = attributes[best_attribute_index]
    node = Node(best_attribute)
    attribute_values = np.unique(X[best_attribute])
    for value in attribute_values:
        subset_indices = X[best_attribute] == value
        subset_X = X[subset_indices].drop(columns=[best_attribute])
        subset_y = y[subset_indices]
        if len(subset_y) == 0:
            node.prediction = np.argmax(np.bincount(y))
            return node
        else:
            new_attributes = attributes.copy()
            new_attributes.remove(best_attribute)
            node.children[value] = id3(subset_X, subset_y, new_attributes)
    return node

def predict(root, sample):
    if root.prediction is not None:
        return root.prediction
    attribute_value = sample[root.attribute]
    if attribute_value not in root.children:
        return root.prediction
    return predict(root.children[attribute_value], sample)

def main():
    st.title("ID3 Algorithm with Streamlit")

    st.sidebar.header("Upload Data")
    uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Sample:")
        st.write(data.head())

        st.sidebar.header("Attributes")
        attributes = st.sidebar.multiselect("Select attributes", data.columns)

        target_attribute = st.sidebar.selectbox("Select target attribute", data.columns)

        if st.sidebar.button("Run ID3 Algorithm"):
            X = data[attributes]
            y = data[target_attribute]

            root = id3(X, y, attributes)

            st.write("Decision Tree:")
            st.write(root)

            st.write("Predictions:")
            for i, row in data.iterrows():
                prediction = predict(root, row)
                st.write(f"Sample {i+1}: Predicted class - {prediction}")

if __name__ == "__main__":
    main()
