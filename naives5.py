import streamlit as st
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load data
categories = ['alt.atheism', 'comp.graphics', 'sci.med', 'soc.religion.christian']
train_data = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
test_data = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

# Create a pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])

# Train the model
text_clf.fit(train_data.data, train_data.target)

# Predict the test set
predicted = text_clf.predict(test_data.data)

# Calculate metrics
accuracy = accuracy_score(test_data.target, predicted)
precision = precision_score(test_data.target, predicted, average='macro')
recall = recall_score(test_data.target, predicted, average='macro')

# Output metrics
st.write("Accuracy:", accuracy)
st.write("Precision:", precision)
st.write("Recall:", recall)

# User input for classification
st.header("Classify New Document")
user_input = st.text_area("Enter your document text here:")

if user_input:
    prediction = text_clf.predict([user_input])
    predicted_category = train_data.target_names[prediction[0]]
    st.write(f"Predicted Category: {predicted_category}")
