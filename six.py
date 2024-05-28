import pandas as pd
import streamlit as st
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load the dataset
data = pd.read_csv('who_covid_data.csv')

# Display the first few rows of the dataset for verification
st.write("Dataset Preview")
st.write(data.head())

# Define the structure of the Bayesian Network
model = BayesianNetwork([
    ('Fever', 'COVID'),
    ('Cough', 'COVID'),
    ('Fatigue', 'COVID'),
    ('Difficulty_Breathing', 'COVID'),
    ('Sore_Throat', 'COVID'),
    ('Loss_of_Taste_or_Smell', 'COVID')
])

# Fit the model using Maximum Likelihood Estimator
model.fit(data, estimator=MaximumLikelihoodEstimator)

# Inference
inference = VariableElimination(model)

# Streamlit app
st.title('COVID-19 Diagnosis using Bayesian Network')

st.write('Enter the symptoms to diagnose the possibility of COVID-19:')

fever = st.selectbox('Fever', ['Yes', 'No'])
cough = st.selectbox('Cough', ['Yes', 'No'])
fatigue = st.selectbox('Fatigue', ['Yes', 'No'])
difficulty_breathing = st.selectbox('Difficulty Breathing', ['Yes', 'No'])
sore_throat = st.selectbox('Sore Throat', ['Yes', 'No'])
loss_of_taste_or_smell = st.selectbox('Loss of Taste or Smell', ['Yes', 'No'])

# Map input to numerical values
symptom_mapping = {'Yes': 1, 'No': 0}
fever_code = symptom_mapping[fever]
cough_code = symptom_mapping[cough]
fatigue_code = symptom_mapping[fatigue]
difficulty_breathing_code = symptom_mapping[difficulty_breathing]
sore_throat_code = symptom_mapping[sore_throat]
loss_of_taste_or_smell_code = symptom_mapping[loss_of_taste_or_smell]

# Perform inference
if st.button('Diagnose'):
    evidence = {
        'Fever': fever_code,
        'Cough': cough_code,
        'Fatigue': fatigue_code,
        'Difficulty_Breathing': difficulty_breathing_code,
        'Sore_Throat': sore_throat_code,
        'Loss_of_Taste_or_Smell': loss_of_taste_or_smell_code
    }
    result = inference.map_query(variables=['COVID'], evidence=evidence)
    diagnosis = 'Positive' if result['COVID'] == 1 else 'Negative'
    st.write(f'The diagnosis for COVID-19 is: **{diagnosis}**')

# Main execution for running in VS Code
if __name__ == "__main__":
    # This will run the Streamlit app when executing the script directly
    import os
    os.system('streamlit run ' + __file__)
