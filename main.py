import streamlit as st
import pandas as pd

def candidate_elimination(data, target_attribute):
  """
  Implements the Candidate Elimination Algorithm (CEA)

  Args:
      data (pandas.DataFrame): The dataset containing attributes and target values.
      target_attribute (str): The name of the target attribute (class label).

  Returns:
      tuple: A tuple containing final specific hypothesis as a list and final general hypothesis as a list of lists.
  """

  # Initialize specific and general hypotheses
  specific_h = list(data.iloc[0, :])
  general_h = [['?' for _ in range(len(specific_h))] for _ in range(len(specific_h))]

  for i, row in data.iterrows():
    # Update specific hypothesis for positive examples
    if row[target_attribute] == 'Yes':
      for j in range(len(specific_h)):
        if row.iloc[j] != specific_h[j]:
          specific_h[j] = '?'
    # Update general hypothesis for negative examples
    else:
      for j in range(len(specific_h)):
        if row.iloc[j] != specific_h[j]:
          general_h[j][j] = specific_h[j]
        else:
          general_h[j][j] = '?'

  return specific_h, general_h

def main():
  """
  Streamlit application entry point
  """

  st.title("Candidate Elimination Algorithm")

  # Upload dataset
  uploaded_file = st.file_uploader("Upload dataset (CSV)", type=['csv'])

  if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Select target attribute
    target_attribute = st.selectbox("Select target attribute", data.columns)

    # Run CEA and display results
    if st.button("Run Candidate Elimination"):
      final_specific_h, final_general_h = candidate_elimination(data.copy(), target_attribute)

      st.subheader("Final Specific Hypothesis:")
      st.write(final_specific_h)

      st.subheader("Final General Hypothesis:")
      for row in final_general_h:
        st.write(row)

if __name__ == '__main__':
  main()
