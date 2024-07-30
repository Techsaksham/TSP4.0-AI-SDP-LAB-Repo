import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load the trained model
model = joblib.load('iris_model.pkl')

# Load the Iris dataset for target names
iris = load_iris()

# Define the user interface
st.title("Iris Species Prediction")

st.write("""
This app predicts the **Iris species** based on the input features.
""")

# Input fields for the features
sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)
petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0)
petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=1.0)

# Predict button
if st.button('Predict'):
    # Create an array from the input
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make a prediction
    prediction = model.predict(input_data)

    # Display the prediction
    st.write(f'The predicted Iris species is: {iris.target_names[prediction][0]}')
