import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Load the saved model
@st.cache_resource
def load_model():
    with open('model/hypo.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    # Set page config
    st.set_page_config(
        page_title="Thyroid Disease Prediction",
        page_icon="🏥",
        layout="wide"
    )

    # Title and description
    st.title("Thyroid Disease Prediction System")
    st.write("Enter the patient's information to predict thyroid condition")

    # Define features in the exact order they were during training
    features = [
        'age', 'sex', 'on_thyroxine', 'query_on_thyroxine', 
        'on_antithyroid_medication', 'sick', 'pregnant', 
        'thyroid_surgery', 'I131_treatment', 'query_hypothyroid',
        'query_hyperthyroid', 'lithium', 'goitre', 'tumor', 
        'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI'
    ]

    # Create input fields
    input_data = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Numerical inputs
        input_data['age'] = st.number_input("Age", min_value=0, max_value=100, value=30)
        input_data['TSH'] = st.number_input("TSH Level", min_value=0.0, max_value=100.0, value=1.0)
        input_data['T3'] = st.number_input("T3 Level", min_value=0.0, max_value=100.0, value=1.0)
        input_data['TT4'] = st.number_input("TT4 Level", min_value=0.0, max_value=100.0, value=1.0)
        input_data['T4U'] = st.number_input("T4U Level", min_value=0.0, max_value=100.0, value=1.0)
        input_data['FTI'] = st.number_input("FTI Level", min_value=0.0, max_value=100.0, value=1.0)

    with col2:
        # Binary inputs
        input_data['sex'] = 1 if st.selectbox("Sex", ["Male", "Female"]) == "Male" else 0
        input_data['on_thyroxine'] = 1 if st.selectbox("On Thyroxine?", ["No", "Yes"]) == "Yes" else 0
        input_data['query_on_thyroxine'] = 1 if st.selectbox("Query on Thyroxine?", ["No", "Yes"]) == "Yes" else 0
        input_data['on_antithyroid_medication'] = 1 if st.selectbox("On Antithyroid Medication?", ["No", "Yes"]) == "Yes" else 0
        input_data['sick'] = 1 if st.selectbox("Sick?", ["No", "Yes"]) == "Yes" else 0
        input_data['pregnant'] = 1 if st.selectbox("Pregnant?", ["No", "Yes"]) == "Yes" else 0

    col3, col4 = st.columns(2)
    
    with col3:
        input_data['thyroid_surgery'] = 1 if st.selectbox("Thyroid Surgery?", ["No", "Yes"]) == "Yes" else 0
        input_data['I131_treatment'] = 1 if st.selectbox("I131 Treatment?", ["No", "Yes"]) == "Yes" else 0
        input_data['query_hypothyroid'] = 1 if st.selectbox("Query Hypothyroid?", ["No", "Yes"]) == "Yes" else 0
        input_data['query_hyperthyroid'] = 1 if st.selectbox("Query Hyperthyroid?", ["No", "Yes"]) == "Yes" else 0

    with col4:
        input_data['lithium'] = 1 if st.selectbox("On Lithium?", ["No", "Yes"]) == "Yes" else 0
        input_data['goitre'] = 1 if st.selectbox("Goitre?", ["No", "Yes"]) == "Yes" else 0
        input_data['tumor'] = 1 if st.selectbox("Tumor?", ["No", "Yes"]) == "Yes" else 0
        input_data['hypopituitary'] = 1 if st.selectbox("Hypopituitary?", ["No", "Yes"]) == "Yes" else 0
        input_data['psych'] = 1 if st.selectbox("Psych?", ["No", "Yes"]) == "Yes" else 0

    # Prediction button
    if st.button("Predict Thyroid Condition"):
        try:
            # Load model
            model = load_model()
            
            # Create DataFrame with features in correct order
            df = pd.DataFrame([input_data])
            df = df[features]  # Ensure correct feature order
            
            # Scale numerical features
            scaler = StandardScaler()
            numerical_cols = ['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            
            # Make prediction
            prediction = model.predict(df)[0]
            
            # Map prediction to class
            prediction_map = {
                0: "Negative (No Thyroid Disease)",
                1: "Compensated Hypothyroid",
                2: "Primary Hypothyroid",
                3: "Secondary Hypothyroid"
            }
            
            # Show prediction
            st.success(f"Predicted Condition: {prediction_map[prediction]}")
            
            # Show prediction probability
            probabilities = model.predict_proba(df)[0]
            st.write("Prediction Probabilities:")
            for i, prob in enumerate(probabilities):
                st.write(f"{prediction_map[i]}: {prob:.2%}")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.write("Please check the model file and feature order.")

if __name__ == "__main__":
    main()