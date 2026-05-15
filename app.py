import streamlit as st 
import pandas as pd 
import numpy as np 
import pickle

st.title("House price Prediction!")



st.markdown(
    """
    <style>
    /* This targets the label text of the selectbox in the sidebar */
    section[data-testid="stSidebar"] .stSelectbox label p {
        font-size: 24px !important;
        font-weight: bold !important;
        color: #ff4b4b; /* Optional: change color too */
    }

    /* This targets the actual text inside the selection box */
    section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
        font-size: 18px !important;
        font-weight: normal; 
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.sidebar: 
    options = ['Prediction'] 
    option = st.selectbox("Menu", options, width=500) 

if option == 'Prediction':
    
    
    CRIM = st.number_input("Enter the CRIM ")
    ZN = st.number_input("Enter the ZN ")
    INDUS = st.number_input("Enter the INDUS")
    CHAS = st.number_input("Enter the CHAS ")
    NOX = st.number_input("Enter the NOX ")
    RM = st.number_input("Enter the RM ")
    AGE = st.number_input("Enter the AGE ")
    DIS = st.number_input("Enter the DIS ")
    RAD = st.number_input("Enter the RAD ")
    TAX = st.number_input("Enter the TAX")
    PTRATIO = st.number_input("Enter the PTRATIO ")
    B = st.number_input("Enter the B ")
    LSTAT = st.number_input("Enter the LSTAT ") 

    input_values = [CRIM, ZN,INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT] 
    input_values = np.array(input_values).reshape(1, -1) 
    
    # loading the model 
    with open("C:/guvi/project_7/house_price.venv/model_regression.pkl", "rb") as f:
        model = pickle.load(f)  
    with open("C:/guvi/project_7/house_price.venv/scaler.pkl", "rb") as f: 
        scaler = pickle.load(f) 
    if st.button("predict"):
        result = model.predict(scaler.transform(input_values))
        st.write(result)



