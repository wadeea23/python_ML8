import joblib
import pandas as pd
import streamlit as st

model = joblib.load('car_model_simple.pkl')
st.title("Car Price Prediction")
st.write("Enter the car details")
hp = st.number_input("Engine HP", min_value=0, value=150)
cylinders = st.number_input("Engine Cylinders", min_value=1, value=4)
year = st.selectbox("Year", range(1990, 2027), index=30)
if st.button("Predict"):
    data = pd.DataFrame({
        'Engine HP': [hp],
        'Engine Cylinders': [cylinders],
        'Year': [year]})

    price = model.predict(data)[0]
    st.success(f"Predicted Car Price: ${price:,.2f}")
