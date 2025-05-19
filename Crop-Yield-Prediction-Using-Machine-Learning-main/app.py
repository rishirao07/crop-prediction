import streamlit as st
import pandas as pd
import pickle
from st_social_media_links import SocialMediaIcons

# Load the retrained models
dtr = pickle.load(open(r'C:\Users\raori\Downloads\crop-prediction\Crop-Yield-Prediction-Using-Machine-Learning-main\models\dtr_model.pkl', 'rb'))
preprocessor = pickle.load(open(r'C:\Users\raori\Downloads\crop-prediction\Crop-Yield-Prediction-Using-Machine-Learning-main\models\preprocesser.pkl', 'rb'))

# Streamlit app configuration
st.set_page_config(page_title="Agricultural Yield Prediction", layout="centered", page_icon="🌾")

# Title and description
st.title("🌾 Agricultural Yield Prediction")
st.markdown("""
This app predicts the yield based on various inputs such as average rainfall, pesticide use, temperature, area, and crop type.
""")

# Input fields
st.sidebar.header("Enter Input Features")

year = st.sidebar.number_input("Year", min_value=1900, max_value=2100, step=1, value=2024)
average_rainfall = st.sidebar.number_input("Average Rainfall (mm/year)", min_value=0.0, max_value=5000.0, step=0.1, value=1000.0)
pesticides_tonnes = st.sidebar.number_input("Pesticides Used (tonnes)", min_value=0.0, max_value=1000.0, step=0.1, value=50.0)
avg_temp = st.sidebar.number_input("Average Temperature (°C)", min_value=-50.0, max_value=60.0, step=0.1, value=25.0)
area = st.sidebar.selectbox("Area", [
    "India", "United States", "Brazil", "China", "Australia", "Nigeria", "Canada"
])
item = st.sidebar.selectbox("Item", [
    "Maize", "Potatoes", "Rice, paddy", "Sorghum", "Soybeans", "Wheat"
])

# Button to trigger prediction
if st.sidebar.button("Predict Yield"):
    # Prepare input features as a DataFrame
    features = pd.DataFrame([[year, average_rainfall, pesticides_tonnes, avg_temp, area, item]],
                            columns=['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item'])

    # Transform features using the preprocessor
    transformed_features = preprocessor.transform(features)
    prediction = dtr.predict(transformed_features).reshape(1, -1)

    # Display the prediction result
    st.success(f"🌱 The predicted agricultural yield is: {prediction[0][0]:.2f} tonnes")

