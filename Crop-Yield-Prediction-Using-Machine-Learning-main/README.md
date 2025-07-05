# Crop-Yield-Prediction-Using-Machine-Learning

## Project Summary
This project focuses on building a predictive system for agricultural crop yield using machine learning models. By leveraging historical data on weather, pesticides, and other environmental factors, this project aims to estimate crop yields for various regions and crop types. The models developed include Linear Regression, Lasso, Ridge, Decision Tree, and K-Nearest Neighbors. The dataset used comes from historical agricultural data, including crop yield, rainfall, and temperature, and covers various countries and crops.

## Objectives
1. **Data Cleaning & Exploration:** Remove missing values, duplicates, and irrelevant columns to prepare the dataset for modeling.
   [Dataset Link](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)
3. **Modeling & Prediction:** Train and evaluate several regression models, including:
   - Linear Regression
   - Lasso Regression
   - Ridge Regression
   - Decision Tree Regressor
   - K-Nearest Neighbors Regressor
4. **Performance Comparison:** Compare the performance of these models using metrics like Mean Absolute Error (MAE) and R² Score.
5. **Predictive System:** Develop a prediction system that allows users to input environmental and crop data to predict crop yield for specific regions.
6. **Save Models:** Save the best-performing models and preprocessing steps using Python's `pickle` library for future use.

## Workflow
1. **Data Preprocessing:**
   - Imported the dataset and dropped irrelevant columns.
   - Handled missing values and removed duplicates.
   - Encoded categorical features (Area and Item) using OneHotEncoder and scaled numerical features with StandardScaler.

2. **Model Training & Evaluation:**
   - Split the data into training and test sets.
   - Trained five different regression models.
   - Evaluated model performance based on MAE and R² Score.
   - Visualized the model performance for comparison.

3. **Prediction System:**
   - Created a function that allows predictions based on user inputs (year, rainfall, pesticide usage, temperature, area, and crop type).
   - Used Decision Tree and K-Nearest Neighbors as the main models for prediction.

4. **Model Saving:**
   - Saved the trained models (`dtr_model.pkl` and `knn_model.pkl`) and the preprocessing pipeline (`preprocesser.pkl`) using `pickle`.
  
## Snapshot
![model comperasion](https://github.com/user-attachments/assets/e771c9cd-e2c3-46c0-bf6a-a897753688e6)

![user interface]()

## Conclusion
The Decision Tree Regressor and K-Nearest Neighbors showed the most promise in accurately predicting crop yield based on historical data. The project demonstrates the utility of machine learning in agricultural yield prediction, which can assist farmers, policymakers, and agronomists in making informed decisions about crop production.

## Future Work
1. **Model Improvement:** Experiment with more advanced models such as Random Forest, Gradient Boosting, or Neural Networks to improve prediction accuracy.
2. **Feature Engineering:** Incorporate more features such as soil quality, water usage, and fertilizer data to enhance the model’s predictive power.
3. **Deployment:** Develop a web-based user interface (possibly using Streamlit) for end-users to easily input data and obtain yield predictions in real time.
4. **Time Series Analysis:** Introduce time series forecasting techniques to account for trends and seasonality in yield prediction.
