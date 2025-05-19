import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import pickle

# Load the dataset
cy = pd.read_csv(r"C:\Users\raori\Downloads\crop-prediction\Crop-Yield-Prediction-Using-Machine-Learning-main\dataset\yield_df.csv")
cy.drop('Unnamed: 0', axis=1, inplace=True)

# Data Cleaning
cy.drop_duplicates(inplace=True)

# Data Visualization
plt.figure(figsize=(15,20))
sns.countplot(y = cy['Area'])
plt.show()

plt.figure(figsize=(15,20))
sns.countplot(y = cy['Item'])
plt.show()

# Yield per country and crop visualizations
country = cy['Area'].unique()
yield_per_country = [cy[cy['Area'] == state]['hg/ha_yield'].sum() for state in country]

plt.figure(figsize = (15, 20))
sns.barplot(y = country, x = yield_per_country)
plt.show()

crops = cy['Item'].unique()
yield_per_crop = [cy[cy['Item'] == crop]['hg/ha_yield'].sum() for crop in crops]

plt.figure(figsize = (15, 20))
sns.barplot(y = crops, x = yield_per_crop)
plt.show()

# Preparing Data for Training
cy = cy[['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp', 'Area', 'Item', 'hg/ha_yield']]
X = cy.drop('hg/ha_yield', axis=1)
y = cy['hg/ha_yield']

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=True)

# Preprocessing (OneHotEncoding + Standard Scaling)
ohe = OneHotEncoder(drop='first')
scale = StandardScaler()

preprocesser = ColumnTransformer(
    transformers=[
        ('StandardScale', scale, [0, 1, 2, 3]),  # Scale numerical features
        ('OneHotEncode', ohe, [4, 5])            # One-hot encode categorical features
    ],
    remainder='passthrough'
)

X_train_dummy = preprocesser.fit_transform(X_train)
X_test_dummy = preprocesser.transform(X_test)

# Model Training and Evaluation
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'Decision Tree': DecisionTreeRegressor(),
    'KNN': KNeighborsRegressor(),
}

# Function to evaluate models
def evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"Train MAE of {model_name}: {train_mae:.4f}, Train R²: {train_r2:.4f}")
    print(f"Test MAE of {model_name}: {test_mae:.4f}, Test R²: {test_r2:.4f}")
    print("-" * 50)

# Evaluate all models
for name, model in models.items():
    evaluate_model(model, name, X_train_dummy, y_train, X_test_dummy, y_test)

# Save the best models
dtr = DecisionTreeRegressor()
dtr.fit(X_train_dummy, y_train)

knn = KNeighborsRegressor()
knn.fit(X_train_dummy, y_train)

# Save the models using pickle
pickle.dump(dtr, open(r"C:\Users\raori\Downloads\crop-prediction\Crop-Yield-Prediction-Using-Machine-Learning-main\models\dtr_model.pkl", "wb"))
pickle.dump(knn, open(r"C:\Users\raori\Downloads\crop-prediction\Crop-Yield-Prediction-Using-Machine-Learning-main\models\knn_model.pkl", "wb"))
pickle.dump(preprocesser, open(r"C:\Users\raori\Downloads\crop-prediction\Crop-Yield-Prediction-Using-Machine-Learning-main\models\preprocesser.pkl", "wb"))

# Prediction function using the trained model
def prediction(Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item):
    features = np.array([[Year, average_rain_fall_mm_per_year, pesticides_tonnes, avg_temp, Area, Item]], dtype=object)
    transformed_features = preprocesser.transform(features)
    predicted_yield = dtr.predict(transformed_features).reshape(-1, 1)
    return predicted_yield[0][0]

# Example prediction
dtr_result = prediction(1990, 1485.0, 121.0, 16.37, 'Albania', 'Maize')
knn_result = prediction(1990, 1485.0, 121.0, 16.37, 'Albania', 'Maize')

print(f"Decision Tree Prediction: {dtr_result}")
print(f"KNN Prediction: {knn_result}")
