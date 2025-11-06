# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and load the dataset.

2. Split the dataset into training and testing sets.

3. Train the model using the SGD regressor.

4. Predict and evaluate the model performance.
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: KISHORE V
RegisterNumber:212224240077
*/
# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

# Sample dataset
data = {
    'Area': [1500, 1800, 2400, 3000, 3500, 4000, 4200, 5000],
    'Bedrooms': [3, 4, 3, 5, 4, 6, 5, 7],
    'Age': [10, 15, 20, 8, 12, 5, 7, 3],
    'Price': [300000, 400000, 500000, 600000, 650000, 700000, 720000, 800000],
    'Occupants': [4, 5, 4, 6, 5, 7, 6, 8]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define features (X) and targets (y)
X = df[['Area', 'Bedrooms', 'Age']]
y = df[['Price', 'Occupants']]  # Two target variables

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize SGD Regressor (for multiple outputs)
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
model = MultiOutputRegressor(sgd)

# Train the model
model.fit(X_train_scaled, y_train)

# Predict the outputs
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
print("Predicted [Price, Occupants]:\n", y_pred)
print("\nMean Squared Error:", mse)
print("R² Score:", r2)
```

## Output:
<img width="444" height="158" alt="Screenshot 2025-10-10 141011" src="https://github.com/user-attachments/assets/7a28a618-1bd2-4c96-b993-942d1fd2d5b0" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
