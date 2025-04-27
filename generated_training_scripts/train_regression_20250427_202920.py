# [MODEL TRAINING]
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Assume data is loaded into a pandas DataFrame called 'df'
# with columns 'Model Year', 'Base MSRP', 'Latitude', 'Longitude', 'Electric Range'

# Sample data (replace with your actual data loading)
data = {'Model Year': [2020, 2021, 2022, 2023, 2024],
        'Base MSRP': [30000, 35000, 40000, 45000, 50000],
        'Latitude': [34.0522, 37.7749, 40.7128, 47.6062, 33.4484],
        'Longitude': [-118.2437, -122.4194, -74.0060, -122.3321, -112.0740],
        'Electric Range': [200, 250, 300, 350, 400]}
df = pd.DataFrame(data)


# Define features and target
features = ['Model Year', 'Base MSRP', 'Latitude', 'Longitude']
target = 'Electric Range'

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Store performance metrics
metrics = {'Mean Squared Error': mse, 'R-squared': r2}