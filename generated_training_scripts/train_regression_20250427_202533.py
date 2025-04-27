# [MODEL TRAINING]
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Sample data (replace with your actual data loading)
data = {'Model Year': [2018, 2020, 2019, 2021, 2022],
        'Make': ['Toyota', 'Honda', 'Toyota', 'Ford', 'Tesla'],
        'Model': ['Camry', 'Civic', 'RAV4', 'F-150', 'Model 3'],
        'Vehicle Age': [5, 3, 4, 2, 1],
        'Electric Range': [300, 250, 320, 280, 350]}
df = pd.DataFrame(data)

# Preprocessing
le_make = LabelEncoder()
le_model = LabelEncoder()
df['Make'] = le_make.fit_transform(df['Make'])
df['Model'] = le_model.fit_transform(df['Model'])


# Features and target
features = ['Model Year', 'Make', 'Model', 'Vehicle Age']
target = 'Electric Range'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

metrics = {'mse': mse, 'r2': r2}