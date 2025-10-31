import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# Create a dataset of 1000 random road accidents
n_accidents = 1000
data = {
    'Weather': np.random.choice([1, 2, 3, 4], n_accidents),
    'Road_Type': np.random.choice([1, 2, 3], n_accidents),
    'Vehicle_Type': np.random.choice([1, 2, 3, 4], n_accidents),
    'Time_of_Day': np.random.choice([1, 2, 3, 4], n_accidents),
    'Speed_Limit': np.random.randint(30, 120, n_accidents),
    'Driver_Age': np.random.randint(18, 75, n_accidents),
    'Alcohol_Involved': np.random.choice([0, 1], n_accidents),
    'Road_Surface': np.random.choice([1, 2, 3], n_accidents)
}

df = pd.DataFrame(data)

# Calculate accident severity using a simple formula with some randomness
df['Severity'] = (
    df['Weather'] * 0.35 +
    df['Road_Type'] * 0.25 +
    df['Vehicle_Type'] * 0.4 +
    df['Time_of_Day'] * 0.2 +
    (df['Speed_Limit'] / 45) * 0.45 +
    ((75 - df['Driver_Age']) / 22) * 0.25 +
    df['Alcohol_Involved'] * 1.1 +
    df['Road_Surface'] * 0.3 +
    np.random.normal(0, 0.45, n_accidents)
)

df['Severity'] = np.clip(df['Severity'], 1, 5)

# Quick look at the dataset
print("Sample of the dataset:")
print(df.head())

# Split features and target
X = df.drop('Severity', axis=1)
y = df['Severity']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Show model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:\nMSE: {mse:.3f}, R²: {r2:.3f}")

# Save the model for later use
with open('road_accident_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Create a random example accident for prediction
example_accident = pd.DataFrame({
    'Weather': [np.random.randint(1, 5)],
    'Road_Type': [np.random.randint(1, 4)],
    'Vehicle_Type': [np.random.randint(1, 5)],
    'Time_of_Day': [np.random.randint(1, 5)],
    'Speed_Limit': [np.random.randint(30, 120)],
    'Driver_Age': [np.random.randint(18, 75)],
    'Alcohol_Involved': [np.random.randint(0, 2)],
    'Road_Surface': [np.random.randint(1, 4)]
})

# Load model and predict
with open('road_accident_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

predicted_severity = loaded_model.predict(example_accident)[0]

# Interpret severity
severity_labels = {1: "Minor", 2: "Moderate", 3: "Serious", 4: "Severe", 5: "Fatal"}
closest_level = min(severity_labels.keys(), key=lambda x: abs(x - predicted_severity))

print("\nExample Accident Prediction:")
print(example_accident)
print(f"Predicted Severity Score: {predicted_severity:.2f}")
print(f"Interpreted as: {severity_labels[closest_level]} accident")
