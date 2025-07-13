# preprocess.py
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import pickle

# Load data
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv("https://raw.githubusercontent.com/madhav481010/Diabetes-Prediction/main/diabetes.csv", names=columns, header=0)
print(df)
print('Diabetes-preprocess')
# Replace biologically implausible zeroes with NaN
na_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[na_columns] = df[na_columns].replace(0, np.nan)

# Separate features and labels
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Apply KNN imputation (after standardizing for distance-based algorithm)
scaler_before_impute = StandardScaler()
X_scaled_for_knn = scaler_before_impute.fit_transform(X)

# KNN Imputer (n_neighbors=5 is common)
imputer = KNNImputer(n_neighbors=5)
X_imputed_scaled = imputer.fit_transform(X_scaled_for_knn)

# Convert back to DataFrame with original column names
X_imputed = pd.DataFrame(scaler_before_impute.inverse_transform(X_imputed_scaled), columns=X.columns)

# Optional: clip values after imputation if necessary
X_imputed['BloodPressure'] = X_imputed['BloodPressure'].clip(lower=40, upper=140)
X_imputed['BMI'] = X_imputed['BMI'].clip(lower=15, upper=50)
X_imputed['Glucose'] = X_imputed['Glucose'].clip(lower=50, upper=200)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Final scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save processed data
os.makedirs('data', exist_ok=True)
with open('data/X_train.pkl', 'wb') as f: pickle.dump(X_train_scaled, f)
with open('data/X_test.pkl', 'wb') as f: pickle.dump(X_test_scaled, f)
with open('data/y_train.pkl', 'wb') as f: pickle.dump(y_train, f)
with open('data/y_test.pkl', 'wb') as f: pickle.dump(y_test, f)
with open('data/scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
with open('data/imputer.pkl', 'wb') as f: pickle.dump(imputer, f)

print("Enhanced preprocessing with KNN imputation completed.")
