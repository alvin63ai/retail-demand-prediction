
import os
import numpy as np
import pandas as pd
import joblib

def load_model(model_path):

    model_rf = joblib.load(model_path)
    return model_rf

def prepare_features(data):

    X = data.loc[:, 'sales1':'sales12']
    X = X.copy() # aviod SettingWithCopyWarning
    X['Mean'] = data.loc[:, 'sales1':'sales12'].replace(0, np.nan).mean(axis=1)
    X['STD'] = data.loc[:, 'sales1':'sales12'].replace(0, np.nan).std(axis=1)
    X.fillna(0, inplace=True)
    return X

def predict(model, X):

    y_pred = model.predict(X)
    return y_pred

# function to find non-zero values
def find_non_zero_values(row):
    cols = data.loc[:, 'sales1':'sales12'].columns
    non_zero_values = [row[col] for col in cols if row[col] > 0]
    return non_zero_values

# Calculate the std and mean of the latest non-zero values
def calc_std(row):
    return np.std(row['NonZeroValues'][-6:])
def calc_mean(row):
    return np.mean(row['NonZeroValues'][-6:])

def prediction_check(data, y_pred):

    data['Prediction'] = y_pred

    # Apply the function to filter non-zero value to each row
    data['NonZeroValues'] = data.apply(find_non_zero_values, axis=1)

    # upper limit and lower limit
    data['MaxLimit'] = data.apply(calc_mean, axis=1) + data.apply(calc_std, axis=1) *2 # upper limit = mean + 2std
    data['MinLimit'] = data.apply(calc_mean, axis=1) # lower limit = mean

    # Find out if Model Predicted values are out of range
    data['Remark'] = np.where(data['Prediction'] > data['MaxLimit'], 'Over Estimate',
                              np.where(data['Prediction'] < data['MinLimit'], 'Under Estimate', ''))

    # Filter rows with 0 sold qty in 'sales1' to 'sales12' more than n times
    qty_data = data.loc[:, 'sales1':'sales12']
    n = 10
    df_zero_qty = data[(qty_data == 0).sum(axis=1) >= n]
    # Replace the model Estimate with 'Not enough data' for rows with 0 sold qty in 'sales1' to 'sales12' more than n times
    data.loc[df_zero_qty.index, 'Remark'] = 'Not enough data'

    # Remove the NonZeroValues column, L4MaxLimit, and L4MinLimit
    data.drop(['NonZeroValues', 'MaxLimit', 'MinLimit'], axis=1, inplace=True)
    
    return data

# Load the input data
data = pd.read_csv('input/new_sales_data.csv')

# Load the model
model_path = 'model/model_rf.pkl'
model_rf = load_model(model_path)

# preiction path
prediction_path = 'output/prediction.csv'
os.makedirs('output', exist_ok=True)

# Prepare features
X = prepare_features(data)

# Predict
y_pred = predict(model_rf, X)

# Prediction Check
data = prediction_check(data, y_pred)

# Print the completed message
print("Prediction completed.")

# Print the prediction
print(data.to_string())

# Save to csv
data.to_csv(prediction_path, index=False)
print(f"Saved to {prediction_path}")