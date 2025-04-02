import os
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# set proper working directory if not yet, and change new_directory to wherever your repo is locally
new_directory = r"C:\Users\parma\data-science-guest-experience\data-science-guest-experience\Scripts\Subgroup_B\modeling" 
os.chdir(new_directory)

# Define relative paths
df_combined_path = "../../../data/processed/survey_and_weather_processed_data.pkl"
df_all_combined_path = "../../../data/processed/survey_iot_weather_processed_data.pkl"
df_iot_path = "../../../data/processed/iot_data.pkl"

# Load the DataFrames
df_combined_processed = pd.read_pickle(df_combined_path)
df_all_combined_processed = pd.read_pickle(df_all_combined_path)
df_iot = pd.read_pickle(df_iot_path)

# Verify loading
print(df_combined_processed.head())
print(df_all_combined_processed.head())
print(df_iot.head())

## Modelling with XGBoost
"""
Purpose:
Trains an XGBoost regression model on the given dataset to predict the specified target variable (default: 'Avg_Wait_Time'). 
The function returns the trained model and evaluates its performance using common regression metrics (RMSE and MAE).
 Additionally, it calculates and prints the correlation between the predicted target and the input features.

Arguments:
- df (pd.DataFrame): The dataset to train the model on. This dataset should include both features and the target variable. 
                    The features may include categorical variables such as Favorite_Attraction, Age_Group, Employment_Status, and more.
- target (str): The column name of the target variable that the model is trying to predict. 
                By default, this is set to 'Avg_Wait_Time', but it can be changed to any other column in the dataset (e.g., Queue_Time).

Returns:
- model (XGBRegressor): The trained XGBoost regression model.
- metrics (dict): A dictionary containing evaluation metrics for the model, specifically:
- RMSE: Root Mean Squared Error (RMSE) for model performance evaluation.
- MAE: Mean Absolute Error (MAE) for model performance evaluation.
"""

def train_demand_model(df, target='Avg_Wait_Time'):
    # define feature columns based on the dataset
    if 'theme_Zone_Visited' in df.columns: # if IoT data is present
        features = [
            'Favorite_Attraction', 'Satisfaction_Score', 'Age_Group', 'Employment_Status',
            'Visit_Quarter', 'Event', 'Attraction_Reason', 'Season', 'rainfall', 'air_temperature',
            'relative_humidity', 'wind_speed','Visitor_ID', 'Loyalty_Member', 'Age', 'Gender',
            'Theme_Zone_Visited', 'Attraction', 'Check_In', 'Queue_Time', 'Check_Out', 'Restaurant_Spending',
            'Merchandise_Spending', 'Total_Spending', 'Day_of_Week', 'Is_Weekend', 'Is_Popular_Attraction',
            'Year', 'Month'
        ]
    else:
        features = [ # for survey-only data
            'Favorite_Attraction', 'Satisfaction_Score', 'Age_Group', 'Employment_Status',
            'Visit_Quarter', 'Event', 'Attraction_Reason', 'Season', 'rainfall', 'air_temperature',
            'relative_humidity', 'wind_speed'
        ]

    df = df[features + [target]]

    # Define features and target
    X = df[features]
    y = df[target]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize XGBoost model
    model = XGBRegressor(
        random_state=42,
        n_estimators=500,
        max_depth=4,
        learning_rate=0.1,
        verbosity=0
    )
    
    # cross-validation
    cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validation negative MSE scores: {cross_val_scores}")
    print(f"Mean cross-validation score: {np.mean(cross_val_scores)}")

    # fit model on training data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # calculate evaluation metrics
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred)
    }

    print("Model trained successfully.")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")

    # create a DataFrame with predicted values and features from X_test
    df_test = X_test.copy()
    df_test['Predicted_' + target] = y_pred

    # calculate correlation between predicted values and features
    correlation = df_test.corr()['Predicted_' + target].sort_values(ascending=False)

    print("\nCorrelation with Predicted " + target + ":")
    print(correlation)

    return model, metrics

## Training model without the IoT data
## Evaluation of model training with the merged survey and weather data yields reasonable prediction with low RMSE and MAE.
model_1, metrics_1 = train_demand_model(df_combined_processed)

## Training model with IOT data
## Evaluation of model training with IOT data shows that model performs better when IoT data is involved.
model_2, metrics_2 = train_demand_model(df_all_combined_processed)

## XGBoost Modelling specifically for IoT Data only

def train_demand_model_2(df, target='Average_Queue_Time'):
    # feature columns
    features = [
            'Visitor_ID', 'Loyalty_Member', 'Age', 'Gender',
            'Theme_Zone_Visited', 'Attraction', 'Check_In', 'Queue_Time', 'Check_Out', 'Restaurant_Spending',
            'Merchandise_Spending', 'Total_Spending', 'Day_of_Week', 'Is_Weekend', 'Is_Popular_Attraction'
        ]
    df = df[features + [target]]

    # define features and target
    X = df[features]
    y = df[target]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # initialize XGBoost model
    model = XGBRegressor(
        random_state=42,
        n_estimators=500,
        max_depth=4,
        learning_rate=0.1,
        verbosity=0
    )
    
    # cross-validation
    cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Cross-validation negative MSE scores: {cross_val_scores}")
    print(f"Mean cross-validation score: {np.mean(cross_val_scores)}")

    # fit model on training data
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # calculate evaluation metrics
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred)
    }

    print("Model trained successfully.")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")

    # plot Check_In vs Predicted Queue
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test['Check_In'], y_pred, color='blue', alpha=0.6)
    plt.title('Check-In Time vs Predicted Queue Size')
    plt.xlabel('Check-In Time')
    plt.ylabel('Predicted Queue Size')
    plt.show()

    # plot Check_Out vs Predicted Queue
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test['Check_Out'], y_pred, color='green', alpha=0.6)
    plt.title('Check-Out Time vs Predicted Queue Size')
    plt.xlabel('Check-Out Time')
    plt.ylabel('Predicted Queue Size')
    plt.show()

    # create a DataFrame with predicted values and features from X_test
    df_test = X_test.copy()
    df_test['Predicted_' + target] = y_pred

     # calculate correlation between predicted values and features
    correlation = df_test.corr()['Predicted_' + target].sort_values(ascending=False)

    print("\nCorrelation with Predicted " + target + ":")
    print(correlation)

    return model, metrics

# Training model with just IoT data
model_3, metrics_3 = train_demand_model_2(df_iot)

# Save the model for survey and weather data only
with open('../../../models/demand_model_survey_weather.pkl', 'wb') as model_file:
    pickle.dump(model_1, model_file)

print("Model saved successfully!")

with open('../../../models/demand_model_survey_weather_iot.pkl', 'wb') as model_file:
    pickle.dump(model_2, model_file)

print("Model saved successfully!")

with open('../../../models/demand_model_iot.pkl', 'wb') as model_file:
    pickle.dump(model_3, model_file)

print("Model saved successfully!")

