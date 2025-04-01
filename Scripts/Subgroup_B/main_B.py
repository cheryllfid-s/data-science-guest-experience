import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import os

new_directory = r"C:\Users\parma\data-science-guest-experience\data-science-guest-experience\Scripts\Subgroup_B" 
os.chdir(new_directory)

def mainB():

    # load model and data for Question 1
    # load model and data for demand prediction model that takes in survey and weather data
    with open("models/demand_model_survey_weather.pkl", "rb") as model_file:
        model_1 = pickle.load(model_file)

    # load the preprocessed survey & weather dataset
    df_combined_processed = pd.read_pickle("../../data/processed/survey_and_weather_processed_data.pkl")

    # define feature columns
    features = [
        'Favorite_Attraction', 'Satisfaction_Score', 'Age_Group', 'Employment_Status',
        'Visit_Quarter', 'Event', 'Attraction_Reason', 'Season', 'rainfall', 'air_temperature',
        'relative_humidity', 'wind_speed'
    ]
    target = 'Avg_Wait_Time'

    # extract the feature data for prediction
    X_new = df_combined_processed[features]
    y_actual = df_combined_processed[target]

    # predict
    y_pred = model_1.predict(X_new)
    df_combined_processed["Predicted_Avg_Wait_Time"] = y_pred

    # show the predictions
    print("Question 1: Demand prediction in terms of average queue time using survey and weather data")
    print(df_combined_processed[["Predicted_Avg_Wait_Time"]].head())

    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)

    # print the evaluation of the model
    print("\nEvaluation Metrics:")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

    # load the comparison results saved earlier
    with open("b_qn2_comparison.pkl", "rb") as f:
        comp = pickle.load(f)

    # print current layout
    print("\nQuestion 3: Optimization of Attraction Layout and Schedules")
    print("\nCurrent USS Layout:")
    for attraction, time in comp["avg_wait_times_1_multi"].items():
        print(f"{attraction}: {time:.2f} min")
    print(f"Average Wait Time per Guest: {comp['avg_wait_per_guest_1']:.2f} min")

    # print modified layout to show time difference
    print("\nModified USS Layout (We want to close right entrance, simulation swapped Transformers and CYLON)")
    for attraction, time in comp["avg_wait_times_2_multi"].items():
        print(f"{attraction}: {time:.2f} min")
    print(f"Average Wait Time per Guest: {comp['avg_wait_per_guest_2']:.2f} min")

    # Demand prediction model that predicts average wait time based on IoT data
    # Define paths
    # Load dataset
    df_iot = pd.read_pickle("../../data/processed/iot_data.pkl")

    # Define feature columns
    features_model_3 = [
        'Visitor_ID', 'Loyalty_Member', 'Age', 'Gender',
        'Theme_Zone_Visited', 'Attraction', 'Check_In', 'Queue_Time', 'Check_Out', 'Restaurant_Spending',
        'Merchandise_Spending', 'Total_Spending', 'Day_of_Week', 'Is_Weekend', 'Is_Popular_Attraction'
    ]

    target = 'Average_Queue_Time'

    # Load model
    with open("models/demand_model_iot.pkl", "rb") as model_file:
        model_3 = pickle.load(model_file)

    # Prepare the input data
    X_model_3 = df_iot[features_model_3]
    y_true_3 = df_iot[target]

    # Make and store predictions
    y_pred_3 = model_3.predict(X_model_3)
    df_iot["Predicted_Avg_Wait_Time"] = y_pred_3
    df_iot["Attraction"] = label_encoders['Attraction'].inverse_transform(df_iot["Attraction"]) # inverse transform Attraction column back to original names

    # Evaluate model
    rmse_3 = np.sqrt(mean_squared_error(y_true_3, y_pred_3))
    mae_3 = mean_absolute_error(y_true_3, y_pred_3)

    print("\n Question 3: Demand prediction using only IoT data")
    print(f" Evaluation Metrics - RMSE: {rmse_3:.4f}, MAE: {mae_3:.4f}")
    print("\n🔹 Sample Predictions:")
    print(df_iot[['Date', 'Attraction', 'Average_Queue_Time', 'Predicted_Avg_Wait_Time']].head(10))


if __name__ == "__main__":
    mainB()