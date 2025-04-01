import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.preprocessing import OneHotEncoder

# new_directory = r"C:\Users\parma\data-science-guest-experience\data-science-guest-experience\Scripts\Subgroup_B" 
# os.chdir(new_directory)


def preprocess_new_data(new_data):
    """Preprocess new input data by encoding categorical variables."""
    # Remove target variables if they exist
    target_vars = ["staff_count", "reg_worker", "part_worker"]
    new_data = new_data.drop(columns=[col for col in target_vars if col in new_data.columns], errors='ignore')
    
    # Apply one-hot encoding
    encoder = OneHotEncoder(drop="first", sparse=False)
    encoded_cats = encoder.fit_transform(new_data[["ATTRACTION", "PARK"]])
    new_data = new_data.drop(columns=["ATTRACTION", "PARK"]).join(pd.DataFrame(encoded_cats, 
                                                         columns=encoder.get_feature_names_out()))
    return new_data

def predict_staff_count(model, new_data):
    """Predict staff count using the trained model and round up the predictions."""
    if model is None or new_data is None:
        print("Error: Model or preprocessed data is missing. Cannot make predictions.")
        return None
    return np.ceil(model.predict(new_data))

def load_model(model_path):
    """Load the trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None

def mainB():
    # load model and data for Question 1
    # load model and data for demand prediction model that takes in survey and weather data
    with open("../../models/demand_model_survey_weather.pkl", "rb") as model_file:
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

 
    # Call q2 model simulation results
    with open("../../models/q2_optimization_layout.pkl", "rb") as f:
        comp = pickle.load(f)

    # Print: Current Layout
    print("\nQuestion 2: Optimization of Attraction Layout and Schedules")
    print("\nCurrent USS Layout (Two Entrances) - Multi Queue:")
    for attraction, time in comp["avg_wait_times_1_multi"].items():
        print(f"{attraction}: {time:.2f} min")
    print(f"Average Wait Time per Guest: {comp['avg_wait_per_guest_1']:.2f} min")
    

    # Print: Modified Layout
    print("\nModified USS Layout (Left Entrance Only, Swapped Transformers and CYLON) - Multi Queue:")
    for attraction, time in comp["avg_wait_times_2_multi"].items():
        print(f"{attraction}: {time:.2f} min")
    print(f"Average Wait Time per Guest: {comp['avg_wait_per_guest_2']:.2f} min")
    
    print("\nQuestion 3: Resource Allocation")
    model_path = "../../models/q3_resource_allocation.pkl"
    data_path = "../../data/processed/q3_resource_allocation.csv"
    # Load the trained model
    model = load_model(model_path)
    # Load new data
    try:
        new_data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return
    # Preprocess new data
    new_data_processed = preprocess_new_data(new_data)
    if new_data_processed is None:
        return
    # Make predictions
    predicted_staff = predict_staff_count(model, new_data_processed)
    if predicted_staff is not None:
        # Add predicted staff count to the original data
        new_data["Predicted_Staff_Count"] = predicted_staff
        
        # Print the dataset with the predicted staff count
        print("Dataset with Predicted Staff Count:")
        print(new_data.head())  # Print the first few rows of the dataset with predictions
    

##################
    # Demand prediction model that predicts average wait time based on IoT data
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
    with open("../../models/demand_model_iot.pkl", "rb") as model_file:
        model_3 = pickle.load(model_file)

    # Prepare the input data
    X_model_3 = df_iot[features_model_3]
    y_true_3 = df_iot[target]

    # Make and store predictions
    y_pred_3 = model_3.predict(X_model_3)
    df_iot["Predicted_Avg_Wait_Time"] = y_pred_3

    # Evaluate model
    rmse_3 = np.sqrt(mean_squared_error(y_true_3, y_pred_3))
    mae_3 = mean_absolute_error(y_true_3, y_pred_3)

    print("\n Question 3: Demand prediction using only IoT data")
    print(f" Evaluation Metrics - RMSE: {rmse_3:.4f}, MAE: {mae_3:.4f}")
    print("\n🔹 Sample Predictions:")
    print(df_iot[['Date', 'Attraction', 'Average_Queue_Time', 'Predicted_Avg_Wait_Time']].head(10))
    
if __name__ == "__main__":
    mainB()
