import pickle
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import os
from sklearn.preprocessing import OneHotEncoder
# from Modeling.q4_predicting_guest_complaints import load_bert_model
from transformers import BertForSequenceClassification
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
# new_directory = r"C:\Users\parma\data-science-guest-experience\data-science-guest-experience\Scripts\Subgroup_B" 
# os.chdir(new_directory)


def preprocess_new_data(new_data):
    """Preprocess new input data by encoding categorical variables."""
    # Remove target variables if they exist
    target_vars = ["staff_count", "reg_worker", "part_worker"]
    new_data = new_data.drop(columns=[col for col in target_vars if col in new_data.columns], errors='ignore')
    
    # Apply one-hot encoding
    encoder = OneHotEncoder(drop="first", sparse_output=False)
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
    ##### QUESTION 1 #####
    # load model and data for Question 1
    # load model and data for demand prediction model that takes in survey and weather data
    with open("../../models/demand_model_survey_weather.pkl", "rb") as model_file:
        model_1 = pickle.load(model_file)

    # load the preprocessed survey & weather dataset
    df_combined_processed = pd.read_pickle("../../data/processed data/survey_and_weather_processed_data.pkl")

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

    ##### QUESTION 2 #####
    # Call q2 model simulation results
    with open("../../models/q2_optimization_layout.pkl", "rb") as f:
        comp = pickle.load(f)

    # Print: Current Layout
    print("\nQuestion 2: Optimization of Attraction Layout and Schedules")
    print("\nCurrent USS Layout (Two Entrances):")
    for attraction, time in comp["avg_wait_times_1_multi"].items():
        print(f"{attraction}: {time:.2f} min")
    print(f"Average Wait Time per Guest: {comp['avg_wait_per_guest_1']:.2f} min")
    

    # Print: Modified Layout
    print("\nModified USS Layout (Swapped Transformers and CYLON, we want to use only the left entrance):")
    for attraction, time in comp["avg_wait_times_2_multi"].items():
        print(f"{attraction}: {time:.2f} min")
    print(f"Average Wait Time per Guest: {comp['avg_wait_per_guest_2']:.2f} min")

    
    
    ##### QUESTION 3 #####
    print("\nQuestion 3: Resource Allocation")
    model_path = "../../models/q3_resource_allocation.pkl"
    data_path = "../../data/processed data/q3_resource_allocation.csv"
    
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
        
        # Select relevant columns for display
        columns_to_display = ["Date", "Attraction", "Park", "Predicted_Staff_Count"]
        if "Actual_Staff_Count" in new_data.columns:
            columns_to_display.append("Actual_Staff_Count")
            new_data["Difference"] = new_data["Predicted_Staff_Count"] - new_data["Actual_Staff_Count"]
            columns_to_display.append("Difference")
            
            # Compute MAE (Mean Absolute Error)
            mae = np.mean(np.abs(new_data["Difference"]))
            print(f"Mean Absolute Error (MAE): {mae:.2f}")
        
        # Print the dataset with selected columns
        print("Dataset with Predicted vs Actual Staff Count:")
        print(new_data[columns_to_display].head())

    
    ##### QUESTION 4 #####
    def load_bert_model(model_path='../../models/bert_model.pt', tokenizer_path='../../models/bert_tokenizer.pkl'):
        """
        load saved BERT model and tokenizer

        Args:
            model_path: path to model state dictionary
            tokenizer_path: path to tokenizer pickle file

        Returns:
            model: loaded BERT model
            tokenizer: loaded tokenizer
        """
        try:
            # load tokenizer
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)

            # initialize model
            model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

            # load model state
            model.load_state_dict(torch.load(model_path))

            # set to evaluation mode
            model.eval()

            print("successfully loaded model and tokenizer")
            return model, tokenizer

        except Exception as e:
            print(f"error loading model: {str(e)}")
            return None, None
    # Load the BERT model
    model, tokenizer = load_bert_model()
    def predict_complaint_severity(text, model, tokenizer):
        try:
            # Tokenize the text
            encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
            # predict
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids=encoding['input_ids'],
                            attention_mask=encoding['attention_mask'])
                probs = F.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                
            return prediction, probs[0][1].item()
        except Exception as e:
            print(f"Error predicting complaint severity: {str(e)}")
            return None, None

    print("\nQuestion 4: Customer Complaint Severity Prediction")
    
    # Example complaint texts
    sample_complaints = [
        "I waited two hours, but the ride suddenly closed",
        "The staff was very friendly and helped me solve the problem",
        "The food was too expensive and not tasty"
    ]
    
    # Predict the severity of each complaint
    for i, complaint in enumerate(sample_complaints):
        severity, prob = predict_complaint_severity(complaint, model, tokenizer)
        if severity is not None:
            severity_level = "Severe" if severity == 1 else "Moderate"
            print(f"Complaint {i+1}: {complaint}")
            print(f"Predicted Severity: {severity_level}, Severity Probability: {prob:.4f}")
            print()
    
    # If there is actual complaint data, load and predict
    try:
        # Load the complaint data (example)
        complaints_df = pd.read_csv("../../data/processed data/customer_complaints.csv")
        print("Customer Complaint Data Analysis:")
        
        # Add prediction results to the dataframe
        complaints_df['severity_pred'] = None
        complaints_df['severity_prob'] = None
        
        for idx, row in complaints_df.head(10).iterrows():  # Process only the first 10 as an example
            severity, prob = predict_complaint_severity(row['complaint_text'], model, tokenizer)
            complaints_df.at[idx, 'severity_pred'] = severity
            complaints_df.at[idx, 'severity_prob'] = prob
        
        # Display results
        print(complaints_df[['complaint_text', 'severity_pred', 'severity_prob']].head(10))
        
        # Calculate the ratio of severe complaints
        severe_ratio = (complaints_df['severity_pred'] == 1).mean()
        print(f"Severe Complaint Ratio: {severe_ratio:.2%}")
    except FileNotFoundError:
        print("No complaint data file found, only using example data")
    except Exception as e:
        print(f"Error processing complaint data: {str(e)}")
        
    ##### QUESTION 5 #####
    # Demand prediction model that predicts average wait time based on IoT data
    # Load dataset
    df_iot = pd.read_pickle("../../data/processed data/iot_data.pkl")

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

    print("\n Question 5: Demand prediction using only IoT data")
    print(f" Evaluation Metrics - RMSE: {rmse_3:.4f}, MAE: {mae_3:.4f}")
    print("\n Sample Predictions:")
    print(df_iot[['Date', 'Attraction', 'Average_Queue_Time', 'Predicted_Avg_Wait_Time']].head(10))


if __name__ == "__main__":
    mainB()
