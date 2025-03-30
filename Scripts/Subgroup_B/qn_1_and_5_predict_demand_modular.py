# import the necessary libraries
import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import simpy
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

# set proper working directory if not yet, and change new_directory to wherever your repo is locally
new_directory = r"C:\Users\parma\data-science-guest-experience\data-science-guest-experience\Scripts\Subgroup_B" 
os.chdir(new_directory)

# Loading datasets
## Load survey data
"""
Purpose:
# Loads and processes a survey dataset to prepare it for further analysis or modeling. 
# The function handles renaming columns, standardizing attraction names, mapping wait time categories, normalizing satisfaction scores, and creating synthetic data for long wait attractions. 
# It also infers the visitor's season of visit and assigns synthetic events.

# Arguments:
# file_path (str): Path to the CSV file containing the survey data (default: "../../data/survey.csv").

# Returns:
# pd.DataFrame: A cleaned and processed DataFrame ready for further analysis or modeling. The resulting DataFrame includes columns like Favorite_Attraction, Avg_Wait_Time, Satisfaction_Score, Age_Group, Employment_Status, Visit_Quarter, and others, with appropriate transformations applied.
"""
def load_survey_data(file_path="../../data/survey.csv"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found. Please provide the survey dataset.")
    
    df = pd.read_csv(file_path)
    
    # rename columns to replace the long questions
    rename_map = {
        "Which age group do you belong to?": "Age_Group",
        "What is your employment status?": "Employment_Status",
        "Which part of the year did you visit USS?": "Visit_Season",
        "Which ride or attraction was your favourite?": "Favorite_Attraction",
        "Why this attraction in particular? ": "Attraction_Reason",
        "Did you experience any rides with longer-than-expected wait times? If yes, which ride(s)?": "Long_Wait_Attractions",
        "How long did you wait in line for rides on average during your visit?": "Avg_Wait_Time",
        "On a scale of 1-5, how would you rate your overall experience at USS?": "Satisfaction_Score"
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # map vague answers to default answer, which is the most popular attraction
    invalid_answers = ["Can't remember.", 'Cannot remember ', 'Did not take any rides . Only my son', 'Enchanted Airways']
    df['Favorite_Attraction'] = df['Favorite_Attraction'].apply(lambda x: "Battlestar Galactica: HUMAN" if x in invalid_answers else x)
    
    # mapping attraction names so it's standardised across all datasets (especially with IoT data)
    attraction_map = {
        "CYLON": "Battlestar Galactica: CYLON",
        "HUMAN": "Battlestar Galactica: HUMAN",
        "Transformers": "Transformers: The Ride",
        "Revenge of the Mummy": "Revenge of the Mummy",
        "Sesame Street Spaghetti Space Chase": "Sesame Street Spaghetti Space Chase",
        "Puss in Boots": "Puss In Boots' Giant Journey",
        "Canopy Flyer": "Canopy Flyer",
        "Treasure Hunters": "Treasure Hunters"
    }
    if 'Favorite_Attraction' in df.columns:
        df['Favorite_Attraction'] = df['Favorite_Attraction'].apply(lambda x: attraction_map.get(x, x))
    
    # taking the median for the wait times and mapping it according to each range
    wait_time_map = {
        "Less than 15 minutes": 10,
        "15 to 30 minutes": 22.5,
        "31 to 45 minutes": 37.5,
        "46 to 60 minutes": 52.5,
        "61 to 90 minutes": 75,
        "More than 90 minutes": 100
    }
    if "Avg_Wait_Time" in df.columns:
        df["Avg_Wait_Time"] = df["Avg_Wait_Time"].map(wait_time_map).fillna(37.5)
    else:
        df["Avg_Wait_Time"] = 37.5
    
    # normalise satisfaction score
    df["Satisfaction_Score"] = pd.to_numeric(df.get("Satisfaction_Score", 3), errors="coerce")
    df["Satisfaction_Score"] = (
        (df["Satisfaction_Score"] - df["Satisfaction_Score"].min()) /
        (df["Satisfaction_Score"].max() - df["Satisfaction_Score"].min())
    )
    
    # mapping seasons
    df["Visit_Quarter"] = df.get("Visit_Season", "Jul-Sep").apply(lambda x: "Jul-Sep" if "jul" in str(x).lower() else "Oct-Dec")
    
    # synthetically generate if there are events or not
    np.random.seed(42)
    df["Event"] = df["Visit_Season"].apply(lambda x: 
        "Special Event" if x == "Oct-Dec" else np.random.choice(["None", "Special Event"], p=[0.8, 0.2])
    ) #typically people visit end of year for HHN, otherwise might have other smaller events organised throughout the year which is randomly assigned. 
    
    # skipping long wait explosion for brevity
    base_columns = ["Favorite_Attraction", "Avg_Wait_Time", "Satisfaction_Score", "Age_Group", 
                    "Employment_Status", "Visit_Quarter", "Event"]
    if "Attraction_Reason" in df.columns:
        base_columns.append("Attraction_Reason")
    
    return df[base_columns]

# Load survey and display
df_survey = load_survey_data()
print(df_survey.head())

# Survey dataset is small, so generate more data to feed into the model

metadata = Metadata.detect_from_dataframe(data=df_survey) # retrieve metadata
synthesizer = GaussianCopulaSynthesizer(metadata) # retrieve synthesiser
synthesizer.fit(df_survey) # fitting the survey data
synthetic_data = synthesizer.sample(num_rows=1000) # generate synthetic data
combined_survey_df = pd.concat([df_survey, synthetic_data], ignore_index=True)
print(combined_survey_df.head()) # new survey data with synthetic data

# Load IoT data (to analyse impact of IoT data on predictive accuracy and thus operational flexibility)
""" 
Purpose:
Loads and preprocesses synthetic IoT data to prepare it for further analysis or modeling. 
The function handles the following preprocessing steps:
- Converts date columns to datetime format.
- Flags weekend visits.
- Flags visits to popular attractions.
- Expands the "Attraction_Times" data into individual records for each attraction visited.

Arguments:
file_path (str): Path to the synthetic IoT CSV file. Default is "../../data/synthetic_iot_data_v3.csv".

Returns:
pd.DataFrame: A cleaned and enriched DataFrame that includes detailed information about the visitor's attractions, like:
- Date: Date of the visit.
- Loyalty_Member: Whether the visitor is a loyalty member ("Yes" or "No").
- Age: Age of the visitor.
- Theme_Zone_Visited: Zones of the theme park visited by the visitor.
- Attraction: Name of the visited attraction.
- Check_In: Time when the visitor checked into the attraction.
- Queue_Time: Time spent in the queue for the attraction.
- Check_Out: Time when the visitor checked out of the attraction.
- Average_Queue_Time: The average queue time for the visitor across all visited attractions.
- Restaurant_Spending: The amount spent on food and beverages.
- Merchandise_Spending: The amount spent on merchandise.
- Total_Spending: The total spending (restaurant + merchandise).
- Day_of_Week: Day of the week when the visit occurred.
- Is_Weekend: Flag indicating whether the visit occurred on a weekend.
- Is_Popular_Attraction: Flag indicating whether the visited attraction is considered "popular." 
"""

def load_iot_data(file_path="../../data/synthetic_iot_data_v3.csv"):
    if not os.path.exists(file_path):
        print(f"Warning: IoT data file {file_path} not found. Skipping IoT data integration.")
        return None
    
    df_iot = pd.read_csv(file_path) # load data
    
    df_iot['Date'] = pd.to_datetime(df_iot['Date']) # convert 'Date' to datetime
    
    df_iot['Day_of_Week'] = df_iot['Date'].dt.day_name() # retrieve day of the week

    df_iot['Is_Weekend'] = df_iot['Day_of_Week'].isin(["Saturday", "Sunday"]) # flag for weekend visits

    # define popular attractions
    POPULAR_ATTRACTIONS = {
        "Revenge of the Mummy",
        "Battlestar Galactica: CYLON",
        "Transformers: The Ride",
        "Battlestar Galactica: HUMAN",
        "Sesame Street Spaghetti Space Chase"
    }

    # flag if any visited attraction is popular
    df_iot['Is_Popular_Attraction'] = df_iot['Attraction_Times'].apply(
        lambda x: any(attraction['Attraction'] in x for attraction in eval(x)) 
    )

    # function to process check in and out times from attractions
    def process_attraction_times(attraction_times):
        attractions_data = []

        attractions = eval(attraction_times)  # converts stringified list into actual list of dictionaries
        for attraction in attractions:
            attractions_data.append({
                "Attraction": attraction['Attraction'],
                "Check_In": attraction['Check_In'],
                "Queue_Time": attraction['Queue_Time'],
                "Check_Out": attraction['Check_Out']
            })
        return attractions_data

    # apply the processing function to each row in the 'Attraction_Times' column
    df_iot['Processed_Attraction_Times'] = df_iot['Attraction_Times'].apply(process_attraction_times)

    # expand the Processed_Attraction_Times list into separate rows, to create separate rows for each attraction visited
    df_iot_expanded = df_iot.explode('Processed_Attraction_Times', ignore_index=True)

    # extract individual columns from the expanded 'Processed_Attraction_Times' list
    df_iot_expanded['Attraction'] = df_iot_expanded['Processed_Attraction_Times'].apply(lambda x: x['Attraction'])
    df_iot_expanded['Check_In'] = df_iot_expanded['Processed_Attraction_Times'].apply(lambda x: x['Check_In'])
    df_iot_expanded['Queue_Time'] = df_iot_expanded['Processed_Attraction_Times'].apply(lambda x: x['Queue_Time'])
    df_iot_expanded['Check_Out'] = df_iot_expanded['Processed_Attraction_Times'].apply(lambda x: x['Check_Out'])

    # drop the 'Processed_Attraction_Times' column
    df_iot_expanded.drop(columns=['Processed_Attraction_Times'], inplace=True)

    # select relevant columns based on your IoT dataset
    relevant_columns = [
        "Visitor_ID", "Date", "Loyalty_Member", "Age", "Gender", "Theme_Zone_Visited",
        "Attraction", "Check_In", "Queue_Time", "Check_Out", "Average_Queue_Time",
        "Restaurant_Spending", "Merchandise_Spending", "Total_Spending", "Day_of_Week", "Is_Weekend", "Is_Popular_Attraction"
    ]
    
    # filter the dataframe to include only relevant columns
    df_iot_expanded = df_iot_expanded[relevant_columns]

    return df_iot_expanded

# Load IoT data 
df_iot = load_iot_data()
print(df_iot.head())

## Load weather data
"""
Fetches or loads monthly weather data from Singapore’s government open data API and aggregates it into seasonal averages 
to be used as input features in demand prediction models.

Purpose:
- Automates the process of downloading or loading historical monthly weather data for 2024.
- Maps each month to a seasonal category used in survey responses (e.g., "January - March").
- Outputs a clean dataset with average weather values per season.
"""

def fetch_weather_data(file_path="../../data/singapore_seasonal_weather.csv"):
    if os.path.exists(file_path):
        print(f"Loaded existing weather data from: {file_path}")
        return pd.read_csv(file_path)

    print("Fetching weather data from API...")

    base_url = "https://api.data.gov.sg/v1/environment/"
    weather_types = ["rainfall", "air-temperature", "relative-humidity", "wind-speed"]
    months = [f"2024-{str(m).zfill(2)}-15" for m in range(1, 13)]
    month_names = [datetime.strptime(m, "%Y-%m-%d").strftime("%B") for m in months]

    all_data = []

    for date_str, month_name in zip(months, month_names):
        print(f"Fetching data for: {date_str}")
        daily_data = {"month": month_name}

        for weather_type in weather_types:
            url = f"{base_url}{weather_type}"
            params = {"date": date_str}
            response = requests.get(url, params=params)

            if response.status_code == 200:
                try:
                    data = response.json()
                    readings = data["items"][0]["readings"]
                    avg_value = sum(d["value"] for d in readings) / len(readings)
                    daily_data[weather_type] = avg_value
                except (KeyError, IndexError):
                    print(f"Missing data for {weather_type} on {date_str}")
                    daily_data[weather_type] = None
            else:
                print(f"Error fetching {weather_type} for {date_str}: {response.status_code}")
                daily_data[weather_type] = None

        all_data.append(daily_data)

    df = pd.DataFrame(all_data)

    # map months to seasons
    month_to_season = {
        "January": "January - March", "February": "January - March", "March": "January - March",
        "April": "April - June", "May": "April - June", "June": "April - June",
        "July": "July - September", "August": "July - September", "September": "July - September",
        "October": "October - December", "November": "October - December", "December": "October - December"
    }
    df["Season"] = df["month"].map(month_to_season)

    # average weather data by season
    df_seasonal = df.groupby("Season").agg({
        "rainfall": "mean",
        "air-temperature": "mean",
        "relative-humidity": "mean",
        "wind-speed": "mean"
    }).reset_index()

    df_seasonal.rename(columns={
        "air-temperature": "air_temperature",
        "relative-humidity": "relative_humidity",
        "wind-speed": "wind_speed"
    }, inplace=True)

    # save to disk
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df_seasonal.to_csv(file_path, index=False)
    print(f"Saved seasonal weather data to: {file_path}")

    return df_seasonal

# Retrieve weather data
df_weather = fetch_weather_data("../../data/singapore_seasonal_weather.csv")
print(df_weather.head())

# Merging datasets

"""
Purpose:
Merges survey data with seasonal weather data, and appends IoT data if provided. 
The function combines three datasets: survey data, weather data, and optionally IoT data. 
It ensures that seasonal information from the survey and IoT data is aligned with the weather data for further analysis.

Arguments:
- survey_df (pd.DataFrame): The survey dataset, which must contain a 'Visit_Quarter' column indicating the quarter of the year when the visit occurred.
- weather_df (pd.DataFrame): Seasonal weather data that contains a 'Season' column. The weather data should be structured with seasonal weather attributes, such as temperature or humidity.
- iot_df (pd.DataFrame, optional): The IoT dataset (if available). It should contain a 'Season' column for season-based merging. If not provided, only the survey and weather data will be merged.

Returns:
A combined dataset that includes survey data, weather data, and optionally IoT data, merged based on the 'Season' column. The merged dataset preserves all relevant columns from the input datasets.
"""

def merge_survey_weather_iot(survey_df, weather_df, iot_df=None):
    # map 'Visit_Quarter' to 'Season' based on the provided data
    quarter_to_season = {
        "Jan-Mar": "January - March",
        "Apr-Jun": "April - June",
        "Jul-Sep": "July - September",
        "Oct-Dec": "October - December",
    }
    survey_df['Season'] = survey_df['Visit_Quarter'].map(quarter_to_season)

    # merge survey with weather data based on 'Season'
    merged_survey = pd.merge(survey_df, weather_df, on='Season', how='left')

    if iot_df is None:
        return merged_survey

    # if season column doesn't exist in IoT data, add season column based on date of visit
    if 'Season' not in iot_df.columns:
        print("'Season' column missing in IoT data. Assigning season synthetically...")
        if 'Date' in iot_df.columns:
            iot_df['Date'] = pd.to_datetime(iot_df['Date'])
            month_to_season = {
                1: "January - March", 2: "January - March", 3: "January - March",
                4: "April - June", 5: "April - June", 6: "April - June",
                7: "July - September", 8: "July - September", 9: "July - September",
                10: "October - December", 11: "October - December", 12: "October - December"
            }
            iot_df['Season'] = iot_df['Date'].dt.month.map(month_to_season)
        else:
            iot_df['Season'] = np.random.choice(
                ["January - March", "April - June", "July - September", "October - December"],
                size=len(iot_df),
                p=[0.1, 0.1, 0.4, 0.4]
            )

    # rename 'Average_Queue_Time' to 'Avg_Wait_Time' in IoT data
    if 'Average_Queue_Time' in iot_df.columns:
        iot_df.rename(columns={'Average_Queue_Time': 'Avg_Wait_Time'}, inplace=True)

    # merge IoT with weather based on 'Season'
    merged_iot = pd.merge(iot_df, weather_df, on='Season', how='left')

    # append both datasets to preserve all columns
    combined = pd.concat([merged_survey, merged_iot], ignore_index=True, join='outer')

    return combined

## Merged dataset without IOT data
## To put into model later to check the effect of just survey and weather data on the accuracy of the model.
df_combined = merge_survey_weather_iot(combined_survey_df, df_weather)
print(df_combined)

## Merged dataset with IOT data
## To train model with all 3 datasets combined together and see the difference in evaluation metrics.

df_all_combined = merge_survey_weather_iot(combined_survey_df, df_weather, df_iot)
print(df_all_combined)

## Preparing data for modelling
"""
Purpose:
Processes a dataset for modeling, specifically ensuring it is in a format suitable for XGBoost, which requires all inputs 
to be numerical. The function handles categorical feature encoding, date extraction, missing values imputation, and 
specific preprocessing for IoT data if present.

Arguments:
- df (pd.DataFrame): The merged dataset containing either survey data, IoT data, or both. 
It must include features like Favorite_Attraction, Age_Group, Date, and possibly IoT-specific columns like Check_In, Queue_Time, etc.
- iot_data (bool): A flag indicating whether IoT data is present in the dataset. 
If set to True, the function will handle IoT-specific columns, such as Check_In, Queue_Time, etc. 
If set to False, the function assumes the dataset is survey-only data.

Returns:
 A processed DataFrame that is ready for modeling. This DataFrame contains all features encoded as numerical values and handles missing data.
"""
def process_data_for_model(df, iot_data=False):
    encoders = {}  # store label encoders

    # encode categorical columns
    label_cols = ['Favorite_Attraction', 'Age_Group', 'Employment_Status', 
                  'Visit_Quarter', 'Event', 'Attraction_Reason', 'Day_of_Week', 'Season']
    
    for col in label_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le  # store encoder

    # process 'Date' column
    if 'Date' in df.columns:
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day_of_Week'] = df['Date'].dt.weekday
        df.drop('Date', axis=1, inplace=True)

    # handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':  
            df[col] = df[col].fillna("Unknown")
        elif df[col].dtype in ['float64', 'int64']:  
            df[col] = df[col].fillna(df[col].mean())

    # handle IoT columns
    if iot_data:
        iot_columns = ['Check_In', 'Queue_Time', 'Check_Out', 'Restaurant_Spending', 
                       'Merchandise_Spending', 'Total_Spending']
        
        for col in iot_columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[col].mean())

        # encode IoT-specific columns
        for col in ['Attraction', 'Visitor_ID']:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                encoders[col] = le

    return df, encoders  


# Process dataset with IoT data
df_all_combined_processed, encoders_1 = process_data_for_model(df_all_combined, iot_data=True)
print(df_all_combined_processed.head())
print(df_all_combined_processed.columns.tolist()) #to check all the columns are correct

# Process dataset with only survey data
df_combined_processed, encoders_2 = process_data_for_model(df_combined, iot_data=False)
print(df_combined_processed.head())
print(df_combined_processed.columns.tolist()) #to check all the columns are correct

## Modelling with XGBoost
"""
Purpose:
Trains an XGBoost regression model on the given dataset to predict the specified target variable (default: 'Avg_Wait_Time'). The function returns the trained model and evaluates its performance using common regression metrics (RMSE and MAE). Additionally, it calculates and prints the correlation between the predicted target and the input features.

Arguments:
- df (pd.DataFrame): The dataset to train the model on. This dataset should include both features and the target variable. The features may include categorical variables such as Favorite_Attraction, Age_Group, Employment_Status, and more.
- target (str): The column name of the target variable that the model is trying to predict. By default, this is set to 'Avg_Wait_Time', but it can be changed to any other column in the dataset (e.g., Queue_Time).

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

def train_demand_model_2(df, target='Avg_Wait_Time'):
    # feature columns
    features = [
            'Visitor_ID', 'Loyalty_Member', 'Age', 'Gender',
            'Theme_Zone_Visited', 'Attraction', 'Check_In', 'Queue_Time', 'Check_Out', 'Restaurant_Spending',
            'Merchandise_Spending', 'Total_Spending', 'Day_of_Week', 'Is_Weekend', 'Is_Popular_Attraction'
        ]
    df = df[features + [target]]
    
    # encode categorical columns
    label_cols = ['Favorite_Attraction', 'Age_Group', 'Employment_Status', 'Visit_Quarter', 
                  'Event', 'Attraction_Reason', 'Season', 'Day_of_Week', 'Visitor_ID', 'Attraction', "Loyalty_Member", "Gender", "Theme_Zone_Visited"]
    
    for col in label_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

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