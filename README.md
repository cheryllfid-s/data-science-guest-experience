# Improving Theme Park Guest Experience with Data Analytics

## Project Overview
This project aims to develop a data-driven system that maps and analyzes guest journeys within the theme park industry, from pre-visit planning to post-visit feedback. By leveraging advanced analytics, machine learning, and predictive modeling, the system will gain deeper insights into guest preferences and pain points, helping to improve overall guest experiences consistently. This project centers on Universal Studios Singapore, utilizing business metrics to identify opportunities for improvement within their amusement park.

## Prerequisites
- Python 3.10 or higher
- Git
- [Docker](https://www.docker.com/get-started)
- `bert.pt` downloaded from [OneDrive](https://nusu-my.sharepoint.com/personal/e0929810_u_nus_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fe0929810%5Fu%5Fnus%5Fedu%2FDocuments%2Fbert%5Fmodel%2Ept&parent=%2Fpersonal%2Fe0929810%5Fu%5Fnus%5Fedu%2FDocuments&ga=1) 

Once `bert.pt` is downloaded, follow the first step below and clone the repository. After that, move `bert.pt` into the models/ directory. The following visual has been provided:
```bash
models/
└── bert_model.pt
```

## Running the Docker
### 1. Clone the Repository
Clone the GitHub repository and navigate into the project folder
```bash
git clone https://github.com/darderrdur17/data-science-guest-experience.git
cd data-science-guest-experience
```
### 2. Build the Docker Container
This command builds and starts the docker container
```bash
docker-compose build
```

### 3. Run the Docker and Choose an Entry Point
You can choose which analysis module to run using `docker-compose`.

#### Available options:
- `main.py`: Runs all components
- `main_A.py`: Runs Guest Journey Analysis and Segmentation
- `main_B.py`: Runs Experience Optimization and Predictive Modeling

#### To run `main.py`:
```bash
docker-compose up uss_main
```

#### To run `main_A.py`:
```bash
docker-compose up uss_a
```

#### To run `main_B.py`:
```bash
docker-compose up uss_b
```

### 4. Stop the Docker Container
Once you're done, you can stop and remove the container with the following command
```bash
docker-compose down
```

## Running the API
Set up virtual environment by typing python3 -m venv venv into cmd.
Install dependencies: pip install -r requirements.txt
Set working directory to src folder, and run uvicorn api:app --reload 
Swagger UI: http://127.0.0.1:8000/docs

API Endpoints
1. /models - Get Available Models
Method: GET
Description: Returns a list of available models and their expected features.

2. /predict/{model_name} - Make Prediction with a Model
Method: POST
Path Parameter:
- model_name: The name of the model to use for prediction (e.g., demand_model_iot.pkl or demand_model_survey_weather.pkl).
Request Body:
- The input should be a JSON object with feature names and values based on the selected model.
Response:
- A JSON response with the prediction result.

Error Handling
400 - Bad Request: The provided features do not match the expected format or are of incorrect types.
404 - Not Found: The specified model was not found in the system.
500 - Internal Server Error: An error occurred while processing the prediction.

Sample inputs for various models:
1. demand_model_iot.pkl: 
{
  "Visitor_ID": 12345,
  "Loyalty_Member": 1,
  "Age": 25,
  "Gender": 1,
  "Theme_Zone_Visited": 2,
  "Attraction": 4,
  "Check_In": 10.5,
  "Queue_Time": 5.0,
  "Check_Out": 15.0,
  "Restaurant_Spending": 20.5,
  "Merchandise_Spending": 30.0,
  "Total_Spending": 50.5,
  "Day_of_Week": 2,
  "Is_Weekend": false,
  "Is_Popular_Attraction": true
}
2. demand_model_survey_weather.pkl:
{
  "Favorite_Attraction": 3,
  "Satisfaction_Score": 4.5,
  "Age_Group": 2,
  "Employment_Status": 1,
  "Visit_Quarter": 1,
  "Event": 0,
  "Attraction_Reason": 1,
  "Season": 3,
  "rainfall": 0.0,
  "air_temperature": 25.0,
  "relative_humidity": 60.0,
  "wind_speed": 5.0
}

## Running the Dashboard
Assuming the GitHub repository has been cloned, here are the steps to run the dashboard.

### 1. Check Your Directory
Make sure you are at .../data-science-guest-experience/src

### 2. Run the Dashboard on your Terminal
This command builds and runs the dashboard.
```bash
streamlit run streamlit_dashboard.py
```
