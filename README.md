# Improving Theme Park Guest Experience with Data Analytics

## Project Overview
This project aims to develop a data-driven system that maps and analyzes guest journeys within the theme park industry, from pre-visit planning to post-visit feedback. By leveraging advanced analytics, machine learning, and predictive modeling, the system will gain deeper insights into guest preferences and pain points, helping to improve overall guest experiences consistently. This project centers on Universal Studios Singapore, utilizing business metrics to identify opportunities for improvement within their amusement park.

## Prerequisites
- Python 3.10 or higher
- Git
- [Docker](https://www.docker.com/get-started)
- `bert_model.pt` downloaded from [OneDrive](https://nusu-my.sharepoint.com/personal/e0929810_u_nus_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fe0929810%5Fu%5Fnus%5Fedu%2FDocuments%2Fbert%5Fmodel%2Ept&parent=%2Fpersonal%2Fe0929810%5Fu%5Fnus%5Fedu%2FDocuments&ga=1) 

Once `bert_model.pt` is downloaded, follow the first step below and clone the repository. After that, move `bert_model.pt` into the models/ directory. The following visual has been provided:
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
- Ensure working directory is at data-science-guest-experience, set up virtual environment by running python3 -m venv venv. If virtual environment already exists, run .\venv\Scripts\activate
- Install dependencies: pip install -r requirements.txt
- Set working directory to src folder, and run uvicorn api:app --reload 
- Swagger UI: http://127.0.0.1:8000/docs

### API Endpoints
1. /predict_demand/iot - Make Demand Prediction with IoT data
- Method: POST
- Description: Predicts demand using IoT data.
- Request Body:
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
- Response:
{
  "model": "demand_model_iot.pkl",
  "predicted queue time": [predicted_value]
}

2. /predict_demand/survey_weather - Predict Demand using Survey and Weather data
- Method: POST
- Description: Predicts demand using survey and weather data.
- Request body:
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

- Response:
{
  "model": "demand_model_survey_weather.pkl",
  "predicted queue time": [predicted_value]
}

3. /complaint_severity - Analyze Single Complaint Severity
Method: POST
Description: Analyzes a single customer complaint and determines its severity.
Request Body:
- A JSON object with a "complaint_text" field containing the complaint text.
Example:
```json
{
  "complaint_text": "I waited two hours, but the ride suddenly closed"
}
```
Response:
- A JSON response with the complaint text, severity prediction (0 or 1), severity level ("general" or "severe"), and severity probability.

4. /batch_complaints - Analyze Multiple Complaints
Method: POST
Description: Analyzes multiple customer complaints in a single request and provides severity analysis for each.
Request Body:
- A JSON array of objects, each with a "complaint_text" field.
Example:
```json
[
  {"complaint_text": "I waited two hours, but the ride suddenly closed"},
  {"complaint_text": "The staff was very friendly and helped me solve the problem"},
  {"complaint_text": "The food was too expensive and not tasty"}
]
```
Response:
- A JSON response with an array of complaint analyses, the ratio of severe complaints, and the total number of complaints.

5. /segment (method: POST)  
Segments guests into clusters and provides trait-based segment labels.

- Example:
```json
{
  "age_group": "25 - 34 years old",
  "group_type": "Family (including children)",
  "visit_purpose": "Family outing",
  "express_pass": "Yes",
  "experience_rating": 4.5,
  "awareness": "Social media",
  "response_to_ads": "Yes, but they did not influence my decision",
  "preferred_promotion": "Family/group discounts"
}
```
Response:
```json
{
  "your_cluster": 1,
  "your_segment": "Premium Spenders",
  "segment_summary": [...]
}
```

### Error Handling
400 - Bad Request: The provided features do not match the expected format or are of incorrect types.
404 - Not Found: The specified model was not found in the system.
500 - Internal Server Error: An error occurred while processing the prediction.

## Running the Dashboard
Assuming the GitHub repository has been cloned, here are the steps to run the dashboard.

### 1. Check Your Directory
Make sure you are at ../data-science-guest-experience

### 2. Launching the Dashboard
Enter the following command into your terminal to launch the dashboard:
```bash
streamlit run src/streamlit_dashboard.py
```
