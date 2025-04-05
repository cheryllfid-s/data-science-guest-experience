from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
import pickle
import numpy as np
import os
from subgroup_a.modeling.q2_guest_segmentation_model import guest_segmentation_model

app = FastAPI()

# Path to models directory
MODELS_DIR = "../models"

# Define expected features and types for each model
model_features = {
    "bert_tokenizer.pkl": {
        "features": ["input_text"],  # Example: Text input for BERT tokenizer
        "feature_types": ["str"]
    },
    "demand_model_iot.pkl": {
        "features": ['Visitor_ID', 'Loyalty_Member', 'Age', 'Gender', 'Theme_Zone_Visited',
                     'Attraction', 'Check_In', 'Queue_Time', 'Check_Out', 'Restaurant_Spending',
                     'Merchandise_Spending', 'Total_Spending', 'Day_of_Week', 'Is_Weekend', 'Is_Popular_Attraction'],
        "feature_types": ['int', 'int', 'int', 'int', 'int', 'int', 'float', 'float', 'float', 'float', 
                          'float', 'float', 'int', 'bool', 'bool']
    },
    "demand_model_survey_weather.pkl": {
        "features": ['Favorite_Attraction', 'Satisfaction_Score', 'Age_Group', 'Employment_Status', 'Visit_Quarter', 
                     'Event', 'Attraction_Reason', 'Season', 'rainfall', 'air_temperature', 
                     'relative_humidity', 'wind_speed'],
        "feature_types": ['int', 'float', 'int', 'int', 'int', 'int', 'int', 'int', 'float', 'float', 
                          'float', 'float']
    },
    "q2_optimization_layout.pkl": {
        "features": [],  # No explicit features available
        "feature_types": []
    },
    "q3_resource_allocation.pkl": {
        "features": [],  # Assuming no features available or needs to be handled separately
        "feature_types": []
    }
}

# Load all pickle models
models = {}
for filename in [
    "bert_tokenizer.pkl",
    "demand_model_iot.pkl",
    "demand_model_survey_weather.pkl",
    "q2_optimization_layout.pkl",
    "q3_resource_allocation.pkl"
]:
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        with open(filepath, "rb") as f:
            models[filename] = pickle.load(f)
        print(f"✅ Successfully loaded {filename}")
    except Exception as e:
        print(f"❌ Error loading {filename}: {e}")

print(f"Loaded models: {models.keys()}")

# Guest segmentation model (initialized with placeholder data)
segmentation_model = guest_segmentation_model(
    df_combined=None, df_labeled=None, scaled=None, pca=None
)

# Pydantic models to accept the features for each model

class IOTFeaturesRequest(BaseModel):
    Visitor_ID: int
    Loyalty_Member: int
    Age: int
    Gender: int
    Theme_Zone_Visited: int
    Attraction: int
    Check_In: float
    Queue_Time: float
    Check_Out: float
    Restaurant_Spending: float
    Merchandise_Spending: float
    Total_Spending: float
    Day_of_Week: int
    Is_Weekend: bool
    Is_Popular_Attraction: bool

class SurveyWeatherFeaturesRequest(BaseModel):
    Favorite_Attraction: int
    Satisfaction_Score: float
    Age_Group: int
    Employment_Status: int
    Visit_Quarter: int
    Event: int
    Attraction_Reason: int
    Season: int
    rainfall: float
    air_temperature: float
    relative_humidity: float
    wind_speed: float

@app.get("/")
def home():
    return {"message": "API is running!"}

@app.get("/models")
def get_models():
    """Returns a list of available models with their feature expectations."""
    model_info = {}
    
    # Loop through model names and details in the model_features dictionary
    for model_name, details in model_features.items():
        model_info[model_name] = {
            "features": details["features"],
            "feature_types": details["feature_types"]
        }
    
    # Debugging: print the model_info dictionary to check its contents
    print(model_info)

    return {"models": model_info}

@app.post("/predict/{model_name}")
def predict(model_name: str, features_request: dict):
    """Predict using the specified pickle model."""
    
    # Ensure model_name exists in loaded models
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    
    # Validate feature input based on model type
    model_info = model_features.get(model_name)
    
    if not model_info:
        raise HTTPException(status_code=404, detail=f"No feature information available for '{model_name}'")
    
    expected_features = model_info["features"]
    expected_types = model_info["feature_types"]
    
    # Check if the features in the request match the expected features
    if model_name == "demand_model_iot.pkl":
        features_request = IOTFeaturesRequest(**features_request)
    elif model_name == "demand_model_survey_weather.pkl":
        features_request = SurveyWeatherFeaturesRequest(**features_request)
    
    # Convert features to the appropriate types
    converted_features = []
    for feature, expected_type in zip(features_request.dict().values(), expected_types):
        if expected_type == "int":
            converted_features.append(int(feature))
        elif expected_type == "float":
            converted_features.append(float(feature))
        elif expected_type == "bool":
            converted_features.append(bool(feature))

    # Reshape and validate feature dimensions
    try:
        input_data = np.array(converted_features).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reshaping features: {e}")

    # Predict using the model
    model = models[model_name]
    try:
        prediction = model.predict(input_data)
        print(f"Prediction result for model '{model_name}': {prediction}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
    
    return {"model": model_name, "prediction": prediction.tolist()}

@app.post("/segment")
def segment_guest(data: dict):
    """Perform guest segmentation."""
    
    # Ensure guest segmentation model is initialized
    if segmentation_model.df_combined is None:
        raise HTTPException(status_code=500, detail="Guest segmentation model is not initialized with data")
    
    try:
        result = segmentation_model.run_pipeline()
        print("Guest segmentation pipeline result:", result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during guest segmentation: {e}")
    
    return {"segment_summary": result[0].to_dict(), "updated_data": result[1].to_dict()}

@app.get("/q2_layout_results", response_class=PlainTextResponse)
def get_q2_layout_results():
    model_name = "q2_optimization_layout.pkl"
    
    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not loaded")

    results = models[model_name]

    try:
        def format_section(title, avg_waits, total_wait):
            lines = [f"{title}"]
            for ride, wait in avg_waits.items():
                lines.append(f"{ride}: {wait:.2f} min")
            lines.append(f"Average Wait Time per Guest: {total_wait:.2f} min")
            return "\n".join(lines)

        section1 = format_section(
            "Current USS Layout (Two Entrances):",
            results["avg_wait_times_1_multi"],
            results["avg_wait_per_guest_1"]
        )

        section2 = format_section(
            "Modified USS Layout (Swapped Transformers and CYLON, we want to use only the left entrance):",
            results["avg_wait_times_2_multi"],
            results["avg_wait_per_guest_2"]
        )

        full_output = f"{section1}\n\n{section2}"
        return full_output

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to format layout results: {e}")
