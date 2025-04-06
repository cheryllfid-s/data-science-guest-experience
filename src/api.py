from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
import pickle
import numpy as np
import os
from subgroup_a.modeling.q2_guest_segmentation_model import guest_segmentation_model
import torch
from transformers import BertForSequenceClassification
import torch.nn.functional as F

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

# load BERT model
try:
    # load tokenizer
    with open(os.path.join(MODELS_DIR, "bert_tokenizer.pkl"), 'rb') as f:
        bert_tokenizer = pickle.load(f)
    
    # initialize model
    bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    # load model state
    bert_model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "bert_model.pt")))
    
    # set to evaluation mode
    bert_model.eval()
    
    print("Successfully loaded BERT model and tokenizer")
except Exception as e:
    print(f"Error loading BERT model: {e}")
    bert_model = None
    bert_tokenizer = None

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

class ResourceAllocationRequest(BaseModel):
    ATTRACTION: str
    PARK: str
    WAIT_TIME_MAX: int
    NB_UNITS: int
    GUEST_CARRIED: int
    CAPACITY: int
    ADJUST_CAPACITY: int
    OPEN_TIME: int
    UP_TIME: int
    DOWNTIME: int
    NB_MAX_UNIT: int
    estimate_attendance: int
    month: int
    year: int
    sale: float
    adm_sale: float
    rest_sale: float

class GuestSegmentationRequest(BaseModel):
    age_group: str
    group_type: str
    visit_purpose: str
    express_pass: str
    experience_rating: float
    awareness: str
    response_to_ads: str
    preferred_promotion: str

class ComplaintTextRequest(BaseModel):
    complaint_text: str = Field(..., description="customer complaint text")

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
def segment_guest(data: GuestSegmentationRequest):
    """Perform guest segmentation using user input features."""
    if segmentation_model.df_labeled is None:
        raise HTTPException(status_code=500, detail="Guest segmentation model is not initialized with base data")

    try:
        user_df = pd.DataFrame([data.dict()])
        combined_df = pd.concat([segmentation_model.df_labeled.copy(), user_df], ignore_index=True)
        encoded_df = combined_df.copy()
        for col in encoded_df.select_dtypes(include='object').columns:
            encoded_df[col] = LabelEncoder().fit_transform(encoded_df[col])

        
        scaled = StandardScaler().fit_transform(encoded_df)
        pca = PCA(n_components=2).fit_transform(scaled)

        # Rerun the model pipeline with updated data
        updated_model = guest_segmentation_model(
            df_combined=encoded_df,
            df_labeled=combined_df,
            scaled=scaled,
            pca=pca
        )

        summary, df_labeled = updated_model.run_pipeline()

        
        predicted_cluster = int(df_labeled.iloc[-1]["cluster"])
        segment_name = summary.loc[predicted_cluster, "Segment"]

        return {
            "your_cluster": predicted_cluster,
            "your_segment": segment_name,
            "segment_summary": summary.to_dict(orient="records")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during guest segmentation: {e}")

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

@app.post("/predict/resource_allocation")
def predict_resource_allocation(data: ResourceAllocationRequest):
    model_name = "q3_resource_allocation.pkl"

    if model_name not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not loaded")

    model = models[model_name]

    try:
        df = pd.DataFrame([data.dict()])

        # Load encoder
        encoder_path = os.path.join(MODELS_DIR, "encoder_q3.pkl")
        if not os.path.exists(encoder_path):
            raise HTTPException(status_code=500, detail="Encoder file not found")

        with open(encoder_path, "rb") as f:
            encoder = pickle.load(f)

        encoded_cats = encoder.transform(df[["ATTRACTION", "PARK"]])
        encoded_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out())

        df = df.drop(columns=["ATTRACTION", "PARK"]).join(encoded_df)
        df = df[model.feature_names_in_]

        predicted_staff = np.ceil(model.predict(df)).astype(int)[0]

        return {"Predicted_Staff_Count": int(predicted_staff)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict/complaint_severity")
def predict_complaint_severity(data: ComplaintTextRequest):
    """predict customer complaint severity"""
    
    if bert_model is None or bert_tokenizer is None:
        raise HTTPException(status_code=500, detail="BERT model or tokenizer not loaded")
    
    try:
        # tokenize text
        encoding = bert_tokenizer(data.complaint_text, return_tensors="pt", padding=True, truncation=True)
        
        # predict
        with torch.no_grad():
            outputs = bert_model(input_ids=encoding['input_ids'],
                        attention_mask=encoding['attention_mask'])
            probs = F.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            severity_prob = probs[0][1].item()
        
        severity_level = "severe" if prediction == 1 else "general"
        
        return {
            "complaint_text": data.complaint_text,
            "severity_prediction": prediction,
            "severity_level": severity_level,
            "severity_probability": severity_prob
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting complaint severity: {str(e)}")

@app.post("/predict/batch_complaints")
def predict_batch_complaints(data: list[ComplaintTextRequest]):
    """batch predict customer complaint severity"""
    
    if bert_model is None or bert_tokenizer is None:
        raise HTTPException(status_code=500, detail="BERT model or tokenizer not loaded")
    
    results = []
    
    try:
        for complaint_request in data:
            # tokenize text
            encoding = bert_tokenizer(complaint_request.complaint_text, return_tensors="pt", padding=True, truncation=True)
            
            # predict
            with torch.no_grad():
                outputs = bert_model(input_ids=encoding['input_ids'],
                            attention_mask=encoding['attention_mask'])
                probs = F.softmax(outputs.logits, dim=1)
                prediction = torch.argmax(probs, dim=1).item()
                severity_prob = probs[0][1].item()
            
            severity_level = "severe" if prediction == 1 else "general"
            
            results.append({
                "complaint_text": complaint_request.complaint_text,
                "severity_prediction": prediction,
                "severity_level": severity_level,
                "severity_probability": severity_prob
            })
        
        # calculate severe complaint ratio
        severe_count = sum(1 for result in results if result["severity_prediction"] == 1)
        severe_ratio = severe_count / len(results) if results else 0
        
        return {
            "complaints": results,
            "severe_complaint_ratio": severe_ratio,
            "total_complaints": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error batch predicting complaint severity: {str(e)}")

