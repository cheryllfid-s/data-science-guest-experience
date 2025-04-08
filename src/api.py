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
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from subgroup_a.datapreparation_A import cleaning_q2
from typing import List

app = FastAPI()

MODELS_DIR = "../models"

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

# Guest segmentation model (Q2)
try:
    cleaned_combined, cleaned_labeled, scaled, pca = cleaning_q2()
    segmentation_model = guest_segmentation_model(
        df_combined=cleaned_combined,
        df_labeled=cleaned_labeled,
        scaled=scaled,
        pca=pca
    )
    print("✅ Guest segmentation model initialized")
except Exception as e:
    segmentation_model = guest_segmentation_model(None, None, None, None)
    print(f"❌ Failed to initialize guest segmentation model: {e}")

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
    
    print("✅ Successfully loaded BERT model and tokenizer")
except Exception as e:
    print(f"Error loading BERT model: {e}")
    bert_model = None
    bert_tokenizer = None


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

    class Config:
        schema_extra = {
            "example": {
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
                "Is_Weekend": False,
                "Is_Popular_Attraction": True
            }
        }

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

    class Config:
        schema_extra = {
            "example": {
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
        }

class ResourceAllocationRequest(BaseModel):
    ATTRACTION: str
    PARK: str
    WAIT_TIME_MAX: float
    NB_UNITS: float
    GUEST_CARRIED: float
    CAPACITY: float
    ADJUST_CAPACITY: float
    OPEN_TIME: float
    UP_TIME: float
    DOWNTIME: float
    NB_MAX_UNIT: float
    estimate_attendance: float
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

@app.post("/predict_demand/iot")
def predict_demand_iot(features_request: IOTFeaturesRequest):
    """Predict demand using IoT data model."""
    
    # convert features to appropriate types
    converted_features = list(features_request.dict().values())
    
    # reshape and validate feature dimensions
    try:
        input_data = np.array(converted_features).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reshaping features: {e}")

    model = models["demand_model_iot.pkl"]
    try:
        prediction = model.predict(input_data)
        print(f"Predicted queue time in minutes for IoT model: {prediction}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
    
    return {"model": "demand_model_iot.pkl", "predicted queue time (in mins)": prediction.tolist()}

@app.post("/predict_demand/survey_weather")
def predict_demand_survey_weather(features_request: SurveyWeatherFeaturesRequest):
    """Predict demand using survey and weather data model."""
    
    converted_features = list(features_request.dict().values())
    
    try:
        input_data = np.array(converted_features).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reshaping features: {e}")

    model = models["demand_model_survey_weather.pkl"]
    try:
        prediction = model.predict(input_data)
        print(f"Predicted queue time in minutes for Survey and Weather model: {prediction}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
    
    return {"model": "demand_model_survey_weather.pkl", "predicted queue time (in mins)": prediction.tolist()}

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

        
        updated_model = guest_segmentation_model(
            df_combined=encoded_df,
            df_labeled=combined_df,
            scaled=scaled,
            pca=pca
        )
        summary, df_labeled = updated_model.run_pipeline()

        # Segment naming 
        def assign_segment(row):
            if row["Age Group"] == "18 - 24 years old" and row["Visit Purpose"] == "Social gathering":
                return "Social-Driven Youths" if row["Express Pass Usage %"] > 50 else "Budget-Conscious Youths"
            elif row["Age Group"] == "25 - 34 years old" and row["Visit Purpose"] == "Family outing":
                return "Premium Spenders" if row["Express Pass Usage %"] > 50 else "Value-Conscious Families"
            else:
                return "Other"

        summary["Segment"] = summary.apply(assign_segment, axis=1)
        summary = summary[['Segment'] + [col for col in summary.columns if col != 'Segment']]

        
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

def predict_staff_count(model, new_data: pd.DataFrame):
    return np.ceil(model.predict(new_data)).astype(int)

def preprocess_new_data(new_data):
    """Preprocess new input data by removing categorical variables."""
    target_vars = ["staff_count", "reg_worker", "part_worker"]
    new_data = new_data.drop(columns=[col for col in target_vars if col in new_data.columns], errors='ignore')

    # Define the relative path to the training data
    training_data_path = "../data/processed data/q3_resource_allocation.csv"
    
    # Load the original training dataset
    training_data = pd.read_csv(training_data_path)
    
    # Initialize the encoder and fit it using the training dataset
    encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown='ignore')
    encoder.fit(training_data[["ATTRACTION", "PARK"]])
    
    # Transform the new data
    encoded_cats = encoder.transform(new_data[["ATTRACTION", "PARK"]])

    # Now, drop the original "ATTRACTION" and "PARK" columns and append the encoded columns
    new_data = new_data.drop(columns=["ATTRACTION", "PARK"]).join(pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out()))

    return new_data
    
@app.post("/predict_resource_allocation")
def predict_resource_allocation(requests: List[ResourceAllocationRequest]):
    model = models.get("q3_resource_allocation.pkl")
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Convert request list to DataFrame
    input_df = pd.DataFrame([r.dict() for r in requests])
    
    try:
        processed_df = preprocess_new_data(input_df)
        predictions = predict_staff_count(model, processed_df)
        input_df["Predicted_Staff_Count"] = predictions
        result = input_df[["year", "month", "Predicted_Staff_Count"]]
        return result.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/complaint_severity")
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

@app.post("/batch_complaints")
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

