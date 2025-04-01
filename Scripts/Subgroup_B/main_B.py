import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def load_model(model_path):
    """Load the trained model from a pickle file."""
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

def preprocess_new_data(new_data, encoder_path):
    """Preprocess new input data for prediction."""
    # Load the one-hot encoder
    with open(encoder_path, 'rb') as enc_file:
        encoder = pickle.load(enc_file)
    
    # Apply one-hot encoding to categorical features
    encoded_cats = encoder.transform(new_data[["ATTRACTION", "PARK"]])
    new_data = new_data.drop(columns=["ATTRACTION", "PARK"]).join(pd.DataFrame(encoded_cats, 
                                                     columns=encoder.get_feature_names_out()))
    return new_data

def predict_staff_count(model, new_data):
    """Predict staff count using the trained model."""
    predictions = model.predict(new_data)
    return predictions

def main():
    """Main function to load the model, preprocess data, and make predictions."""
    # Load the trained model
    model = load_model("staff_count_model.pkl")
    
    # Load sample new data (replace with actual new data)
    new_data = pd.read_csv("new_data.csv")  # Ensure this file exists with correct format
    
    # Preprocess the new data
    new_data_processed = preprocess_new_data(new_data, "encoder.pkl")
    
    # Make predictions
    predicted_staff = predict_staff_count(model, new_data_processed)
    
    # Display results
    print("Predicted Staff Count for New Data:")
    print(predicted_staff)
    
    # Load the comparison results saved earlier
    with open("b_qn2_comparison.pkl", "rb") as f:
        comp = pickle.load(f)

    # Print: Current Layout
    print("\nQuestion 3: Optimization of Attraction Layout and Schedules")
    print("\nCurrent USS Layout (Two Entrances) - Multi Queue:")
    for attraction, time in comp["avg_wait_times_1_multi"].items():
        print(f"{attraction}: {time:.2f} min")
    print(f"Average Wait Time per Guest: {comp['avg_wait_per_guest_1']:.2f} min")
    print("Visit Counts:", comp["visit_counts_1_multi"])

    # Print: Modified Layout
    print("\nModified USS Layout (Left Entrance Only, Swapped Transformers and CYLON) - Multi Queue:")
    for attraction, time in comp["avg_wait_times_2_multi"].items():
        print(f"{attraction}: {time:.2f} min")
    print(f"Average Wait Time per Guest: {comp['avg_wait_per_guest_2']:.2f} min")
    print("Visit Counts:", comp["visit_counts_2_multi"])
    
if __name__ == "__main__":
    main()
