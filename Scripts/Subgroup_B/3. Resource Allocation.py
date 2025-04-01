import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the first dataset (Staff Data)
# This dataset contains information on visitor counts, staff numbers, and sales figures.
file_path = "../../data/hv15401j.xls"  
staff = pd.read_excel(file_path, header=6, sheet_name='月・実数')

# Clean the dataset by removing unnecessary rows and columns
staff = staff.drop(index=0).drop(staff.columns[-1], axis=1)

# Rename columns for better readability
staff.columns = ["time_code", "date", "total_sales", "admission_sales", "restaurant_sales", "total_visitors",
                 "individual_visitors", "group_visitors", "total_employees", "regular_workers", "part_time_workers"]

# Format the date column and extract year and month
staff = staff[staff["date"].str.match(r'^\d{4}年\s?\d{1,2}月$')].copy()
staff["date"] = pd.to_datetime(staff["date"].str.replace("年 ", "年", regex=False), format="%Y年%m月")
staff['date'] = staff['date'].dt.to_period('M')
staff['month'] = staff['date'].dt.month
staff['year'] = staff['date'].dt.year

# Load additional datasets (Theme Park Data)
path = kagglehub.dataset_download("ayushtankha/hackathon")
waiting_time = pd.read_csv(f"{path}/waiting_times.csv")
link_attraction = pd.read_csv(f"{path}/link_attraction_park.csv")
attendance = pd.read_csv(f"{path}/attendance.csv")

# Process attraction-to-park mapping dataset
link_attraction[['ATTRACTION', 'PARK']] = link_attraction['ATTRACTION;PARK'].str.split(';', expand=True)
link_attraction.drop(columns=['ATTRACTION;PARK'], inplace=True)

# Merge datasets to combine waiting time data with park-attraction mapping
merged_df = waiting_time.merge(link_attraction, left_on="ENTITY_DESCRIPTION_SHORT", right_on="ATTRACTION", how="left")

# Convert date columns to datetime format
merged_df['WORK_DATE'] = pd.to_datetime(merged_df['WORK_DATE'])
attendance['USAGE_DATE'] = pd.to_datetime(attendance['USAGE_DATE'])

# Merge attendance data with the main dataset
final_df = merged_df.merge(attendance, left_on=["WORK_DATE", "PARK"], right_on=["USAGE_DATE", "FACILITY_NAME"], how="right")

# Drop unnecessary columns related to attraction details and time logs
final_df.drop(columns=["ENTITY_DESCRIPTION_SHORT", "USAGE_DATE", "FACILITY_NAME", 'DEB_TIME', 'DEB_TIME_HOUR',
                        'FIN_TIME', 'WAIT_TIME_MAX'], inplace=True)

# Filter out records where guest count is missing or zero
final_df = final_df[final_df['GUEST_CARRIED'] > 0]

# Aggregate data to compute mean values per attraction and park
avg_df = final_df.groupby(['WORK_DATE', 'ATTRACTION', 'PARK'], as_index=False).mean(numeric_only=True)
avg_df.iloc[:, 3:] = np.floor(avg_df.iloc[:, 3:])  # Round numeric values

# Compute total guest count per park per day
guest_sum = avg_df.groupby(['WORK_DATE', 'PARK'])['GUEST_CARRIED'].sum().reset_index()
guest_sum.rename(columns={'GUEST_CARRIED': 'TOTAL_GUEST_CARRIED_PARK'}, inplace=True)

# Merge total guest count back into the dataset
avg_df = avg_df.merge(guest_sum, on=['WORK_DATE', 'PARK'], how='left')

# Estimate attendance per ride using the available park-level attendance data
avg_df["estimate_attendance"] = round(avg_df['GUEST_CARRIED'] / avg_df['TOTAL_GUEST_CARRIED_PARK'] * avg_df['attendance'])

# Drop redundant columns after estimation
avg_df.drop(columns=['attendance', 'TOTAL_GUEST_CARRIED_PARK'], inplace=True)

# Convert date to period format for monthly analysis
avg_df["year_month"] = avg_df["WORK_DATE"].dt.to_period("M")

# Merge estimated attendance data with staff dataset
dfa = pd.merge(avg_df, staff, left_on="year_month", right_on="date", how="left").drop(columns=['date', 'time_code', 'year_month'])

# Feature Engineering: Estimate sales and worker counts for theme parks
dfa["sale"] = dfa["estimate_attendance"] / (dfa["total_visitors"] / dfa["total_sales"])
dfa["adm_sale"] = dfa["estimate_attendance"] / (dfa["total_visitors"] / dfa["admission_sales"])
dfa["rest_sale"] = dfa["estimate_attendance"] / (dfa["total_visitors"] / dfa["restaurant_sales"])

# Estimate the number of workers required based on visitor counts
dfa["reg_worker"] = np.ceil(dfa["estimate_attendance"] / (dfa["total_visitors"] / dfa["regular_workers"]))
dfa["part_worker"] = np.ceil(dfa["estimate_attendance"] / (dfa["total_visitors"] / dfa["part_time_workers"]))
dfa["staff_count"] = dfa["part_worker"] + dfa["reg_worker"]

# Drop original sales and staff columns after estimation
dfa.drop(columns=["total_sales", "admission_sales", "restaurant_sales", "total_visitors", "individual_visitors", 
                   "group_visitors", "total_employees", "regular_workers", "part_time_workers"], inplace=True)

# Drop WORK_DATE column as it is no longer needed
df = dfa.drop(columns=["WORK_DATE"])

# Encode categorical variables (ATTRACTION and PARK) using one-hot encoding
encoder = OneHotEncoder(drop="first", sparse=False)
encoded_cats = encoder.fit_transform(df[["ATTRACTION", "PARK"]])
df = df.drop(columns=["ATTRACTION", "PARK"]).join(pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out()))

# Define features and target variables for prediction
X = df.drop(columns=["staff_count", "reg_worker", "part_worker"])
y_staff, y_reg_worker, y_part_worker = df["staff_count"], df["reg_worker"], df["part_worker"]

# Split data into training and testing sets
X_train, X_test, y_staff_train, y_staff_test = train_test_split(X, y_staff, test_size=0.2, random_state=42)

# Train a Random Forest model to predict staff count
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_staff_train)

# Perform cross-validation to assess model performance
cv_scores = cross_val_score(model, X_train, y_staff_train, cv=5, scoring='r2')
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean CV R² score: {cv_scores.mean():.2f}")

# Generate predictions on the test set
y_pred = model.predict(X_test)

# Define an evaluation function for model performance
def evaluate(y_true, y_pred, target_name):
    print(f"\n--- {target_name} Prediction Evaluation ---")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.2f}")

evaluate(y_staff_test, y_pred, "Staff Count")

# Save the trained model
with open("staff_count_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
print("Model saved as 'staff_count_model.pkl'.")

# Visualization of actual vs predicted staff count
# plt.figure(figsize=(10, 5))
# sns.scatterplot(x=y_staff_test, y=y_pred)
# plt.xlabel("Actual Staff Count")
# plt.ylabel("Predicted Staff Count")
# plt.title("Actual vs Predicted Staff Count")
# plt.show()
