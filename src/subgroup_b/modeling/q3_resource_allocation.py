# Load and preprocess the staff dataset, which includes visitor counts, staff numbers, and sales data.
file_path = '../../../data/raw data/Staff.xls'
staff = pd.read_excel(file_path, header=6, sheet_name='月・実数')

# Remove unnecessary rows and columns to clean the dataset.
staff = staff.drop(index=0).drop(staff.columns[-1], axis=1)

# Rename columns for better readability.
staff.columns = ["time_code", "date", "total_sales", "admission_sales", "restaurant_sales", "total_visitors",
                 "individual_visitors", "group_visitors", "total_employees", "regular_workers", "part_time_workers"]

# Format the date column, extract year and month information.
staff = staff[staff["date"].str.match(r'^\d{4}年\s?\d{1,2}月$')].copy()
staff["date"] = pd.to_datetime(staff["date"].str.replace("年 ", "年", regex=False), format="%Y年%m月")
staff['date'] = staff['date'].dt.to_period('M')
staff['month'] = staff['date'].dt.month
staff['year'] = staff['date'].dt.year

# Load theme park datasets, including waiting times, attraction mappings, and attendance records.
path = kagglehub.dataset_download("ayushtankha/hackathon")
waiting_time = pd.read_csv(f"{path}/waiting_times.csv")
link_attraction = pd.read_csv(f"{path}/link_attraction_park.csv")
attendance = pd.read_csv(f"{path}/attendance.csv")

# Process the attraction-to-park mapping dataset.
link_attraction[['ATTRACTION', 'PARK']] = link_attraction['ATTRACTION;PARK'].str.split(';', expand=True)
link_attraction.drop(columns=['ATTRACTION;PARK'], inplace=True)

# Merge datasets to integrate waiting time data with park-attraction mappings.
merged_df = waiting_time.merge(link_attraction, left_on="ENTITY_DESCRIPTION_SHORT", right_on="ATTRACTION", how="left")

# Convert date columns to datetime format for consistency.
merged_df['WORK_DATE'] = pd.to_datetime(merged_df['WORK_DATE'])
attendance['USAGE_DATE'] = pd.to_datetime(attendance['USAGE_DATE'])

# Merge attendance data with the main dataset for a more complete view.
final_df = merged_df.merge(attendance, left_on=["WORK_DATE", "PARK"], right_on=["USAGE_DATE", "FACILITY_NAME"], how="right")

# Remove unnecessary columns related to attraction details and time logs.
final_df.drop(columns=["ENTITY_DESCRIPTION_SHORT", "USAGE_DATE", "FACILITY_NAME", 'DEB_TIME', 'DEB_TIME_HOUR',
                        'FIN_TIME'], inplace=True)

# Filter out records with missing or zero guest count.
final_df = final_df[final_df['GUEST_CARRIED'] > 0]

# Aggregate data to calculate the mean values per attraction and park.
avg_df = final_df.groupby(['WORK_DATE', 'ATTRACTION', 'PARK'], as_index=False).mean(numeric_only=True)
avg_df.iloc[:, 3:] = np.floor(avg_df.iloc[:, 3:])  # Round numeric values down.

# Compute total guest count per park per day.
guest_sum = avg_df.groupby(['WORK_DATE', 'PARK'])['GUEST_CARRIED'].sum().reset_index()
guest_sum.rename(columns={'GUEST_CARRIED': 'TOTAL_GUEST_CARRIED_PARK'}, inplace=True)

# Merge total guest count back into the dataset.
avg_df = avg_df.merge(guest_sum, on=['WORK_DATE', 'PARK'], how='left')

# Estimate attendance per ride using park-level attendance data.
avg_df["estimate_attendance"] = round(avg_df['GUEST_CARRIED'] / avg_df['TOTAL_GUEST_CARRIED_PARK'] * avg_df['attendance'])

# Drop redundant columns after estimation.
avg_df.drop(columns=['attendance', 'TOTAL_GUEST_CARRIED_PARK'], inplace=True)

# Convert the date to a period format for monthly analysis.
avg_df["year_month"] = avg_df["WORK_DATE"].dt.to_period("M")

# Merge estimated attendance data with the staff dataset.
dfa = pd.merge(avg_df, staff, left_on="year_month", right_on="date", how="left").drop(columns=['date', 'time_code', 'year_month'])

# Feature engineering: Estimate sales and required workforce for theme parks.
dfa["sale"] = dfa["estimate_attendance"] / (dfa["total_visitors"] / dfa["total_sales"])
dfa["adm_sale"] = dfa["estimate_attendance"] / (dfa["total_visitors"] / dfa["admission_sales"])
dfa["rest_sale"] = dfa["estimate_attendance"] / (dfa["total_visitors"] / dfa["restaurant_sales"])

# Estimate the number of workers required based on visitor counts.
dfa["reg_worker"] = np.ceil(dfa["estimate_attendance"] / (dfa["total_visitors"] / dfa["regular_workers"]))
dfa["part_worker"] = np.ceil(dfa["estimate_attendance"] / (dfa["total_visitors"] / dfa["part_time_workers"]))
dfa["staff_count"] = dfa["part_worker"] + dfa["reg_worker"]

# Remove original sales and staff columns after estimation.
dfa.drop(columns=["total_sales", "admission_sales", "restaurant_sales", "total_visitors", "individual_visitors", 
                   "group_visitors", "total_employees", "regular_workers", "part_time_workers"], inplace=True)

# Drop the WORK_DATE column as it is no longer needed.
df = dfa.drop(columns=["WORK_DATE"])

# Save the cleaned dataset for further analysis.
df.to_csv('q3_staff_allocation.csv', index=False)

# Encode categorical variables (ATTRACTION and PARK) using one-hot encoding.
encoder = OneHotEncoder(drop="first", sparse_output=False)
encoded_cats = encoder.fit_transform(df[["ATTRACTION", "PARK"]])
df = df.drop(columns=["ATTRACTION", "PARK"]).join(pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out()))

# Define feature variables (X) and target variables (y) for prediction.
X = df.drop(columns=["staff_count", "reg_worker", "part_worker"])
y_staff, y_reg_worker, y_part_worker = df["staff_count"], df["reg_worker"], df["part_worker"]

# Split the dataset into training and testing sets.
X_train, X_test, y_staff_train, y_staff_test = train_test_split(X, y_staff, test_size=0.2, random_state=42)

# Train a Random Forest model to predict staff count.
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_staff_train)

# Perform cross-validation to evaluate model performance.
cv_scores = cross_val_score(model, X_train, y_staff_train, cv=5, scoring='r2')
print(f"Cross-validation R² scores: {cv_scores}")
print(f"Mean CV R² score: {cv_scores.mean():.2f}")

# Generate predictions on the test dataset.
y_pred = model.predict(X_test)

# Define a function to evaluate model performance.
def evaluate(y_true, y_pred, target_name):
    print(f"\n--- {target_name} Prediction Evaluation ---")
    print(f"MAE: {mean_absolute_error(y_true, y_pred):.2f}")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.2f}")
    print(f"R² Score: {r2_score(y_true, y_pred):.2f}")

# Evaluate the trained model on staff count predictions.
evaluate(y_staff_test, y_pred, "Staff Count")

# Save the trained model for future use.
with open("q3_resource_allocation.pkl", "wb") as model_file:
    pickle.dump(model, model_file)
print("Model saved.")

# Visualization: Uncomment to plot actual vs predicted staff count.
# plt.figure(figsize=(10, 5))
# sns.scatterplot(x=y_staff_test, y=y_pred)
# plt.xlabel("Actual Staff Count")
# plt.ylabel("Predicted Staff Count")
# plt.title("Actual vs Predicted Staff Count")
# plt.show()
