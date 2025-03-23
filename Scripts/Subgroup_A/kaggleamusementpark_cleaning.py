import kagglehub
import pandas as pd
import os

# Download latest version of dataset
path = kagglehub.dataset_download("ayushtankha/hackathon")
print("Path to dataset files:", path)

#waiting times csv
waitingtimes_path = os.path.join(path, "waiting_times.csv")
waiting_times_df = pd.read_csv(waitingtimes_path)
#park to attraction link
parkattr_path = os.path.join(path, "link_attraction_park.csv")
parkattr_df = pd.read_csv(parkattr_path)
#attendance data
attendace_path = os.path.join(path, "attendance.csv")
attendance_df = pd.read_csv(attendace_path)

#cleaning and merging
waiting_times_df['DEB_TIME'] = pd.to_datetime(waiting_times_df['DEB_TIME'])
waiting_times_df['DEB_TIME'] = waiting_times_df['DEB_TIME'].dt.strftime('%H:%M')
waiting_times_df['GUEST_CARRIED'] = waiting_times_df['GUEST_CARRIED'].astype(int)
waiting_times_df['CAPACITY'] = waiting_times_df['CAPACITY'].round().astype(int)
waiting_times_df['ADJUST_CAPACITY'] = waiting_times_df['ADJUST_CAPACITY'].round().astype(int)

waitingtimes_oper = waiting_times_df[
    (waiting_times_df['GUEST_CARRIED'] != 0) & 
    (waiting_times_df['CAPACITY'] != 0)
]

parkattr_df[['ATTRACTION', 'PARK']] = parkattr_df['ATTRACTION;PARK'].str.split(";", expand=True)
parkattr_df = parkattr_df.drop(columns=['ATTRACTION;PARK'])
merged_df = pd.merge(waitingtimes_oper, parkattr_df, left_on='ENTITY_DESCRIPTION_SHORT', right_on='ATTRACTION', how='inner')
merged_df = merged_df.drop(columns=['ENTITY_DESCRIPTION_SHORT'])

tivoli_g = merged_df[merged_df['PARK'] == 'Tivoli Gardens']


