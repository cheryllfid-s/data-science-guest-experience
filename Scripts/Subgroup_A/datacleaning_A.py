import kagglehub
import pandas as pd
import os

# (Q3) data journey analysis 
def prepare_tivoli_data():
    # Download latest version of disney dataset
    path = kagglehub.dataset_download("ayushtankha/hackathon")
    print("Path to dataset files:", path)

    waitingtimes_path = os.path.join(path, "waiting_times.csv")
    waiting_times_df = pd.read_csv(waitingtimes_path)

    parkattr_path = os.path.join(path, "link_attraction_park.csv")
    parkattr_df = pd.read_csv(parkattr_path)

    attendace_path = os.path.join(path, "attendance.csv")
    attendance_df = pd.read_csv(attendace_path)

    # FORMATTING THE DATA
    # (1) Turn the deb_time into datetime objects and remove seconds (they are all 0)
    waiting_times_df['DEB_TIME'] = pd.to_datetime(waiting_times_df['DEB_TIME'])
    waiting_times_df['DEB_TIME'] = waiting_times_df['DEB_TIME'].dt.strftime('%H:%M')

    # (2) Turn guest_carried into integers + round off capacity and adjust capacity to their nearest integers
    waiting_times_df['GUEST_CARRIED'] = waiting_times_df['GUEST_CARRIED'].astype(int)
    waiting_times_df['CAPACITY'] = waiting_times_df['CAPACITY'].round().astype(int)
    waiting_times_df['ADJUST_CAPACITY'] = waiting_times_df['ADJUST_CAPACITY'].round().astype(int)

    # (3) Sorting based on the date and time of debarkation
    waiting_times_df = waiting_times_df.sort_values(['WORK_DATE', 'DEB_TIME'])

    # Filter out data where rides were not operational
    waitingtimes_oper = waiting_times_df[
        (waiting_times_df['GUEST_CARRIED'] != 0) & 
        (waiting_times_df['CAPACITY'] != 0)
    ]

    waitingtimes_oper = waitingtimes_oper.sort_values(['WORK_DATE', 'DEB_TIME', 'WAIT_TIME_MAX'], ascending=[True, True, False])
    waitingtimes_oper = waitingtimes_oper.drop(columns=['DEB_TIME_HOUR', 'FIN_TIME', 'NB_UNITS', 'OPEN_TIME', 'UP_TIME', 
                                                        'NB_MAX_UNIT', 'DOWNTIME', 'GUEST_CARRIED', 'CAPACITY', 'ADJUST_CAPACITY'])

    # Merge with link_attraction_park.csv
    parkattr_df[['ATTRACTION', 'PARK']] = parkattr_df['ATTRACTION;PARK'].str.split(";", expand=True)
    parkattr_df = parkattr_df.drop(columns=['ATTRACTION;PARK'])
    merged_df = pd.merge(waitingtimes_oper, parkattr_df, left_on='ENTITY_DESCRIPTION_SHORT', right_on='ATTRACTION', how='inner')
    merged_df = merged_df.drop(columns=['ENTITY_DESCRIPTION_SHORT'])

    # Filter for Tivoli Gardens
    tivoli_g = merged_df[merged_df['PARK'] == 'Tivoli Gardens']
    tivoli_g['WORK_DATE'] = pd.to_datetime(tivoli_g['WORK_DATE'])
    tivoli_g['TIMESTAMP'] = pd.to_datetime(tivoli_g['WORK_DATE'].astype(str) + ' ' + tivoli_g['DEB_TIME'])

    # Finding COVID dates
    attendance_df['USAGE_DATE'] = pd.to_datetime(attendance_df['USAGE_DATE'])
    covid = attendance_df[(attendance_df['USAGE_DATE'] >= '2020-03-01') & (attendance_df['USAGE_DATE'] <= '2021-08-01')]

    negative_att = attendance_df[attendance_df['attendance'] < 0]

    return tivoli_g, attendance_df, covid, negative_att
