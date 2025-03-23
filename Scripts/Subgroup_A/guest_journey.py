import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import networkx as nx

# Download latest version of disney dataset
path = kagglehub.dataset_download("ayushtankha/hackathon")
print("Path to dataset files:", path)

waitingtimes_path = os.path.join(path, "waiting_times.csv")
waiting_times_df = pd.read_csv(waitingtimes_path)

parkattr_path = os.path.join(path, "link_attraction_park.csv")
parkattr_df = pd.read_csv(parkattr_path)

attendace_path = os.path.join(path, "attendance.csv")
attendance_df = pd.read_csv(attendace_path)

#FORMATTING THE DATA
#(1) turn the deb_time into datetime objects and remove seconds (they are all 0)
waiting_times_df['DEB_TIME'] = pd.to_datetime(waiting_times_df['DEB_TIME'])
waiting_times_df['DEB_TIME'] = waiting_times_df['DEB_TIME'].dt.strftime('%H:%M')

#(2) turn guest_carried into integers + round off capacity and adjust capacity to their nearest integers
waiting_times_df['GUEST_CARRIED'] = waiting_times_df['GUEST_CARRIED'].astype(int)
waiting_times_df['CAPACITY'] = waiting_times_df['CAPACITY'].round().astype(int)
waiting_times_df['ADJUST_CAPACITY'] = waiting_times_df['ADJUST_CAPACITY'].round().astype(int)

#(3)sorting based on the date and time of debarkation 
waiting_times_df = waiting_times_df.sort_values(['WORK_DATE', 'DEB_TIME'])

#some data has guest_carried = capacity = wait time max = 0 -- ride was not operational
#for common journey paths, we want to analyse ones where all rides were operational 

waitingtimes_oper = waiting_times_df[
    (waiting_times_df['GUEST_CARRIED'] != 0) & 
    (waiting_times_df['CAPACITY'] != 0)
]

waitingtimes_oper = waitingtimes_oper.sort_values(['WORK_DATE','DEB_TIME', 'WAIT_TIME_MAX'],ascending=[True, True, False])
WTsorted_nonzero = waitingtimes_oper.sort_values(['WAIT_TIME_MAX'],ascending=False)  
waitingtimes_oper = waitingtimes_oper.drop(columns=['DEB_TIME_HOUR','FIN_TIME', 'NB_UNITS', 'OPEN_TIME', 'UP_TIME','NB_MAX_UNIT', 'DOWNTIME'])


#merge with link_attraction_park.csv
parkattr_df[['ATTRACTION', 'PARK']] = parkattr_df['ATTRACTION;PARK'].str.split(";", expand=True)
parkattr_df = parkattr_df.drop(columns=['ATTRACTION;PARK'])
merged_df = pd.merge(waitingtimes_oper, parkattr_df, left_on='ENTITY_DESCRIPTION_SHORT', right_on='ATTRACTION', how='inner')
merged_df = merged_df.drop(columns=['ENTITY_DESCRIPTION_SHORT'])

#dividing between parks
tivoli_g = merged_df[merged_df['PARK'] == 'Tivoli Gardens']
tivoli_g['WORK_DATE'] = pd.to_datetime(tivoli_g['WORK_DATE'])
tivoli_g['TIMESTAMP'] = pd.to_datetime(tivoli_g['WORK_DATE'].astype(str) + ' ' + tivoli_g['DEB_TIME'])

#finding covid dates

attendance_df['USAGE_DATE'] = pd.to_datetime(attendance_df['USAGE_DATE'])
covid = attendance_df[(attendance_df['USAGE_DATE'] >= '2020-03-01') & (attendance_df['USAGE_DATE'] <= '2021-08-01')]

plt.plot(covid['USAGE_DATE'], covid['attendance'])
plt.xlabel('USAGE_DATE')
plt.ylabel('attendance')
plt.title('Attendance over time (2019 to mid 2021)')
plt.show()

#total negative data
negative_att = attendance_df[attendance_df['attendance'] < 0]

#noncovid data
noncovid = attendance_df[(attendance_df['USAGE_DATE'] < '2020-03-01') | (attendance_df['USAGE_DATE'] > '2021-08-01')]
print(noncovid)
noncovidneg = noncovid[noncovid['attendance'] < 0] #noncovid negatives = 0 
noncovid = noncovid[noncovid['attendance'] > 0]

#Separating noncovid data based on 'FACILITY_NAME'
tivoli_noncovid = noncovid[noncovid['FACILITY_NAME'] == 'Tivoli Gardens']

# Calculate the median attendance
tivoli_median_attendance = tivoli_noncovid['attendance'].median()

# Get the two rows corresponding to the median attendance
tivoli_median_rows = tivoli_noncovid.iloc[(tivoli_noncovid['attendance'] - tivoli_median_attendance).abs().argsort()[:2]]

print("\nTivoli Gardens median attendance rows:")
print(tivoli_median_rows)

tivoli_g_all = tivoli_g #keep for later

#finding covid dates

attendance_df['USAGE_DATE'] = pd.to_datetime(attendance_df['USAGE_DATE'])
covid = attendance_df[(attendance_df['USAGE_DATE'] >= '2020-03-01') & (attendance_df['USAGE_DATE'] <= '2021-08-01')]

plt.plot(covid['USAGE_DATE'], covid['attendance'])
plt.xlabel('USAGE_DATE')
plt.ylabel('attendance')
plt.title('Attendance over time (2019 to mid 2021)')
plt.show()

#total negative data
negative_att = attendance_df[attendance_df['attendance'] < 0]

#noncovid data
noncovid = attendance_df[(attendance_df['USAGE_DATE'] < '2020-03-01') | (attendance_df['USAGE_DATE'] > '2021-08-01')]
print(noncovid)
noncovidneg = noncovid[noncovid['attendance'] < 0] #noncovid negatives = 0 
noncovid = noncovid[noncovid['attendance'] > 0]

#Separating noncovid data based on 'FACILITY_NAME'
tivoli_noncovid = noncovid[noncovid['FACILITY_NAME'] == 'Tivoli Gardens']

# Calculate the median attendance
tivoli_median_attendance = tivoli_noncovid['attendance'].median()

# Get the two rows corresponding to the median attendance
tivoli_median_rows = tivoli_noncovid.iloc[(tivoli_noncovid['attendance'] - tivoli_median_attendance).abs().argsort()[:2]]

print("\nTivoli Gardens median attendance rows:")
print(tivoli_median_rows)

tivoli_g_all = tivoli_g #keep for later

#line graph plotting for median wait times against time of day:
def plot_median_wait_times(df, title_suffix=""):
    median_wait_times = df.groupby(['DEB_TIME', 'ATTRACTION'])['WAIT_TIME_MAX'].median().unstack()
    plt.figure(figsize=(12, 6))
    colors = plt.cm.get_cmap('tab20', len(median_wait_times.columns))

    for i, attraction in enumerate(median_wait_times.columns):
        plt.plot(median_wait_times.index, median_wait_times[attraction], label=attraction, color=colors(i), marker='o', markersize=4)

    plt.xlabel('Time of Day')
    plt.ylabel('Median Waiting Time (minutes)')
    plt.title(f'Median Waiting Time Throughout the Day {title_suffix} - Tivoli Gardens')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(False)
    plt.xticks(median_wait_times.index[::4], rotation=45)  # Show x-axis labels per hour
    plt.show()

#correlation matrix between all rides function:
def plot_ride_correlation_heatmap(df, title_suffix=""):

    tivoli_pivot = df.pivot(index='TIMESTAMP', columns='ATTRACTION', values='WAIT_TIME_MAX')
    correlation_matrix = tivoli_pivot.corr()

    plt.figure(figsize=(12, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(f"Correlation Between Ride Waiting Times {title_suffix}")
    plt.show()


#correlation matrix between rides (family) (youth/adults) function:
def analyze_ride_correlations(df, title_suffix="", reference_ride=""):
    tivoli_pivot = df.pivot(index='TIMESTAMP', columns='ATTRACTION', values='WAIT_TIME_MAX')
    correlation_matrix = tivoli_pivot.corr()

    reference_correlations = correlation_matrix[reference_ride]

    family_rides = []
    youth_rides = []

    for ride, correlation in reference_correlations.items():
        if correlation < 0.5:
            family_rides.append(ride)
        else:
            youth_rides.append(ride)

    print("Family Rides:", family_rides)
    print("Youth/Young Adult Rides:", youth_rides)

    family_corr = correlation_matrix.loc[family_rides, family_rides]
    youth_corr = correlation_matrix.loc[youth_rides, youth_rides]

    plt.figure(figsize=(8, 6))
    sns.heatmap(family_corr, annot=True, cmap='coolwarm')
    plt.title(f"Correlation Between Family Rides {title_suffix}")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(youth_corr, annot=True, cmap='coolwarm')
    plt.title(f"Correlation Between Youth/Young Adult Rides {title_suffix}")
    plt.show()

#analysing (possible) guest movements:
def analyze_guest_movement(df, reference_ride=''):

    tivoli_pivot = df.pivot(index='TIMESTAMP', columns='ATTRACTION', values='WAIT_TIME_MAX')
    correlation_matrix = tivoli_pivot.corr()

    reference_correlations = correlation_matrix[reference_ride]

    family_rides = []
    youth_rides = []

    for ride, correlation in reference_correlations.items():
        if correlation < 0.5:
            family_rides.append(ride)
        else:
            youth_rides.append(ride)

    # Family Rides Movement Analysis
    family_pivot = tivoli_pivot[family_rides]
    family_shifted = family_pivot.shift(1)
    family_changes = family_pivot - family_shifted

    family_movement_counts = []

    for from_ride in family_changes.columns:
        for to_ride in family_changes.columns:
            if from_ride != to_ride:
                movement = ((family_changes[from_ride] < 0) & (family_changes[to_ride] > 0)).sum()
                family_movement_counts.append({"from": from_ride, "to": to_ride, "count": movement})

    fam_movements = pd.DataFrame(family_movement_counts)

    # Youth Rides Movement Analysis
    youth_pivot = tivoli_pivot[youth_rides]
    youth_shifted = youth_pivot.shift(1)
    youth_changes = youth_pivot - youth_shifted

    youth_movement_counts = []

    for from_ride in youth_changes.columns:
        for to_ride in youth_changes.columns:
            if from_ride != to_ride:
                movement = ((youth_changes[from_ride] < 0) & (youth_changes[to_ride] > 0)).sum()
                youth_movement_counts.append({"from": from_ride, "to": to_ride, "count": movement})

    youth_adult_movements = pd.DataFrame(youth_movement_counts)

    return fam_movements, youth_adult_movements

def analyze_guest_movement_covid(df):
    tivoli_pivot = df.pivot(index='TIMESTAMP', columns='ATTRACTION', values='WAIT_TIME_MAX')
    ride_changes = tivoli_pivot.diff() 

    movement_counts = []

    for from_ride in ride_changes.columns:
        for to_ride in ride_changes.columns:
            if from_ride != to_ride:
                movement = ((ride_changes[from_ride] < 0) & (ride_changes[to_ride] > 0)).sum()
                movement_counts.append({"from": from_ride, "to": to_ride, "count": movement})

    movements_df = pd.DataFrame(movement_counts)

    return movements_df

def calculate_avg_wait_times(df):
    wait = df.groupby('ATTRACTION')['WAIT_TIME_MAX'].mean().to_dict()
    wait_df = pd.DataFrame(list(wait.items()), columns=['Ride', 'Avg_Wait_Time'])
    return wait_df

def plot_guest_transitions_vs_wait_time(movement_df, waitdf, title, group_name="Families"):

    merged_df = movement_df.merge(waitdf, left_on='from', right_on='Ride').drop(columns=['Ride'])

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=merged_df, x='Avg_Wait_Time', y='count', marker='o')
    plt.xlabel('Average Waiting Time (mins)')
    plt.ylabel('Guest Movement Count')
    plt.title(f'{title} -- {group_name}')
    plt.show()

def markov_chain_analysis(df, title):
    graph = nx.DiGraph()
    for _, row in df.iterrows():
        graph.add_edge(row['from'], row['to'], weight=row['count'])

    rides = list(graph.nodes())
    transition_matrix = np.zeros((len(rides), len(rides)))

    for i, from_ride in enumerate(rides):
        total_outflow = sum(graph.get_edge_data(from_ride, to_ride)['weight'] for to_ride in graph.successors(from_ride))
        if total_outflow > 0:
            for j, to_ride in enumerate(rides):
                if graph.has_edge(from_ride, to_ride):
                    transition_matrix[i, j] = graph.get_edge_data(from_ride, to_ride)['weight'] / total_outflow

    eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
    steady_state = np.abs(eigenvectors[:, np.argmax(eigenvalues)])
    steady_state /= steady_state.sum()

    steady_state_distribution = pd.Series(steady_state, index=rides)

    print(f"\n{title} - Steady-State Distribution:")
    print(steady_state_distribution)

noncovid_dates = noncovid['USAGE_DATE']
tivoli_g = tivoli_g[tivoli_g['WORK_DATE'].isin(noncovid_dates)]

#common guest journey analysis
plot_median_wait_times(tivoli_g, "")
plot_ride_correlation_heatmap(tivoli_g, title_suffix="")

analyze_ride_correlations(tivoli_g, title_suffix="",reference_ride ="Scooby Doo")

fam_movements, youth_adult_movements = analyze_guest_movement(tivoli_g, 'Scooby Doo')

wait_regular = calculate_avg_wait_times(tivoli_g)
plot_guest_transitions_vs_wait_time(fam_movements, wait_regular, 'Family Ride Guest Journey', group_name="Families")
plot_guest_transitions_vs_wait_time(youth_adult_movements, wait_regular, 'Thrill Ride Guest Journey', group_name="Youth/Young Adults")

family_outflow = fam_movements.groupby('from')['count'].sum().sort_values(ascending=False)
print("Family Rides - Highest Outgoing Traffic:")
print(family_outflow)

youth_outflow = youth_adult_movements.groupby('from')['count'].sum().sort_values(ascending=False)
print("\nYouth Rides - Highest Outgoing Traffic:")
print(youth_outflow)

print("\nFamily Rides - Most Frequent Transitions:")
print(fam_movements.sort_values(by='count', ascending=False))

print("\nYouth Rides - Most Frequent Transitions:")
print(youth_adult_movements.sort_values(by='count', ascending=False))

markov_chain_analysis(fam_movements, "Family Rides")
markov_chain_analysis(youth_adult_movements, "Youth Rides")

tivoli_attendance_df = attendance_df[attendance_df['FACILITY_NAME'] == 'Tivoli Gardens']
daily_attendance_tivoli = tivoli_attendance_df.groupby(tivoli_attendance_df['USAGE_DATE'].dt.date)['attendance'].sum()

busy_threshold = daily_attendance_tivoli.quantile(0.75)
print("Cutoff attendance for busy days:", busy_threshold)

# busy days
busy_days = daily_attendance_tivoli[daily_attendance_tivoli > busy_threshold].index
tivoli_busy = tivoli_g[tivoli_g['WORK_DATE'].isin(busy_days)]

plot_median_wait_times(tivoli_busy, "when busy")
plot_ride_correlation_heatmap(tivoli_busy, title_suffix="when busy") 

analyze_ride_correlations(tivoli_busy, title_suffix="when busy", reference_ride="Scooby Doo")

fam_movements_busy, youth_adult_movements_busy = analyze_guest_movement(tivoli_busy, 'Scooby Doo')
wait_busy = calculate_avg_wait_times(tivoli_busy)
plot_guest_transitions_vs_wait_time(fam_movements_busy, wait_busy, 'Ride Guest Journey During Busy Days', group_name="Families")
plot_guest_transitions_vs_wait_time(youth_adult_movements_busy, wait_busy, 'Ride Guest Journey During Busy Days', group_name="Youth/Young Adults")

family_outflow_busy = fam_movements_busy.groupby('from')['count'].sum().sort_values(ascending=False)
print("Family Rides during Busy Days - Highest Outgoing Traffic:")
print(family_outflow_busy)

youth_outflow = youth_adult_movements_busy.groupby('from')['count'].sum().sort_values(ascending=False)
print("\nYouth Rides during Busy Days- Highest Outgoing Traffic:")
print(youth_outflow)

print("\nFamily Rides during Busy Days - Most Frequent Transitions:")
print(fam_movements_busy.sort_values(by='count', ascending=False))

print("\nYouth Rides during Busy Days - Most Frequent Transitions:")
print(youth_adult_movements_busy.sort_values(by='count', ascending=False))

markov_chain_analysis(fam_movements_busy, "Family Rides during Busy Days")
markov_chain_analysis(youth_adult_movements_busy, "Youth Rides during Busy Days")

quiet_threshold = daily_attendance_tivoli.quantile(0.25)
print("Cutoff attendance for quiet days:", quiet_threshold)

# quiet days
quiet_days = daily_attendance_tivoli[daily_attendance_tivoli < quiet_threshold].index
tivoli_quiet = tivoli_g[tivoli_g['WORK_DATE'].isin(quiet_days)]

plot_median_wait_times(tivoli_quiet, "")
plot_ride_correlation_heatmap(tivoli_quiet, title_suffix="")

analyze_ride_correlations(tivoli_quiet, title_suffix="",reference_ride ="Scooby Doo")

fam_movements_quiet, youth_adult_movements_quiet = analyze_guest_movement(tivoli_quiet, 'Scooby Doo')
wait_regular = calculate_avg_wait_times(tivoli_quiet)
plot_guest_transitions_vs_wait_time(fam_movements, wait_regular, 'Ride Guest Journey During Quiet Days', group_name="Families")
plot_guest_transitions_vs_wait_time(youth_adult_movements, wait_regular, 'Ride Guest Journey During Quiet Days', group_name="Youth/Young Adults")

family_outflow_quiet = fam_movements_quiet.groupby('from')['count'].sum().sort_values(ascending=False)
print("Family Rides during Quiet Days - Highest Outgoing Traffic:")
print(family_outflow)

youth_outflow = youth_adult_movements_quiet.groupby('from')['count'].sum().sort_values(ascending=False)
print("\nYouth Rides during Quite Days- Highest Outgoing Traffic:")
print(youth_outflow)

print("\nFamily Rides during Quite Days - Most Frequent Transitions:")
print(fam_movements_quiet.sort_values(by='count', ascending=False))

print("\nYouth Rides during Quiet Days - Most Frequent Transitions:")
print(youth_adult_movements_quiet.sort_values(by='count', ascending=False))

markov_chain_analysis(fam_movements_quiet, "Family Rides during Quiet Days")
markov_chain_analysis(youth_adult_movements_quiet, "Youth Rides during Quiet Days")

#COVID-19

covid_dates = covid['USAGE_DATE']
covid_tivoli = tivoli_g_all[tivoli_g_all['WORK_DATE'].isin(covid_dates)]

plot_median_wait_times(tivoli_quiet, "")
plot_ride_correlation_heatmap(tivoli_quiet, title_suffix="")

covid_movement = analyze_guest_movement_covid(tivoli_quiet)
wait_covid = calculate_avg_wait_times(covid_tivoli)
plot_guest_transitions_vs_wait_time(covid_movement, wait_covid, 'Ride Guest Journeys during Covid', group_name="")

covid_outflow = covid_movement.groupby('from')['count'].sum().sort_values(ascending=False)
print("\nRides During the Covid-19 Pandemic -- Highest Outgoing Traffic")
print(covid_outflow)

print("\nFRides during the Covid-19 Pandemic - Most Frequent Transitions:")
print(covid_movement.sort_values(by='count', ascending=False))

markov_chain_analysis(covid_movement, "Rides during the Covid-19 Pandemic")
