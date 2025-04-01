import kagglehub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import networkx as nx
from difflib import SequenceMatcher
from itertools import combinations
from scipy.stats import entropy

class GuestJourneyAnalysis:  #Object to make it easier to put into main.py
    def __init__(self, tivoli_g, attendance_df, covid, negative_att):
        self.tivoli_g = tivoli_g
        self.attendance_df = attendance_df  
        self.covid = covid
        self.negative_att = negative_att

    def label_express_pass_by_daytype(self, df, threshold=0.3): 
        #Dividing by day type, using the wait time max to see if there are people using express tickets for certain rides
        #Since quiet, busy, covid, and normal days have different wait times 
        thresholds = df.groupby(['ATTRACTION', 'DAY_TYPE'])['WAIT_TIME_MAX'].quantile(threshold).to_dict()
        
        def is_express(row):
            key = (row['ATTRACTION'], row['DAY_TYPE'])
            return row['WAIT_TIME_MAX'] < thresholds.get(key, np.inf)
        
        df['EXPRESS_PASS'] = df.apply(is_express, axis=1)
        return df

    #Dividing the day types
    def determine_day_types(self, attendance_df, covid, facility_name='Tivoli Gardens'):
        tivoli_attendance_df = attendance_df[attendance_df['FACILITY_NAME'] == facility_name]
        daily_attendance = tivoli_attendance_df.groupby(tivoli_attendance_df['USAGE_DATE'].dt.date)['attendance'].sum()

        busy_threshold = daily_attendance.quantile(0.75)
        quiet_threshold = daily_attendance.quantile(0.25)

        busy_days = daily_attendance[daily_attendance > busy_threshold].index
        quiet_days = daily_attendance[daily_attendance < quiet_threshold].index
        covid_dates = self.covid['USAGE_DATE'].drop_duplicates().dt.date
        return busy_days, quiet_days, covid_dates

    #Tagging the day types
    def tag_day_type(self, df, covid_dates, busy_days, quiet_days):
        df = df.copy()
        df.loc[:, 'DAY_TYPE'] = 'normal'
        df.loc[df['WORK_DATE'].isin(covid_dates), 'DAY_TYPE'] = 'covid'
        df.loc[df['WORK_DATE'].isin(covid_dates), 'DAY_TYPE'] = 'covid'
        df.loc[df['WORK_DATE'].isin(busy_days), 'DAY_TYPE'] = 'busy'
        df.loc[df['WORK_DATE'].isin(quiet_days), 'DAY_TYPE'] = 'quiet'
        return df


    def simulate_guest_sequences(self, df, time_gap_seconds=1800):
        df = df.sort_values(by='TIMESTAMP')
        df['GAP'] = df['TIMESTAMP'].diff().dt.total_seconds().fillna(0) #Time gaps in seconds
        df['GUEST_ID'] = (df['GAP'] > time_gap_seconds).cumsum() #If time gap > 30 mins, assign new guest ID
        guest_sequences = df.groupby('GUEST_ID')['ATTRACTION'].apply(list) #Getting the ride sequence per guest
        express_map = df.groupby('GUEST_ID')['EXPRESS_PASS'].mean().apply(lambda x: x > 0.6) #if >50% rides are fast, guest = express pass
        return guest_sequences, express_map

    def label_guest_sequences_as_express(self, df, guest_sequences, threshold=0.25):
        thresholds = df.groupby(['ATTRACTION', 'DAY_TYPE'])['WAIT_TIME_MAX'].quantile(threshold).to_dict()
        
        guest_labels = {}
        for guest_id, sequence in guest_sequences.items():
            guest_df = df[df['GUEST_ID'] == guest_id]
            if guest_df.empty:
                guest_labels[guest_id] = False
                continue

            low_wait_count = 0
            for _, row in guest_df.iterrows():
                key = (row['ATTRACTION'], row['DAY_TYPE'])
                ride_threshold = thresholds.get(key, np.inf)
                if row['WAIT_TIME_MAX'] < ride_threshold:
                    low_wait_count += 1

            express_ratio = low_wait_count / len(guest_df)
            guest_labels[guest_id] = express_ratio > 0.6  # Consider express if >60% rides were below wait threshold
        return pd.Series(guest_labels)

    
    def label_guest_by_ridetype(self, sequences, family_rides, youth_rides): 
        guest_types = {}
        for guest_id, seq in sequences.items():
            fam_count = sum(1 for ride in seq if ride in family_rides)
            youth_count = sum(1 for ride in seq if ride in youth_rides)
            if fam_count > youth_count:
                guest_types[guest_id] = "Family"
            elif youth_count > fam_count:
                guest_types[guest_id] = "Youth"
            else:
                guest_types[guest_id] = "Mixed"
        return pd.Series(guest_types)
    

    def generate_guest_summary(self, df, reference_ride="Scooby Doo", gap_seconds=1800):
        #Simulate guest sequences
        sequences, _ = self.simulate_guest_sequences(df, time_gap_seconds=gap_seconds)

        #Reassign guest ID on main df
        df = df.sort_values(by='TIMESTAMP')
        df['GAP'] = df['TIMESTAMP'].diff().dt.total_seconds().fillna(0)
        df['GUEST_ID'] = (df['GAP'] > gap_seconds).cumsum()

        #Label express pass guests
        guest_express_labels = self.label_guest_sequences_as_express(df, sequences)

        #Segment ride types
        family_rides, youth_rides = family_rides, youth_rides = self.analyze_ride_correlations(df, reference_ride=reference_ride, day_type=None)
        guest_ride_types = self.label_guest_by_ridetype(sequences, family_rides, youth_rides)

        #Build guest summary df
        guest_summary = pd.DataFrame({
            'GUEST_ID': sequences.index,
            'RIDE_SEQUENCE': sequences.values,
            'EXPRESS_PASS': guest_express_labels,
            'RIDE_TYPE': guest_ride_types
        })

        #Assign day type per guest
        guest_daytypes = df.groupby('GUEST_ID')['DAY_TYPE'].first()
        guest_summary['DAY_TYPE'] = guest_summary['GUEST_ID'].map(guest_daytypes)

        return guest_summary

    def plot_median_wait_times(self, df, day_type=None, title_suffix=""): 
        #plotting waiting times line chart
        if day_type:
            df = df[df['DAY_TYPE'] == day_type]
            title_suffix = f"({day_type.title()} Days)"

        median_wait_times = df.groupby(['DEB_TIME', 'ATTRACTION'])['WAIT_TIME_MAX'].median().unstack()

        plt.figure(figsize=(12, 6))
        colors = plt.cm.get_cmap('tab20', len(median_wait_times.columns))

        for i, attraction in enumerate(median_wait_times.columns):
            plt.plot(median_wait_times.index, median_wait_times[attraction],
                    label=attraction, color=colors(i), marker='o', markersize=4)

        plt.xlabel('Time of Day')
        plt.ylabel('Median Waiting Time (minutes)')
        plt.title(f'Median Waiting Time Throughout the Day {title_suffix} - Tivoli Gardens')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xticks(median_wait_times.index[::4], rotation=45)
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    def plot_ride_correlation_heatmap(self, df, day_type=None, title_suffix=""):
        #plotting correlations between rides heatmap
        if day_type:
            df = df[df['DAY_TYPE'] == day_type]
            title_suffix = f"({day_type.title()} Days)"

        tivoli_pivot = df.pivot(index='TIMESTAMP', columns='ATTRACTION', values='WAIT_TIME_MAX')
        correlation_matrix = tivoli_pivot.corr()

        plt.figure(figsize=(12, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title(f"Correlation Between Ride Waiting Times {title_suffix}")
        plt.tight_layout()
        plt.show()

    def analyze_ride_correlations(self, df, reference_ride="", day_type=""):
        #ride correlations using the date type
        if day_type:
            df = df[df['DAY_TYPE'] == day_type]

        tivoli_pivot = df.pivot(index='TIMESTAMP', columns='ATTRACTION', values='WAIT_TIME_MAX')
        correlation_matrix = tivoli_pivot.corr()

        # Correlation with the reference ride
        reference_correlations = correlation_matrix[reference_ride]

        # Segment rides
        family_rides = [ride for ride, corr in reference_correlations.items() if corr < 0.5]
        youth_rides = [ride for ride, corr in reference_correlations.items() if corr >= 0.5]

        return family_rides, youth_rides

    def analyze_guest_movement(self, df, reference_ride=None, day_type=None):
        if day_type:
            df = df[df['DAY_TYPE'] == day_type]

        # Pivot ride wait times
        tivoli_pivot = df.pivot(index='TIMESTAMP', columns='ATTRACTION', values='WAIT_TIME_MAX')

        if day_type == 'covid':
            # Skip segmentation: analyze all ride movements together
            ride_changes = tivoli_pivot.diff()

            movement_counts = []
            for from_ride in ride_changes.columns:
                for to_ride in ride_changes.columns:
                    if from_ride != to_ride:
                        movement = ((ride_changes[from_ride] < 0) & (ride_changes[to_ride] > 0)).sum()
                        movement_counts.append({"from": from_ride, "to": to_ride, "count": movement})

            all_movements = pd.DataFrame(movement_counts)
            return all_movements, None

        # Non-covid days requires a reference_ride (dividing between family and youth)
        if not reference_ride:
            raise ValueError("You must provide a reference_ride for non-covid day types.")

        # Segmenting using correlation
        correlation_matrix = tivoli_pivot.corr()
        reference_correlations = correlation_matrix[reference_ride]

        family_rides = [ride for ride, corr in reference_correlations.items() if corr < 0.5]
        youth_rides = [ride for ride, corr in reference_correlations.items() if corr >= 0.5]

        # Family
        family_pivot = tivoli_pivot[family_rides]
        family_changes = family_pivot - family_pivot.shift(1)

        family_movement_counts = []
        for from_ride in family_changes.columns:
            for to_ride in family_changes.columns:
                if from_ride != to_ride:
                    movement = ((family_changes[from_ride] < 0) & (family_changes[to_ride] > 0)).sum()
                    family_movement_counts.append({"from": from_ride, "to": to_ride, "count": movement})
        fam_movements = pd.DataFrame(family_movement_counts)

        # Youth
        youth_pivot = tivoli_pivot[youth_rides]
        youth_changes = youth_pivot - youth_pivot.shift(1)

        youth_movement_counts = []
        for from_ride in youth_changes.columns:
            for to_ride in youth_changes.columns:
                if from_ride != to_ride:
                    movement = ((youth_changes[from_ride] < 0) & (youth_changes[to_ride] > 0)).sum()
                    youth_movement_counts.append({"from": from_ride, "to": to_ride, "count": movement})
        youth_movements = pd.DataFrame(youth_movement_counts)

        return fam_movements, youth_movements

    def markov_chain_analysis(self, df, title):
        graph = nx.DiGraph()
        for _, row in df.iterrows():
            graph.add_edge(row['from'], row['to'], weight=row['count']) #transition edeges

        rides = list(graph.nodes())
        transition_matrix = np.zeros((len(rides), len(rides)))

        for i, from_ride in enumerate(rides):
            total_outflow = sum(graph.get_edge_data(from_ride, to_ride)['weight'] for to_ride in graph.successors(from_ride))
            if total_outflow > 0:
                for j, to_ride in enumerate(rides):
                    if graph.has_edge(from_ride, to_ride):
                        transition_matrix[i, j] = graph.get_edge_data(from_ride, to_ride)['weight'] / total_outflow

        #Compute steady state distribution:
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        steady_state = np.abs(eigenvectors[:, np.argmax(eigenvalues)])
        steady_state /= steady_state.sum()

        steady_state_distribution = pd.Series(steady_state, index=rides)

        print(f"\n{title} - Steady-State Distribution:")
        print(steady_state_distribution)

    def guest_avg_wait_top_rides(self, df, target_rides):
        df = df[df['ATTRACTION'].isin(target_rides)] #Focus on top rides!
        df = df.sort_values(by='TIMESTAMP')
        df['GAP'] = df['TIMESTAMP'].diff().dt.total_seconds().fillna(0)
        df['GUEST_ID'] = (df['GAP'] > 1800).cumsum() #Segmenting guests by time gaps
        avg_wait = df.groupby(['GUEST_ID', 'DAY_TYPE'])['WAIT_TIME_MAX'].mean().reset_index()
        avg_wait = avg_wait.rename(columns={'WAIT_TIME_MAX': 'AVG_WAIT_TIME_TOP_3'})
        return avg_wait

    def calculate_outflow_for_top_rides(self, df, top_rides, reference_ride="Scooby Doo"):
        outflow_summary = []

        for day_type in df['DAY_TYPE'].unique():
            if day_type == "covid":
                #Depending on DAY_TYPE, since covid doesnt need a reference_ride for division of youth v family with children
                movement_df, _ = self.analyze_guest_movement(df, day_type=day_type)
            else:
                movement_df, _ = self.analyze_guest_movement(df, reference_ride=reference_ride, day_type=day_type)

            outflow = movement_df.groupby('from')['count'].sum()
            filtered_outflow = outflow[outflow.index.isin(top_rides)]
            avg_outflow = filtered_outflow.mean()

            outflow_summary.append({
                'DAY_TYPE': day_type,
                'AVG_OUTFLOW_TOP_3': avg_outflow
            })

        return pd.DataFrame(outflow_summary)

    def plot_combined_wait_and_outflow(self, wait_df, outflow_df):
        combined = wait_df.groupby('DAY_TYPE')['AVG_WAIT_TIME_TOP_3'].mean().reset_index()
        combined = combined.merge(outflow_df, on='DAY_TYPE')

        fig, ax1 = plt.subplots(figsize=(10, 6))

        #Left Y-Axis = Average Wait Time (line)
        ax1.set_ylabel('Avg Wait Time (mins)', color='blue')
        sns.lineplot(data=combined, x='DAY_TYPE', y='AVG_WAIT_TIME_TOP_3', marker='o', ax=ax1, color='blue', label='Avg Wait Time')
        ax1.tick_params(axis='y', labelcolor='blue')

        #Right y-axis = Guest Outflow (bar)
        ax2 = ax1.twinx()
        ax2.set_ylabel('Avg Guest Outflow (top 3 rides)', color='red')
        sns.barplot(data=combined, x='DAY_TYPE', y='AVG_OUTFLOW_TOP_3', ax=ax2, alpha=0.3, color='red')
        ax2.tick_params(axis='y', labelcolor='red')

        plt.title('Guest Willingness to Wait vs Ride Popularity (Top 3 Rides)')
        plt.tight_layout()
        plt.show()

    def plot_avg_wait_boxplot(self, df):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='DAY_TYPE', y='AVG_WAIT_TIME_TOP_3', palette='Set2')
        sns.stripplot(data=df, x='DAY_TYPE', y='AVG_WAIT_TIME_TOP_3', color='black', size=3, alpha=0.3, jitter=0.2)

        plt.title("Guest Wait Time Tolerance for Top 3 Rides (Boxplot)")
        plt.xlabel("Day Type")
        plt.ylabel("Avg Wait Time per Guest (Top 3 Rides) [mins]")
        plt.grid(True, axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def sequence_similarity(self, seq1, seq2):
        return SequenceMatcher(None, seq1, seq2).ratio() #Similarity score between sequences

    def average_similarity(self, sequences):
        combos = list(combinations(sequences, 2)) #Pairwise combinations
        if not combos:
            return 0
        return np.mean([self.sequence_similarity(a, b) for a, b in combos]) #Average similarities across all pairs

    def compute_sequence_entropy(self, seq):
        counts = pd.Series(seq).value_counts(normalize=True) #Frequency Distribution
        return entropy(counts)


    def run_guestjourneyanalysis(self):
        # Differing the covid and non-covid dates
        busy_days, quiet_days, covid_dates = self.determine_day_types(self.attendance_df, self.covid)
        self.tivoli_g = self.tag_day_type(self.tivoli_g, covid_dates, busy_days, quiet_days)
        self.tivoli_g = self.label_express_pass_by_daytype(self.tivoli_g)

        
        #Plotting line graphs and correlation heatmaps for each to view 
        self.plot_median_wait_times(self.tivoli_g, day_type="normal", title_suffix="Normal Days")
        self.plot_ride_correlation_heatmap(self.tivoli_g, day_type="normal", title_suffix="Normal Days")
        self.plot_median_wait_times(self.tivoli_g, day_type="busy", title_suffix="Busy Days")
        self.plot_ride_correlation_heatmap(self.tivoli_g, day_type="busy", title_suffix="Busy Days")
        self.plot_median_wait_times(self.tivoli_g, day_type="quiet", title_suffix="Quiet Days")
        self.plot_ride_correlation_heatmap(self.tivoli_g, day_type="quiet", title_suffix="Quiet Days")
        self.plot_median_wait_times(self.tivoli_g, day_type="covid", title_suffix="COVID Days")

        # Segmenting of Guests -- Family w/ Kids vs. Youths
        family_rides, youth_rides = self.analyze_ride_correlations(self.tivoli_g, day_type='normal', reference_ride="Scooby Doo")
        family_rides_busy, youth_rides_busy = self.analyze_ride_correlations(self.tivoli_g, day_type='busy', reference_ride="Scooby Doo")
        family_rides_quiet, youth_rides_quiet = self.analyze_ride_correlations(self.tivoli_g, day_type='quiet', reference_ride="Scooby Doo")

        #Potential movements from and to rides
        fam_movements, youth_movements = self.analyze_guest_movement(self.tivoli_g, reference_ride='Scooby Doo', day_type="normal")
        fam_movements_busy, youth_movements_busy = self.analyze_guest_movement(self.tivoli_g, reference_ride='Scooby Doo', day_type="busy")
        fam_movements_quiet, youth_movements_quiet = self.analyze_guest_movement(self.tivoli_g, reference_ride='Scooby Doo', day_type="quiet")
        movement_covid, _ = self.analyze_guest_movement(self.tivoli_g, day_type="covid")

        # Normal Days
        family_outflow = fam_movements.groupby('from')['count'].sum().sort_values(ascending=False)
        print("Family Rides - Highest Outgoing Traffic:")
        print(family_outflow)
        print("\nFamily Rides - Most Frequent Transitions:")
        print(fam_movements.sort_values(by='count', ascending=False))
        self.markov_chain_analysis(fam_movements, "Family Rides")

        youth_outflow = youth_movements.groupby('from')['count'].sum().sort_values(ascending=False)
        print("\nYouth Rides - Highest Outgoing Traffic:")
        print(youth_outflow)
        print("\nYouth Rides - Most Frequent Transitions:")
        print(youth_movements.sort_values(by='count', ascending=False))
        self.markov_chain_analysis(youth_movements, "Youth Rides")

        # Busy Days
        family_outflow_busy = fam_movements_busy.groupby('from')['count'].sum().sort_values(ascending=False)
        print("Family Rides during Busy Days - Highest Outgoing Traffic:")
        print(family_outflow_busy)
        print("\nFamily Rides during Busy Days - Most Frequent Transitions:")
        print(fam_movements_busy.sort_values(by='count', ascending=False))
        self.markov_chain_analysis(fam_movements_busy, "Family Rides during Busy Days")

        youth_outflow_busy = youth_movements_busy.groupby('from')['count'].sum().sort_values(ascending=False)
        print("\nYouth Rides during Busy Days - Highest Outgoing Traffic:")
        print(youth_outflow_busy)
        print("\nYouth Rides during Busy Days - Most Frequent Transitions:")
        print(youth_movements_busy.sort_values(by='count', ascending=False))
        self.markov_chain_analysis(youth_movements_busy, "Youth Rides during Busy Days")

        # Quiet Days
        family_outflow_quiet = fam_movements_quiet.groupby('from')['count'].sum().sort_values(ascending=False)
        print("Family Rides during Quiet Days - Highest Outgoing Traffic:")
        print(family_outflow_quiet)
        print("\nFamily Rides during Quiet Days - Most Frequent Transitions:")
        print(fam_movements_quiet.sort_values(by='count', ascending=False))
        self.markov_chain_analysis(fam_movements_quiet, "Family Rides during Quiet Days")

        youth_outflow_quiet = youth_movements_quiet.groupby('from')['count'].sum().sort_values(ascending=False)
        print("\nYouth Rides during Quiet Days - Highest Outgoing Traffic:")
        print(youth_outflow_quiet)
        print("\nYouth Rides during Quiet Days - Most Frequent Transitions:")
        print(youth_movements_quiet.sort_values(by='count', ascending=False))
        self.markov_chain_analysis(youth_movements_quiet, "Youth Rides during Quiet Days")

        # Covid Days
        covid_outflow = movement_covid.groupby('from')['count'].sum().sort_values(ascending=False)
        print("Rides during Covid Days - Highest Outgoing Traffic:")
        print(covid_outflow)
        print("\nRides during Covid Days - Most Frequent Transitions:")
        print(movement_covid.sort_values(by='count', ascending=False))
        self.markov_chain_analysis(movement_covid, "Rides during Covid Days")

        # Identify top 3 rides
        top_rides = self.tivoli_g['ATTRACTION'].value_counts().nlargest(3).index.tolist()
        avg_wait_top3 = self.guest_avg_wait_top_rides(self.tivoli_g, top_rides)
        outflow_top3 = self.calculate_outflow_for_top_rides(self.tivoli_g, top_rides)
        self.plot_combined_wait_and_outflow(avg_wait_top3, outflow_top3)

        self.plot_avg_wait_boxplot(avg_wait_top3)

        guest_summary = self.generate_guest_summary(self.tivoli_g)

        #Sequence entropy to figure out how diverse a guest's ride path is depending on express v non express pass
        #Sequence length to figure out how many rides they tend to go on depending on express v non express pass
        guest_summary['SEQ_ENTROPY'] = guest_summary['RIDE_SEQUENCE'].apply(self.compute_sequence_entropy)
        guest_summary['SEQ_LEN'] = guest_summary['RIDE_SEQUENCE'].apply(len)

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=guest_summary, x='EXPRESS_PASS', y='SEQ_LEN', palette='muted')
        plt.title("Guest Journey Length (Express vs Non-Express)")
        plt.xlabel("Express Pass Used")
        plt.ylabel("Number of Rides in Journey")
        plt.xticks([0, 1], ["No", "Yes"])
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        sns.boxplot(data=guest_summary, x='EXPRESS_PASS', y='SEQ_ENTROPY', palette='Spectral')
        plt.title("Ride Sequence Diversity (Entropy) - Express vs Non-Express")
        plt.xlabel("Express Pass Used")
        plt.ylabel("Entropy of Ride Sequence")
        plt.xticks([0, 1], ["No", "Yes"])
        plt.tight_layout()
        plt.show()

        #quantifying the average similarities among express pass and non express pass users 
        express_seqs = guest_summary[guest_summary['EXPRESS_PASS'] == True]['RIDE_SEQUENCE'].tolist()
        nonexpress_seqs = guest_summary[guest_summary['EXPRESS_PASS'] == False]['RIDE_SEQUENCE'].tolist()
        similarity_express = self.average_similarity(express_seqs)
        similarity_nonexpress = self.average_similarity(nonexpress_seqs)
        print(f"Avg similarity within Express journeys: {similarity_express:.3f}")
        print(f"Avg similarity within Non-Express journeys: {similarity_nonexpress:.3f}")

if __name__ == "__main__":
    analysis = GuestJourneyAnalysis()
    analysis.run_guestjourneyanalysis()
