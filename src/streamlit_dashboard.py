import numpy as np
import streamlit as st
import os 
import pandas as pd
import altair as alt
import pickle
# import matplotlib.pyplot as plt
from pathlib import Path
# importing the other modules:
from subgroup_a.datapreparation_A import *
from subgroup_a.analysis.q1_guest_satisfaction_analysis import *
from subgroup_a.analysis.q3_guest_journey_analysis import GuestJourneyAnalysis
from subgroup_a.analysis.q4_promo_events_analysis import *
from subgroup_a.analysis.q5_external_factors_analysis import *
from subgroup_a.modeling.q2_guest_segmentation_model import *
import warnings
warnings.filterwarnings('ignore')


######################
# PAGE TITLE AND LAYOUT
st.set_page_config(
    page_title="Theme Park Guest Experience Dashboard",
    page_icon="🎢",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")
######################

# To check if the dashboard is running in streamlit
st.success("✅ Dashboard is running!")
st.title("🎡 Welcome to the Theme Park Guest Experience Dashboard!")

#######################
# SIDEBAR
st.sidebar.title("🔍 Filter Controls")
day_type_labels = {"normal": "Normal", "busy": "Busy", "quiet": "Quiet", "covid": "Covid-19"}
day_type_ordered = ["normal", "busy", "quiet", "covid"]
selected_day_type = st.sidebar.selectbox("Select Day Type", day_type_ordered, format_func=lambda x: day_type_labels[x])
segment_filter = st.sidebar.selectbox("Select Group of Guests", ["All", "Youth", "Family with Children"])
express_filter = st.sidebar.selectbox("Select Express Pass Usage", ["All", "Yes", "No"])
#######################

#######################
# PLOTS
# SUBGROUP A ##########
# QUESTION 1 ##########
#load data
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
csv_path = project_root / "data" / "survey.csv"

# plot correlation analysis 
def correlation_analysis(self):
        """Perform correlation analysis of all factors"""
        print("\n7. Correlation Analysis:")
        corr_df = self.df.copy()
        
        # Convert categorical features to numerical
        wait_time_order = ['Less than 15 minutes', '15 to 30 minutes', 
                          '31 to 45 minutes', '46 to 60 minutes',
                          '61 to 90 minutes', 'More than 90 minutes']
        corr_df['Wait Time Score'] = corr_df['How long did you wait in line for rides on average during your visit?']\
            .map({k:v for v,k in enumerate(wait_time_order)})
        
        corr_df['Express Pass'] = corr_df['Did you purchase the Express Pass?'].map({'Yes': 1, 'No': 0})
        
        # Convert other categorical features
        binary_map = {'Yes': 1, 'No': 0}
        corr_df['Food Variety'] = corr_df[' Did you find a good variety of food options?  '].map(binary_map)
        corr_df['Website Used'] = corr_df['Did you visit the USS website while planning your trip?'].map(binary_map)
        
        # Select relevant numerical columns
        corr_columns = {
            'On a scale of 1-5, how would you rate your overall experience at USS?': 'Overall Experience',
            'Were the park staff at USS friendly and helpful? Rate on a scale from 1-5.': 'Staff Friendliness',
            ' How would you rate the food quality and service?  ': 'Food Quality',
            'How easy was it to find relevant information about USS online (ticket pricing, attractions, events, etc.)?': 'Info Accessibility',
            ' Were the shows and performances engaging and enjoyable?  ': 'Show Quality',
            'Wait Time Score': 'Wait Time',
            'Express Pass': 'Express Pass',
            'Food Variety': 'Food Variety',
            'Website Used': 'Website Used'
        }

        # Rename the columns
        corr_df.rename(columns=corr_columns, inplace=True)
        
        # Convert brand image with one-hot encoding
        brand_dummies = pd.get_dummies(corr_df['How would you describe USS\' brand image before visiting?'], 
                                      prefix='Brand')
        corr_df = pd.concat([corr_df, brand_dummies], axis=1)
        
        # Select and rename numerical columns
        corr_df = corr_df[list(corr_columns.values()) + list(brand_dummies.columns)]
        
        # Calculate correlations
        corr_matrix = corr_df.corr()
        
        selected_columns = list(corr_columns.values())  
        corr_subset = corr_matrix.loc[selected_columns, selected_columns]

# QUESTION 2 ##########

# QUESTION 3 ##########
# LOADING THE DATA
script_dir = Path(__file__).resolve().parent
csv_path = script_dir.parent / 'data' / 'processed data' / 'tivoli_g.csv'
tivoli_g = pd.read_csv(csv_path)
tivoli_g['TIMESTAMP'] = pd.to_datetime(tivoli_g['TIMESTAMP'])
tivoli_g['WORK_DATE'] = pd.to_datetime(tivoli_g['WORK_DATE'])

analysis = GuestJourneyAnalysis(tivoli_g)
guest_summary = analysis.generate_guest_summary(tivoli_g)
guest_summary['SEQ_LEN'] = guest_summary['RIDE_SEQUENCE'].apply(len)
guest_summary['SEQ_ENTROPY'] = guest_summary['RIDE_SEQUENCE'].apply(analysis.compute_sequence_entropy)
guest_summary['EXPRESS_PASS_LABEL'] = guest_summary['EXPRESS_PASS'].map({True: "Yes", False: "No"})

# FUNCTIONS FOR THE AQ3 MODULE
def get_filtered_transitions():
    if selected_day_type != "covid":
        filtered_df = tivoli_g[tivoli_g['DAY_TYPE'] == selected_day_type]
        fam_movements, youth_movements = analysis.analyze_guest_movement(filtered_df, reference_ride="Scooby Doo", day_type=selected_day_type)
        family_rides, youth_rides = analysis.analyze_ride_correlations(filtered_df, reference_ride="Scooby Doo", day_type=selected_day_type)

        if segment_filter == "Youth":
            transition_df = youth_movements
        elif segment_filter == "Family with Children":
            transition_df = fam_movements
        else:
            transition_df = pd.concat([fam_movements, youth_movements])
    else:
        transition_df, _ = analysis.analyze_guest_movement(tivoli_g, day_type=selected_day_type)
        family_rides, youth_rides = [], []

    return transition_df, youth_rides, family_rides

def render_guest_journey_analysis():
    st.subheader(f"👥 Guest Movement Analysis for {day_type_labels[selected_day_type]} Days")
    transition_df, youth_rides, family_rides = get_filtered_transitions()

    # Top layout
    col1, col2 = st.columns(2)

    with col1:
        # Plot 1: Outflow Comparison
        st.markdown(f"<h2 style='font-size: 20px;'>Guest Outflow by Segment on {day_type_labels[selected_day_type]} Days</h2>", unsafe_allow_html=True) #making the plot title wrap around
        outflow_data = transition_df.groupby('from')['count'].sum().reset_index()
        outflow_data['Segment'] = outflow_data['from'].apply(
            lambda x: 'Youth' if x in youth_rides else ('Family' if x in family_rides else 'Other')
        )
        summary = outflow_data.groupby('Segment')['count'].sum().reset_index()
        chart = alt.Chart(summary).mark_bar().encode(
            x=alt.X('Segment:N', axis=None),
            y=alt.Y('count:Q', title='Total Outflow'),
            color='Segment:N'
        ).properties(
            width=500,
            height=300
        )
        st.altair_chart(chart, use_container_width=True)

    with col2:
        # Plot 3: Top 5 Outgoing
        st.markdown(f"<h2 style='font-size: 20px;'>Top 5 Rides by Outgoing Transitions on {day_type_labels[selected_day_type]} Days</h2>", unsafe_allow_html=True) #making the plot title wrap around
        top_outgoing = transition_df.groupby('from')['count'].sum().reset_index().nlargest(5, 'count')
        top_out_chart = alt.Chart(top_outgoing).mark_bar().encode(
            x=alt.X('count:Q', title='Total Outflow'),
            y=alt.Y('from:N', sort='-x', title='Ride'),
            color=alt.Color('count:Q', scale=alt.Scale(scheme='tealblues'))
        ).properties(
            width=500,
            height=300
        )
        st.altair_chart(top_out_chart, use_container_width=True)

    # Middle layout
    col3, col4 = st.columns(2)

    with col3:
        # Plot 4: Top 10 Transitions
        st.markdown(f"<h2 style='font-size: 20px;'>Top 10 Most Frequent Ride Transitions on {day_type_labels[selected_day_type]} Days</h2>", unsafe_allow_html=True) #making the plot title wrap around
        top_pairs = transition_df.sort_values('count', ascending=False).head(10)
        top_pairs_chart = alt.Chart(top_pairs).mark_bar().encode(
            x=alt.X('count:Q', title='Transition Count'),
            y=alt.Y('from:N', sort='-x', title='From Ride'),
            color='to:N',
            tooltip=['from', 'to', 'count']
        ).properties(
            width=500,
            height=400
        )
        st.altair_chart(top_pairs_chart, use_container_width=True)

    with col4:
        # Plot 2: Transition Heatmap
        st.markdown(f"<h2 style='font-size: 20px;'>Transition Matrix ({segment_filter}) on {day_type_labels[selected_day_type]} Days</h2>", unsafe_allow_html=True) #making the plot title wrap around
        heatmap = alt.Chart(transition_df).mark_rect().encode(
            x=alt.X('from:N', sort=None, title='From Ride'),
            y=alt.Y('to:N', sort=None, title='To Ride'),
            color=alt.Color('count:Q', scale=alt.Scale(scheme='blues'), title='Transition Count'),
            tooltip=['from', 'to', 'count']
        ).properties(
            width=500,
            height=400
        )
        st.altair_chart(heatmap, use_container_width=True)
        st.markdown("<span style='font-size:14px;'>ℹ️ <b>Note</b>: This matrix shows how guests transition from one ride to another.</span>", unsafe_allow_html=True)

    # Bottom layout
    col5, col6, col7 = st.columns(3)

    with col5:
        st.markdown(f"<h2 style='font-size: 20px;'>Guest Wait Time Tolerance for Top 3 Rides on {day_type_labels[selected_day_type]} Days</h2>", unsafe_allow_html=True)
        top_rides = tivoli_g['ATTRACTION'].value_counts().nlargest(3).index.tolist()
        wait_df = analysis.guest_avg_wait_top_rides(tivoli_g, top_rides)
        filtered_wait_df = wait_df[wait_df['DAY_TYPE'] == selected_day_type].copy()
        filtered_wait_df['DAY_TYPE'] = filtered_wait_df['DAY_TYPE'].map(day_type_labels)

        box = alt.Chart(filtered_wait_df).mark_boxplot(extent='min-max', size=50).encode(
            x=alt.X('DAY_TYPE:N', title='Day Type', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('AVG_WAIT_TIME_TOP_3:Q', title='Average Wait Time (mins)'),
            color=alt.Color('DAY_TYPE:N', legend=None)
        ).properties(
            width=350,
            height=300
        )
        points = alt.Chart(filtered_wait_df).mark_circle(size=30, opacity=0.5, stroke='black', fill=None).encode(
            x='DAY_TYPE:N',
            y='AVG_WAIT_TIME_TOP_3:Q',
            color=alt.Color('DAY_TYPE:N', legend=None)
        )
        st.altair_chart(box + points, use_container_width=True)

    with col6:
        st.markdown("<h2 style='font-size: 20px;'>Guest Journey Length of Express and Non-Express Pass Users</h2>", unsafe_allow_html=True) #making the plot title wrap around
        filtered_summary = guest_summary if express_filter == "All" else guest_summary[guest_summary['EXPRESS_PASS_LABEL'] == express_filter]
        chart = alt.Chart(filtered_summary).mark_boxplot(size=50).encode(
            x=alt.X('EXPRESS_PASS_LABEL:N', title='Express Pass', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('SEQ_LEN:Q', title='Number of Rides'),
            color=alt.Color('EXPRESS_PASS_LABEL:N', legend=None)
        ).properties(
            # title = "Guest Journey Length of Express and/or Non-Express Pass Users",
            width=350,
            height=300
        )
        st.altair_chart(chart, use_container_width=True)

    with col7:
        st.markdown("<h2 style='font-size: 20px;'>Ride Sequence Entropy of Express vs Non-Express Pass Users</h2>", unsafe_allow_html=True) #making the plot title wrap around
        entropy_chart = alt.Chart(filtered_summary).mark_boxplot(size=50).encode(
            x=alt.X('EXPRESS_PASS_LABEL:N', title='Express Pass', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('SEQ_ENTROPY:Q', title='Ride Sequence Entropy'),
            color=alt.Color('EXPRESS_PASS_LABEL:N', legend=None)
        ).properties(
            # title = "Ride Sequence Diversity (Entropy) of Express vs Non-Express Pass Users",
            width=350,
            height=300
        )
        st.altair_chart(entropy_chart, use_container_width=True)
        st.markdown("<span style='font-size:14px;'>ℹ️ <b>Note</b>: Higher entropy indicates <b>lower diversity</b> in ride sequences.</span>", unsafe_allow_html=True)

# RUNNING THE AQ3 MODULE
render_guest_journey_analysis()

# QUESTION 4 ###########
# Load data
df_combined, df_labeled, scaled, pca = cleaning_q2()

# Run segmentation model and visualize data
segmentation_analysis = guest_segmentation_model(df_combined, df_labeled, scaled, pca)
segmentation_analysis.visualize_marketing_analysis()


# QUESTION 5 ###########
q5_df_reviews = q5_clean_data()

# For rain sentiment analysis
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity

q5_df_reviews["sentiment"] = q5_df_reviews["review_text"].apply(get_sentiment)


rain_reviews = q5_df_reviews[q5_df_reviews['mentions_rain'] == True]
no_rain_reviews = q5_df_reviews[q5_df_reviews['mentions_rain'] == False]

# Group by the 'segment' and calculate the average sentiment for reviews mentioning rain
rain_segment_sentiment = rain_reviews.groupby('segment')['sentiment'].mean().reset_index()

# Group by the 'segment' and calculate the average rating for reviews not mentioning rain
no_rain_segment_sentiment = no_rain_reviews.groupby('segment')['sentiment'].mean().reset_index()

# Merging the sentiment results for comparison
sentiment_comparison = pd.merge(rain_segment_sentiment, no_rain_segment_sentiment, on='segment', suffixes=('_rain', '_no_rain'))

sentiment_diff = sentiment_comparison.groupby('segment').apply(
    lambda x: x['sentiment_rain'].mean() - x['sentiment_no_rain'].mean()).reset_index(name='sentiment_diff')

# Rename columns for clarity
sentiment_diff.columns = ['segment', 'sentiment_diff']

# For seasonal guest volume analysis
guest_volume = q5_df_reviews.groupby(['segment', 'month']).size().reset_index(name='guest_count')
def guestvolseason(guest_volume):
    season_mapping = {
    'February': 'Feb-Apr', 'March': 'Feb-Apr', 'April': 'Feb-Apr',
    'May': 'Summer Holidays', 'June': 'Summer Holidays', 'July': 'Summer Holidays',
    'August': 'Aug-Oct', 'September': 'Aug-Oct', 'October': 'Aug-Oct',
    'November': 'Winter Holidays', 'December': 'Winter Holidays', 'January': 'Winter Holidays'
    }

    # Map months to seasons
    guest_volume['season'] = guest_volume['month'].map(season_mapping)

    # Create a duplicated dataframe where Feb-Apr appears again on the right
    guest_volume_dup = guest_volume[guest_volume['season'] == 'Feb-Apr'].copy()
    guest_volume_dup['season'] = 'Feb-Apr (2)'

    # Combine original with duplicated Feb-Apr
    guest_volume_extended = pd.concat([guest_volume, guest_volume_dup])

    # Maintain correct order
    season_order = ['Feb-Apr', 'Summer Holidays', 'Aug-Oct', 'Winter Holidays', 'Feb-Apr (2)']
    guest_volume_extended['season'] = pd.Categorical(guest_volume_extended['season'], categories=season_order, ordered=True)

    # Aggregate guest count by segment and season
    seasonal_guest_volume = guest_volume_extended.groupby(['segment', 'season'], observed=True)['guest_count'].sum().reset_index()
    return seasonal_guest_volume
seasonal_guest_volume = guestvolseason(guest_volume)



#Plot Rain sentiment difference and  Seasonal Volume
def plot_sentiment_and_seasonal_volume(sentiment_diff, seasonal_guest_volume):
    # Segment labels
    segment_labels = {
        0: "Social-Driven Youths",
        1: "Value-Conscious Families",
        2: "Budget-Conscious Youths",
        3: "Premium Spenders"
        }
        # --- Prepare sentiment_diff data
    sentiment_df = sentiment_diff.copy()
    sentiment_df["segment_label"] = sentiment_df["segment"].map(segment_labels)
    sentiment_df["segment_order"] = sentiment_df["segment"]

    # --- Prepare seasonal_guest_volume data
    seasonal_df = seasonal_guest_volume.copy()
    seasonal_df["segment_label"] = seasonal_df["segment"].map(segment_labels)

    # Define a custom order for seasons if needed
    season_order = ["Dec-Feb", "Mar-May", "Jun-Aug", "Sep-Nov"]

    # Create Streamlit columns
    col1, col2 = st.columns(2)

    with col1:
        sentiment_chart = alt.Chart(sentiment_df).mark_bar().encode(
            x=alt.X('segment_label:N', title='Segment',
                    sort=alt.SortField('segment_order', order='ascending')),
            y=alt.Y('sentiment_diff:Q', title='Sentiment Difference (Rain - No Rain)'),
            color=alt.condition(
                alt.datum.sentiment_diff > 0,
                alt.value("steelblue"),
                alt.value("indianred")
            ),
            tooltip=['segment_label:N', 'sentiment_diff:Q']
        ).properties(
            title='Sentiment Difference (Rain vs No Rain)',
            width=500,
            height=400
        ).configure_axisX(
            labelAngle=0,
            labelFontSize=9
        )

        st.altair_chart(sentiment_chart, use_container_width=False)

    with col2:
        line_chart = alt.Chart(seasonal_df).mark_line(point=True).encode(
            x=alt.X('season:N', title='Season', sort=season_order),
            y=alt.Y('guest_count:Q', title='Guest Volume'),
            color=alt.Color('segment_label:N', title='Segment',
                            scale=alt.Scale(scheme='category10')),
                tooltip=['season', 'segment_label', 'guest_count']
        ).properties(
            title='Guest Volume by Segment and Season',
            width=500,
            height=400
        ).configure_axisX(
            labelAngle=0,
            labelFontSize=9
        )

        st.altair_chart(line_chart, use_container_width=False)
plot_sentiment_and_seasonal_volume(sentiment_diff, seasonal_guest_volume)


# SUBGROUP B ##########
import pickle as pk
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# QUESTION 1

# QUESTION 2
st.subheader("🎢 Optimized Layout Simulation Results")

with open("../models/q2_optimization_layout.pkl", "rb") as f:
    comparison_results = pickle.load(f)

# Convert the comparison results to a DataFrame
data = []
for attraction in comparison_results["avg_wait_times_1_multi"]:
    data.append({
        "Attraction": attraction,
        "Layout": "Current (Two Entrances)",
        "Avg Wait Time (mins)": comparison_results["avg_wait_times_1_multi"][attraction]
    })
    data.append({
        "Attraction": attraction,
        "Layout": "Modified (Swapped & One Entrance)",
        "Avg Wait Time (mins)": comparison_results["avg_wait_times_2_multi"][attraction]
    })

wait_df = pd.DataFrame(data)

wait_df['Layout'] = wait_df['Layout'].map({
    'Current (Two Entrances)': 'Current',
    'Modified (Swapped & One Entrance)': 'Modified'
})

label_map = {
    "Revenge of the Mummy": "Mummy",
    "Battlestar Galactica: CYLON": "CYLON",
    "Transformers: The Ride": "Transformers",
    "Puss In Boots' Giant Journey": "Puss In Boots",
    "Sesame Street Spaghetti Space Chase": "Sesame Street"
}
wait_df['Attraction'] = wait_df['Attraction'].map(label_map)

st.markdown(
    "> **Note:** We swap the locations of *Transformers* and *CYLON*, placing congested rides last in the guest journey. "
    "We will also close the right entrance to guide guests through a left-to-right flow, reducing congestion."
)

# Horizontal Bar Chart
st.subheader("Comparison of Average Wait Times by Attraction")
chart = alt.Chart(wait_df).mark_bar().encode(
    x=alt.X('Attraction:N', title='Attraction', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('Avg Wait Time (mins):Q', title='Average Wait Time (mins)'),
    color=alt.Color('Layout:N', title='Layout'),
    tooltip=['Attraction', 'Layout', 'Avg Wait Time (mins)'],
    xOffset='Layout:N'  # Enables grouped side-by-side bars
).properties(
    width=700,
    height=400
)

st.altair_chart(chart, use_container_width=True)

# === Average wait times from the comparison results ===
# Values from simulation results
current_avg = comparison_results['avg_wait_per_guest_1']
modified_avg = comparison_results['avg_wait_per_guest_2']
diff = current_avg - modified_avg
pct_drop = (diff / current_avg) * 100
arrow = "⬇️" if diff > 0 else "⬆️"
color = "green" if diff > 0 else "red"

# Use Streamlit columns for layout
st.markdown("### 📉 Overall Wait Time Improvement")

col1, col2, col3 = st.columns(3)

def kpi_card(title, value, delta=None, arrow=None, color="white"):
    return f"""
    <div style="
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.3);
    ">
        <div style='font-size: 14px; color: #cccccc;'>{title}</div>
        <div style='font-size: 32px; font-weight: bold;'>{value}</div>
        {"<div style='color:" + color + "; font-size: 16px;'>" + arrow + " " + delta + "</div>" if delta else ""}
    </div>
    """

with col1:
    st.markdown(kpi_card("Current Avg Wait Time", f"{current_avg:.2f} mins"), unsafe_allow_html=True)

with col2:
    st.markdown(kpi_card("Modified Avg Wait Time", f"{modified_avg:.2f} mins"), unsafe_allow_html=True)

with col3:
    st.markdown(
        kpi_card(
            "Time Reduction",
            f"{diff:.2f} mins",
            delta=f"{abs(pct_drop):.1f}%",
            arrow=arrow,
            color="green" if diff > 0 else "red"
        ),
        unsafe_allow_html=True
    )

# QUESTION 3

# QUESTION 4

# QUESTION 5
# QUESTION 5 ###########
st.subheader("IoT Data Integration for Experience Optimisation")
df_iot_path = "../data/processed data/iot_data.pkl" 
df_iot = pd.read_pickle(df_iot_path)

def train_demand_model_2(df, target='Average_Queue_Time'):
    features = [
        'Visitor_ID', 'Loyalty_Member', 'Age', 'Gender',
        'Theme_Zone_Visited', 'Attraction', 'Check_In', 'Queue_Time', 'Check_Out',
        'Restaurant_Spending', 'Merchandise_Spending', 'Total_Spending',
        'Day_of_Week', 'Is_Weekend', 'Is_Popular_Attraction'
    ]
    df = df[features + [target]]

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(
        random_state=42,
        n_estimators=500,
        max_depth=4,
        learning_rate=0.1,
        verbosity=0
    )

    cross_val_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred)
    }

    st.success("✅ IoT Demand Model Trained Successfully")
    st.metric("RMSE", f"{metrics['RMSE']:.2f}", help="Root Mean Squared Error — lower is better. Shows average error in predicted queue time.")
    st.metric("MAE", f"{metrics['MAE']:.2f}", help="Mean Absolute Error — average absolute difference between actual and predicted queue time.")

# Optional: Explain metrics in simple terms
    with st.expander("ℹ️ What do these metrics mean?"):
        st.markdown("""
        - **RMSE (Root Mean Squared Error)**: Measures the typical size of prediction errors. Bigger mistakes are penalized more.
        - **MAE (Mean Absolute Error)**: Straight average of how far off predictions are — in queue time minutes.
        """)

    df_test = X_test.copy()
    df_test['Predicted_' + target] = y_pred

    # Altair Plot 1: Check-In vs Predicted Queue
    st.subheader("📊 Check-In Time vs Predicted Queue Time (in mins)")
    checkin_plot = alt.Chart(df_test).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X('Check_In:Q', axis=alt.Axis(title='Check-In Time')),
        y=alt.Y(f'Predicted_{target}:Q', axis=alt.Axis(title='Predicted Queue Time (in mins)')),
        tooltip=['Check_In', f'Predicted_{target}']
    ).properties(
        width=600,
        height=300
    ).interactive()
    st.altair_chart(checkin_plot, use_container_width=True)

    # Altair Plot 2: Check-Out vs Predicted Queue
    st.subheader("📊 Check-Out Time vs Predicted Queue Time (in mins)")
    checkout_plot = alt.Chart(df_test).mark_circle(size=60, color='green', opacity=0.6).encode(
        x=alt.X('Check_Out:Q', axis=alt.Axis(title='Check-Out Time')),
        y=alt.Y(f'Predicted_{target}:Q', axis=alt.Axis(title='Predicted Queue Time (in mins)')),
        tooltip=['Check_Out', f'Predicted_{target}']
    ).properties(
        width=600,
        height=300
    ).interactive()
    st.altair_chart(checkout_plot, use_container_width=True)


    return model, metrics

# Training model with just IoT data
model_3, metrics_3 = train_demand_model_2(df_iot)


#######################


#######################
                       
