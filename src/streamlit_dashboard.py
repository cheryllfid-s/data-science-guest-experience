import numpy as np
import streamlit as st
import os 
import pandas as pd
import altair as alt
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

# Sentiment Diff Train


#Plot Seasonal Volume



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

# QUESTION 3

# QUESTION 4

# QUESTION 5



#######################


#######################
                       
