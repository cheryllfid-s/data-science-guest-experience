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
from subgroup_a.analysis.q5_external_factors_analysis import q5_analyse
from subgroup_a.modeling.q2_guest_segmentation_model import *
import warnings
warnings.filterwarnings('ignore')


# Set the page title and layout
st.set_page_config(
    page_title="Theme Park Guest Experience Dashboard",
    page_icon="🎢",
    layout="wide",
    initial_sidebar_state="expanded")

# alt.themes.enable("dark")

# To check if the dashboard is running in streamlit
st.success("✅ Dashboard is running!")
st.title("🎡 Welcome to the Theme Park Guest Experience Dashboard")

#######################
# SIDEBAR


#######################

#######################
# PLOTS

# SUBGROUP A ##########
# QUESTION 1 ##########

# QUESTION 2 ##########

# QUESTION 3 ##########
script_dir_A3 = Path(__file__).resolve().parent
data_file_A3 = script_dir_A3.parent / 'data' / 'processed data' / 'tivoli_g.csv'

if data_file_A3.exists():
    tivoli_g = pd.read_csv(data_file_A3)
else:
    raise FileNotFoundError(f"Data file not found at {data_file_A3}")

tivoli_g["TIMESTAMP"] = pd.to_datetime(tivoli_g["TIMESTAMP"])
tivoli_g["WORK_DATE"] = pd.to_datetime(tivoli_g["WORK_DATE"])

analysis = GuestJourneyAnalysis(tivoli_g)

guest_summary = analysis.generate_guest_summary(tivoli_g)
guest_summary['SEQ_LEN'] = guest_summary['RIDE_SEQUENCE'].apply(len)
guest_summary['SEQ_ENTROPY'] = guest_summary['RIDE_SEQUENCE'].apply(analysis.compute_sequence_entropy)

# Wait Time Analysis
top_rides = tivoli_g['ATTRACTION'].value_counts().nlargest(3).index.tolist()
wait_df = analysis.guest_avg_wait_top_rides(tivoli_g, top_rides)


st.subheader("🎢 Guest Wait Time Tolerance for Top 3 Rides")
box = alt.Chart(wait_df).mark_boxplot(extent='min-max', size=50).encode(
    x=alt.X('DAY_TYPE:N', title='Day Type', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('AVG_WAIT_TIME_TOP_3:Q', title='Average Wait Time (mins)'),
    color=alt.Color('DAY_TYPE:N', legend=None)
).properties(
    width=800,
    height=400,
    title=''#not necessary since we have the subheader?
)

points = alt.Chart(wait_df).mark_circle(size=30, opacity=0.3, color='black').encode(
    x='DAY_TYPE:N',
    y='AVG_WAIT_TIME_TOP_3:Q'
)

st.altair_chart(box + points, use_container_width=True)

# Ride Length According to Express Pass Usage

guest_summary['EXPRESS_PASS_LABEL'] = guest_summary['EXPRESS_PASS'].map({True: "Yes", False: "No"})
st.subheader("Guest Journey Length (Express vs Non-Express)")
chart = alt.Chart(guest_summary).mark_boxplot(size=50).encode(
    x=alt.X('EXPRESS_PASS_LABEL:N', title='Express Pass', axis=alt.Axis(labelAngle=0)),
    y=alt.Y('SEQ_LEN:Q', title='Number of Rides'),
    color=alt.Color('EXPRESS_PASS_LABEL:N', legend=None)
).properties(
    width=600,
    height=400,
    title='' #not necessary since we have the subheader? will edit later to further standardise again
)

st.altair_chart(chart, use_container_width=True)

# QUESTION 4 ###########



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
                       
