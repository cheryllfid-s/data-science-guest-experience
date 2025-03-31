from datacleaning_A import *
from guest_journey import GuestJourneyAnalysis

def mainA():
    tivoli_g, attendance_df, covid, negative_att = prepare_tivoli_data()
    GuestJourneyAnalysis_obj = GuestJourneyAnalysis(tivoli_g, attendance_df, covid, negative_att)
    Print("Question 3: Guest Journey Patterns")
    GuestJourneyAnalysis_obj.run_guestjourneyanalysis()

if __name__ == "__main__":
    mainA()
