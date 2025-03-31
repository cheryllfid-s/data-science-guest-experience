from datacleaning_A import *
from guest_satisfaction import GuestSatisfactionAnalysis
from guest_journey import GuestJourneyAnalysis
from promo_events_analysis import USSPromoAnalysis
from extfactors import analyse

def mainA():
    satisfaction_analysis = GuestSatisfactionAnalysis()
    print("Question 1: Key Factors Influencing Guest Satisfaction")
    satisfaction_analysis.run_analysis()
    
    tivoli_g, attendance_df, covid, negative_att = prepare_tivoli_data()
    GuestJourneyAnalysis_obj = GuestJourneyAnalysis(tivoli_g, attendance_df, covid, negative_att)
    print("Question 3: Guest Journey Patterns")
    GuestJourneyAnalysis_obj.run_guestjourneyanalysis()

    print("Question 4: Impact of Marketing Strategies on Guest Behaviour")
    q4_df_reviews = q4_prepare_reviews_data()
    q4_df_events = q4_prepare_events_data()
    USSPromoAnalysis_obj = USSPromoAnalysis(q4_df_reviews, q4_df_events)
    USSPromoAnalysis_obj.run_promo_events_analysis()

    print("Question 5: External Factors on Guest Satisfaction")
    q5_df_reviews = q5_clean_data()
    analyse(q5_df_reviews)

if __name__ == "__main__":
    mainA()
