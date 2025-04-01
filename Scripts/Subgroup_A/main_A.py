from datacleaning_A import *
from Analysis.guest_satisfaction import GuestSatisfactionAnalysis
from Modeling.guest_segmentation_model import guest_segmentation_model
from Analysis.guest_journey import GuestJourneyAnalysis
from Analysis.promo_events_analysis import USSPromoAnalysis
from Analysis.extfactors import q5_analyse

def mainA():
    satisfaction_analysis = GuestSatisfactionAnalysis()
    print("Question 1: Key Factors Influencing Guest Satisfaction")
    satisfaction_analysis.run_analysis()

    print("Question 2: Guest Segmentation Model")
    df_combined, df_labeled, scaled, pca = cleaning_q2()
    segmentation_analysis = guest_segmentation_model(df_combined, df_labeled, scaled, pca)
    summary, _ = segmentation_analysis.run_pipeline()
    print("\n===== SILHOUETTE SCORES BY MODEL (Q2) =====")
    print("\n===== CLUSTER SUMMARY (Q2) =====")
    print(summary)
    
    tivoli_g, attendance_df, covid, negative_att = prepare_tivoli_data()
    GuestJourneyAnalysis_obj = GuestJourneyAnalysis(tivoli_g, attendance_df, covid, negative_att)
    print("Question 3: Guest Journey Patterns")
    GuestJourneyAnalysis_obj.run_guestjourneyanalysis()

    print("Question 4: Impact of Marketing Strategies on Guest Behaviour")
    print("")
    q4_df_reviews = q4_prepare_reviews_data()
    q4_df_events = q4_prepare_events_data()
    USSPromoAnalysis_obj = USSPromoAnalysis(q4_df_reviews, q4_df_events)
    USSPromoAnalysis_obj.run_promo_events_analysis()
    segmentation_analysis.run_marketing_analysis()

    print("Question 5: External Factors on Guest Satisfaction")
    q5_df_reviews = q5_clean_data()
    q5_analyse(q5_df_reviews)

if __name__ == "__main__":
    mainA()
