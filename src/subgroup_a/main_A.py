from datapreparation_A import *
from analysis.q1_guest_satisfaction_analysis import GuestSatisfactionAnalysis
from modeling.q2_guest_segmentation_model import guest_segmentation_model
from analysis.q3_guest_journey_analysis import GuestJourneyAnalysis
from analysis.q4_promo_events_analysis import USSPromoAnalysis
from analysis.q5_external_factors_analysis import q5_analyse
import warnings

def mainA():
    warnings.filterwarnings("ignore")
    
    satisfaction_analysis = GuestSatisfactionAnalysis()
    print("Question 1: Key Factors Influencing Guest Satisfaction")
    satisfaction_analysis.run_analysis()

    print("\nQuestion 2: Guest Segmentation Model")
    df_combined, df_labeled, scaled, pca = cleaning_q2()
    segmentation_analysis = guest_segmentation_model(df_combined, df_labeled, scaled, pca)
    summary, _ = segmentation_analysis.run_pipeline()
    print("\n===== SILHOUETTE SCORES BY MODEL (Q2) =====")
    print("\n===== CLUSTER SUMMARY (Q2) =====")
    print(summary)

    print("\nQuestion 3: Guest Journey Patterns")
    tivoli_g, attendance_df, covid, negative_att = prepare_tivoli_data()
    GuestJourneyAnalysis_obj = GuestJourneyAnalysis(tivoli_g, attendance_df, covid, negative_att)
    GuestJourneyAnalysis_obj.run_guestjourneyanalysis()

    print("\nQuestion 4: Impact of Marketing Strategies on Guest Behaviour")
    print("--- Data Preparation ---")
    q4_df_reviews = q4_prepare_reviews_data()
    q4_df_events = q4_prepare_events_data()
    USSPromoAnalysis_obj = USSPromoAnalysis(q4_df_reviews, q4_df_events)
    USSPromoAnalysis_obj.run_promo_events_analysis()
    segmentation_analysis.run_marketing_analysis()

    print("\nQuestion 5: External Factors on Guest Satisfaction")
    q5_df_reviews = q5_clean_data()
    q5_analyse(q5_df_reviews)

if __name__ == "__main__":
    mainA()
