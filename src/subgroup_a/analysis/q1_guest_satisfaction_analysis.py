import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import PercentFormatter
import matplotlib
# matplotlib.use('Qt5Agg')
import time
_original_show = plt.show

# Define custom show behavior
def auto_show_and_close(*args, **kwargs):
    _original_show(*args, **kwargs)  # show the plot
    time.sleep(3)                    # wait for 2 seconds
    plt.close('all')                 # close all figures after showing

# Override the default plt.show()
plt.show = auto_show_and_close
from pathlib import Path
from textblob import TextBlob

class GuestSatisfactionAnalysis:
    def __init__(self, survey_df=None):
        """
        Initialize with survey data.
        If no DataFrame is provided, it will load from default path.
        """
        if survey_df is not None:
            self.df = survey_df
        else:
            self.df = self.load_survey_data()
        
        # Set up visualization style
        sns.set(style="whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        
    def load_survey_data(self):
        """Load survey data from CSV file"""
        script_dir = Path(__file__).parent.parent
        project_root = script_dir.parent.parent
        csv_path = project_root / "data" / "raw data" / "survey.csv"
        
        print(f"Absolute CSV path: {csv_path}")
        print(f"File exists? {csv_path.exists()}")
        
        return pd.read_csv(csv_path)
    
    def run_analysis(self):
        """Run all analysis components"""
        print("\n=== Running Guest Satisfaction Analysis ===")
        
        self.analyze_park_experience()
        self.analyze_rides()
        self.analyze_food()
        self.analyze_staff()
        self.analyze_pricing()
        self.analyze_sentiment()
        self.correlation_analysis()
        self.analyze_post_visit()
        self.calculate_nps()
        
        print("\n=== Guest Satisfaction Analysis Complete ===")
    
    def analyze_park_experience(self):
        """Analyze crowd level concerns and satisfaction"""
        print("\n1. Park Experience Analysis:")
        crowd_concerns = self.df['What concerns did you have before deciding to visit USS?'].str.contains(
            'Ride wait times and crowd levels', na=False)
        crowd_satisfaction = self.df.loc[crowd_concerns, 
            'On a scale of 1-5, how would you rate your overall experience at USS?'].mean()
        print(f"Average satisfaction from visitors concerned about crowds: {crowd_satisfaction:.2f}/5")
    
    def analyze_rides(self):
        """Analyze ride wait times and popularity"""
        print("\n2. Ride Analysis:")
        # Ride wait times vs satisfaction
        plt.figure()
        sns.boxplot(x='How long did you wait in line for rides on average during your visit?',
                    y='On a scale of 1-5, how would you rate your overall experience at USS?',
                    data=self.df,
                    order=['Less than 15 minutes', '15 to 30 minutes', '31 to 45 minutes',
                           '46 to 60 minutes', '61 to 90 minutes', '90+ minutes'])
        plt.title('Ride Wait Times vs Overall Satisfaction')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # plt.savefig('rides_wait_time_vs_satisfaction.png')
        plt.show()

        # Most popular rides
        ride_counts = self.df['Which ride or attraction was your favourite?'].value_counts().head(5)
        plt.figure()
        ride_counts.plot(kind='barh')
        plt.title('Top 5 Most Popular Rides/Attractions')
        plt.xlabel('Number of Votes')
        plt.tight_layout()
        # plt.savefig('top_rides.png')
        plt.show()
    
    def analyze_food(self):
        """Analyze food quality and variety"""
        print("\n3. Food and Beverage Analysis:")
        # Food quality 
        plt.figure()
        self.df[' How would you rate the food quality and service?  '].value_counts(
            normalize=True).sort_index().plot(kind='bar')
        plt.title('Food Quality Ratings Distribution')
        plt.ylabel('Percentage of Responses')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.tight_layout()
        #plt.savefig('food_quality_distribution.png')
        plt.show(block=True)

        # Food variety 
        if not self.df[' Did you find a good variety of food options?  '].dropna().empty:
            food_variety = self.df[' Did you find a good variety of food options?  '].value_counts(normalize=True)
            plt.figure()
            food_variety.plot(kind='pie', autopct='%1.1f%%')
            plt.title('Perception of Food Variety')
            plt.ylabel('')
            plt.tight_layout()
            #plt.savefig('food_variety.png')
            plt.show(block=True)
            plt.pause(0.1)
    
    def analyze_staff(self):
        """Analyze staff friendliness and correlation with satisfaction"""
        print("\n4. Staff Friendliness Analysis:")
        # Staff friendliness 
        plt.figure()
        self.df['Were the park staff at USS friendly and helpful? Rate on a scale from 1-5.'].value_counts(
            normalize=True).sort_index().plot(kind='bar')
        
        plt.title('Staff Friendliness Ratings')
        plt.xlabel('Rating (1-5)')
        plt.ylabel('Percentage of Responses')
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.tight_layout()
        #plt.savefig('staff_friendliness.png')
        plt.show(block=True)
        plt.pause(0.1)

        # Correlation
        staff_corr = self.df[['Were the park staff at USS friendly and helpful? Rate on a scale from 1-5.',
                            'On a scale of 1-5, how would you rate your overall experience at USS?']].corr().iloc[0,1]
        print(f"Correlation between staff rating and overall experience: {staff_corr:.2f}")
    
    def analyze_pricing(self):
        """Analyze pricing concerns and express pass impact"""
        print("\n5. Pricing & Value Analysis:")
        # Ticket price concerns
        price_concern = self.df['What concerns did you have before deciding to visit USS?'].str.contains(
            'Ticket pricing and affordability', na=False)
        price_satisfaction = self.df[price_concern][
            'On a scale of 1-5, how would you rate your overall experience at USS?'].mean()
        print(f"Average satisfaction from price-conscious visitors: {price_satisfaction:.2f}/5")

        # Express pass analysis
        express_pass_corr = self.df[['Did you purchase the Express Pass?',
                                   'On a scale of 1-5, how would you rate your overall experience at USS?']].apply(
                                       lambda x: x.astype('category').cat.codes).corr().iloc[0,1]
        print(f"Correlation between Express Pass purchase and satisfaction: {express_pass_corr:.2f}")
    
    def analyze_sentiment(self):
        """Perform sentiment analysis on feedback"""
        print("\n6. Sentiment Analysis:")
        WEBSITE_FEEDBACK_COL = 'What feedback/suggestions would you like to provide for USS website/app?'
        IMPROVEMENTS_COL = 'What changes or improvements would make your next visit better?  ' 

        def get_sentiment(text):
            if pd.isna(text) or str(text).strip() in ['', 'NIL', 'Nil', 'NA']:
                return 0  # Neutral
            analysis = TextBlob(str(text))
            return analysis.sentiment.polarity

        # Apply sentiment analysis 
        self.df['Website_App_Sentiment'] = self.df[WEBSITE_FEEDBACK_COL].apply(get_sentiment)
        self.df['Improvements_Sentiment'] = self.df[IMPROVEMENTS_COL].apply(get_sentiment)

        # Plot sentiment distribution for website/app feedback
        plt.figure()
        sns.histplot(self.df['Website_App_Sentiment'], bins=np.arange(-1, 1.1, 0.1), kde=True)
        plt.title('Sentiment Distribution for Website/App Feedback')
        plt.xlabel('Sentiment Polarity (-1 to 1)')
        plt.xlim(-1, 1)
        plt.tight_layout()
        #plt.savefig('website_app_feedback_sentiment.png')
        plt.show()

        # Plot improvement suggestions
        plt.figure()
        sns.histplot(self.df['Improvements_Sentiment'], bins=np.arange(-1, 1.1, 0.1), kde=True)
        plt.title('Sentiment Distribution for Improvement Suggestions')
        plt.xlabel('Sentiment Polarity (-1 to 1)')
        plt.xlim(-1, 1)
        plt.tight_layout()
        # plt.savefig('improvement_suggestions_sentiment.png')
        plt.show()

        # Calculate correlation with overall experience
        sentiment_corr = self.df[['Website_App_Sentiment', 'Improvements_Sentiment',
                                'On a scale of 1-5, how would you rate your overall experience at USS?']].corr()
        print("\n=== Sentiment Correlation ===")
        print(sentiment_corr)
    
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

        # Plot the heatmap
        mask = np.triu(np.ones(corr_subset.shape, dtype=bool)) 
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0,
                    annot_kws={'size': 9}, fmt=".2f", mask=mask)
        plt.title('Correlation Matrix of Guest Experience Factors')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        # plt.savefig('correlation_heatmap.png')
        plt.show()
    
    def analyze_post_visit(self):
        """Analyze revisit intention"""
        print("\n8. Post-Visit Analysis:")
        revisit_dist = self.df['Would you choose to revisit USS?'].value_counts(normalize=True)
        plt.figure()
        revisit_dist.plot(kind='bar')
        plt.title('Revisit Intention')
        plt.ylabel('Percentage of Respondents')
        plt.xticks(rotation=45)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.tight_layout()
        # plt.savefig('revisit_intention.png')
        plt.show()

        print(revisit_dist)
    
    def calculate_nps(self):
        """Calculate Net Promoter Score"""
        print("\n9. Net Promoter Score:")
        def nps_calc(recommend_series):
            promoters = (recommend_series == 'Yes').sum()  
            detractors = (recommend_series == 'No').sum()
            total = recommend_series.count()
            return ((promoters - detractors) / total) * 100

        nps = nps_calc(self.df['Would you recommend USS to others?'])
        print(f"Net Promoter Score: {nps:.1f}")

if __name__ == "__main__":
    analysis = GuestSatisfactionAnalysis()
    analysis.run_analysis()
