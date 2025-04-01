import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import torch
import logging
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import copy
import pickle
import os

#Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Download NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Read data
try:
    disney_df = pd.read_csv('../../data/raw/DisneylandReviews.csv', encoding='utf-8', encoding_errors='replace')
    print(f"Successfully loaded Disney data, {len(disney_df)} rows")
except FileNotFoundError:
    raise FileNotFoundError("DisneylandReviews.csv not found. Please ensure the file exists in the current directory.")
except Exception as e:
    raise Exception(f"Error reading Disney data: {str(e)}")

# Validate data types
disney_df['Rating'] = pd.to_numeric(disney_df['Rating'], errors='coerce')
disney_df['Review_Text'] = disney_df['Review_Text'].astype(str)

try:
    uss_df = pd.read_csv('../../data/raw/universal_studio_branches.csv',
                         encoding='utf-8',
                         encoding_errors='replace',
                         engine='python',
                         on_bad_lines='skip')
except:
    try:
        uss_df = pd.read_csv('../../data/raw/universal_studio_branches.csv',
                             encoding='utf-8',
                             encoding_errors='replace',
                             engine='python',
                             error_bad_lines=False)
    except:
        uss_df = pd.read_csv('../../data/raw/universal_studio_branches.csv',
                             encoding='utf-8',
                             encoding_errors='replace',
                             engine='python',
                             sep=',',
                             quoting=3,
                             escapechar='\\')

print(f"Successfully loaded Universal Studios data, {len(uss_df)} rows")

# Rename columns
disney_renamed = disney_df.rename(columns={
    'Review_ID': 'review_id',
    'Rating': 'rating',
    'Year_Month': 'date',
    'Reviewer_Location': 'reviewer_location',
    'Review_Text': 'review_text',
    'Branch': 'branch'
})

uss_renamed = uss_df.rename(columns={
    'reviewer': 'reviewer_name',
    'written_date': 'date',
    'title': 'review_title',
    'review_text': 'review_text'
})

# Add theme park type labels
disney_renamed['park_type'] = 'Disney'
uss_renamed['park_type'] = 'USS'

# Add missing columns
if 'review_id' not in uss_renamed.columns:
    uss_renamed['review_id'] = uss_renamed.index + len(disney_renamed)
if 'review_title' not in disney_renamed.columns:
    disney_renamed['review_title'] = np.nan

# Merge datasets
common_columns = ['review_id', 'rating', 'date', 'reviewer_location', 'reviewer_name', 'review_text', 'review_title', 'branch', 'park_type']

#Ensure both datasets have required columns
if 'reviewer_location' not in uss_renamed.columns:
    uss_renamed['reviewer_location'] = np.nan  # Add missing location column
if 'reviewer_name' not in disney_renamed.columns:
    disney_renamed['reviewer_name'] = np.nan  # Add missing name column

disney_common = disney_renamed[common_columns]
uss_common = uss_renamed[common_columns]
combined_df = pd.concat([disney_common, uss_common], ignore_index=True)


# Process dates
def standardize_date(date_str):
    try:
        if isinstance(date_str, str) and len(date_str.split('-')) == 2:
            year, month = date_str.split('-')
            return f"{year}-{month}-01"
        elif isinstance(date_str, str):
            return pd.to_datetime(date_str).strftime('%Y-%m-%d')
        else:
            return np.nan
    except:
        return np.nan

combined_df['standardized_date'] = combined_df['date'].apply(standardize_date)
combined_df['standardized_date'] = pd.to_datetime(combined_df['standardized_date'], errors='coerce')
combined_df['year'] = combined_df['standardized_date'].dt.year
combined_df['month'] = combined_df['standardized_date'].dt.month


# Convert ratings to numeric
combined_df['rating'] = pd.to_numeric(combined_df['rating'], errors='coerce')

# Create negative experience label (rating <= 4 is negative)
combined_df['bad_experience'] = (combined_df['rating'] <= 4).astype(int)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# 1. Sentiment Analysis
combined_df['sentiment_score'] = combined_df['review_text'].apply(lambda x: sia.polarity_scores(str(x))['compound'])
combined_df['sentiment_negative'] = combined_df['review_text'].apply(lambda x: sia.polarity_scores(str(x))['neg'])
combined_df['sentiment_positive'] = combined_df['review_text'].apply(lambda x: sia.polarity_scores(str(x))['pos'])

# 2. Keyword Extraction
negative_reviews = combined_df[combined_df['bad_experience'] == 1]['review_text']
positive_reviews = combined_df[combined_df['bad_experience'] == 0]['review_text']

tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
tfidf_neg = tfidf.fit_transform(negative_reviews)
tfidf_pos = tfidf.transform(positive_reviews)

feature_names = tfidf.get_feature_names_out()
neg_tfidf_means = tfidf_neg.mean(axis=0).A1
pos_tfidf_means = tfidf_pos.mean(axis=0).A1
diff_means = neg_tfidf_means - pos_tfidf_means

top_diff_indices = diff_means.argsort()[-100:][::-1]
top_neg_words = [feature_names[i] for i in top_diff_indices]

logging.info("\nTop 20 words most distinguishing in negative reviews:")
logging.info(top_neg_words[:20])

# BERT model training and explanation
def tokenize_function(texts, tokenizer, max_length=128):
    return tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

def train_bert_model(train_texts, train_labels, val_texts, val_labels, num_epochs=3):
    # Input validation
    if not train_texts or not train_labels or not val_texts or not val_labels:
        raise ValueError("Training and validation data cannot be empty")
    if len(train_texts) != len(train_labels) or len(val_texts) != len(val_labels):
        raise ValueError("Number of texts and labels must match")

    # Load pre-trained model and tokenizer
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
        model.to(device)
    except Exception as e:
        raise Exception(f"Error loading BERT model: {str(e)}")

    # Prepare data with error handling
    try:
        train_encodings = tokenize_function(train_texts, tokenizer)
        val_encodings = tokenize_function(val_texts, tokenizer)

        train_dataset = TensorDataset(
            train_encodings['input_ids'].to(device),
            train_encodings['attention_mask'].to(device),
            torch.tensor(train_labels, dtype=torch.long).to(device)
        )
        val_dataset = TensorDataset(
            val_encodings['input_ids'].to(device),
            val_encodings['attention_mask'].to(device),
            torch.tensor(val_labels, dtype=torch.long).to(device)
        )
    except Exception as e:
        raise Exception(f"Error preparing datasets: {str(e)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Train model
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    best_val_accuracy = 0
    best_model = None

    try:
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                optimizer.zero_grad()
                outputs = model(input_ids=batch[0], attention_mask=batch[1], labels=batch[2])
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss/len(train_loader)
            print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

            # Validation
            model.eval()
            val_preds = []
            val_true = []
            with torch.no_grad():
                for batch in val_loader:
                    outputs = model(input_ids=batch[0], attention_mask=batch[1])
                    logits = outputs.logits
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_true.extend(batch[2].cpu().numpy())

            val_accuracy = accuracy_score(val_true, val_preds)
            print(f"Validation accuracy: {val_accuracy:.4f}")

            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                best_model = copy.deepcopy(model)

    except Exception as e:
        raise Exception(f"Error during training: {str(e)}")

    print(f"Best validation accuracy: {best_val_accuracy:.4f}")

    # save model and tokenizer
    # create models directory (if it doesn't exist)
    os.makedirs('models', exist_ok=True)

    # move model to CPU for saving
    best_model.to('cpu')

    # save model state dictionary
    torch.save(best_model.state_dict(), 'models/bert_model.pt')

    # save tokenizer
    with open('models/bert_tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    print("model and tokenizer saved to models directory")

    # move model back to original device
    best_model.to(device)

    return best_model, tokenizer

# Use SHAP to explain model predictions
def explain_predictions(model, tokenizer, texts, train_texts, num_samples=100):
    """
    Use hierarchical method to explain BERT predictions:
    1. First get overall predictions using BERT
    2. Then analyze word contributions using simple methods
    """
    # Input validation
    if not isinstance(model, BertForSequenceClassification):
        raise TypeError("Model must be a BertForSequenceClassification instance")
    if not texts or not train_texts:
        raise ValueError("Texts and train_texts cannot be empty")

    try:
        # Prepare sample texts
        sample_size = min(num_samples, len(texts))
        sample_texts = texts[:sample_size]
        sample_texts = [str(t) if not isinstance(t, str) else t for t in sample_texts]

        # Step 1: Get BERT predictions
        def get_bert_prediction(text):
            # Tokenize
            encoding = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )

            # Move to device
            encoding = {k: v.to(device) for k, v in encoding.items()}

            # Get prediction
            model.eval()
            with torch.no_grad():
                outputs = model(input_ids=encoding['input_ids'],
                             attention_mask=encoding['attention_mask'])
                probs = F.softmax(outputs.logits, dim=1)

            return probs.cpu().numpy()[0]

        # Step 2: Analyze word contributions
        def analyze_word_importance(text, pred_prob):
            """Analyze the importance of words in a single text"""
            words = text.split()
            importance_scores = []

            # For each word, test its importance by removing it
            for i, word in enumerate(words):
                # Create text without this word
                text_without_word = ' '.join(words[:i] + words[i+1:])
                # Get new prediction
                new_prob = get_bert_prediction(text_without_word)
                # Calculate importance score (original negative probability - negative probability after word removal)
                importance = pred_prob[1] - new_prob[1]
                importance_scores.append((word, importance))

            return importance_scores

        # Analyze all sample texts
        all_word_importance = []
        for text in tqdm(sample_texts, desc="Analyzing word importance"):
            # Get original prediction
            pred_prob = get_bert_prediction(text)
            # If predicted as negative sentiment
            if pred_prob[1] > 0.5:
                # Analyze word importance
                importance_scores = analyze_word_importance(text, pred_prob)
                all_word_importance.extend(importance_scores)

        # Summarize word importance
        word_importance_dict = {}
        for word, score in all_word_importance:
            if word not in word_importance_dict:
                word_importance_dict[word] = []
            word_importance_dict[word].append(score)

        # Calculate average importance
        avg_importance = {
            word: np.mean(scores)
            for word, scores in word_importance_dict.items()
            if len(scores) >= 3  # Only keep words that appear at least 3 times
        }

        # Visualisations
        try:
            plt.figure(figsize=(12, 8))

            # Obtain top 20 words contributing to negative sentiment
            top_words = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)[:20]
            words, importance = zip(*top_words)

            # Plot image
            plt.barh(range(len(words)), importance)
            plt.yticks(range(len(words)), words)
            plt.title("Top 20 Words Contributing to Negative Sentiment")
            plt.xlabel("Average Contribution to Negative Sentiment")
            plt.tight_layout()
            plt.savefig('word_importance.png')
            plt.close()

            # Print top words contributing to negative sentiment
            print("\nTop words contributing to negative sentiment:")
            for word, imp in top_words:
                print(f"{word}: {imp:.4f}")

        except Exception as e:
            print(f"Error in visualization: {str(e)}")

        return avg_importance

    except Exception as e:
        print(f"Error in explain_predictions: {str(e)}")
        return None

def get_explanation_for_text(model, tokenizer, text):
    """Generate explanation for a single text"""
    try:
        # Get prediction
        encoding = tokenizer(
            text,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids=encoding['input_ids'],
                          attention_mask=encoding['attention_mask'])
            probs = F.softmax(outputs.logits, dim=1)

        pred_prob = probs.cpu().numpy()[0]

        # If predicted as negative
        if pred_prob[1] > 0.5:
            # Analyze word contributions
            words = text.split()
            word_contributions = []

            for i, word in enumerate(words):
                # Create text without this word
                text_without_word = ' '.join(words[:i] + words[i+1:])

                # Get new prediction
                new_encoding = tokenizer(
                    text_without_word,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                new_encoding = {k: v.to(device) for k, v in new_encoding.items()}

                with torch.no_grad():
                    new_outputs = model(input_ids=new_encoding['input_ids'],
                                     attention_mask=new_encoding['attention_mask'])
                    new_probs = F.softmax(new_outputs.logits, dim=1)

                new_prob = new_probs.cpu().numpy()[0]

                # Calculate contribution
                contribution = pred_prob[1] - new_prob[1]
                if contribution > 0:  # Only focus on positive contributions (words that increase negative sentiment)
                    word_contributions.append((word, contribution))

            # Sort and return most important words
            word_contributions.sort(key=lambda x: x[1], reverse=True)
            return {
                'prediction': 'negative',
                'confidence': float(pred_prob[1]),
                'important_words': word_contributions[:5]  # Return top 5 most important words
            }
        else:
            return {
                'prediction': 'positive',
                'confidence': float(pred_prob[0]),
                'important_words': []
            }

    except Exception as e:
        print(f"Error in get_explanation_for_text: {str(e)}")
        return None

# Prepare data
from sklearn.model_selection import train_test_split
texts = combined_df['review_text'].tolist()
labels = combined_df['bad_experience'].tolist()

train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# Train model
print("\nStarting BERT model training...")
model, tokenizer = train_bert_model(train_texts, train_labels, val_texts, val_labels)

# Generate SHAP explanations
print("\nGenerating SHAP explanations...")
shap_values = explain_predictions(model, tokenizer, val_texts, train_texts)

# Analyze complaint severity
def classify_complaint_severity(rating, pred_prob):
    """
    Classify complaints based on rating and model prediction probability

    Args:
        rating (float): Review rating (1-5)
        pred_prob (float): Model's prediction probability for negative class (0-1)

    Returns:
        int: Severity level (0-3)
    """
    try:
        # Input validation
        if not isinstance(rating, (int, float)) or not isinstance(pred_prob, (int, float)):
            raise TypeError("Rating and pred_prob must be numeric")

        if not (0 <= pred_prob <= 1):
            raise ValueError("Prediction probability must be between 0 and 1")

        if not (1 <= rating <= 5):
            raise ValueError("Rating must be between 1 and 5")

        if rating > 4:
            return 0

        if rating <= 2 and pred_prob > 0.8:
            return 3
        elif rating <= 3 and pred_prob > 0.6:
            return 2
        elif rating <= 4:
            return 1
        else:
            return 0

    except Exception as e:
        print(f"Warning: Error in classify_complaint_severity: {str(e)}")
        return 0  # Return default value in case of error

# Predict all reviews
def predict_all(texts, model, tokenizer, batch_size=32):
    # Input validation
    if not texts:
        raise ValueError("Texts cannot be empty")
    if not isinstance(model, BertForSequenceClassification):
        raise TypeError("Model must be a BertForSequenceClassification instance")

    try:
        model.eval()
        predictions = []
        probabilities = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
            batch_texts = texts[i:i + batch_size]
            try:
                encodings = tokenize_function(batch_texts, tokenizer)
                with torch.no_grad():
                    outputs = model(
                        input_ids=encodings['input_ids'].to(device),
                        attention_mask=encodings['attention_mask'].to(device)
                    )
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=1)
                    preds = torch.argmax(logits, dim=1)

                    predictions.extend(preds.cpu().numpy())
                    probabilities.extend(probs[:, 1].cpu().numpy())
            except Exception as e:
                print(f"Warning: Error processing batch {i}-{i+batch_size}: {str(e)}")
                continue

        if not predictions:
            raise ValueError("No predictions were generated")

        return predictions, probabilities

    except Exception as e:
        raise Exception(f"Error in predict_all: {str(e)}")

print("\nPredicting all reviews...")
try:
    predictions, probabilities = predict_all(texts, model, tokenizer)

    # Add prediction results to dataframe with error handling
    combined_df['predicted_negative'] = predictions
    combined_df['negative_probability'] = probabilities
    combined_df['complaint_severity'] = combined_df.apply(
        lambda row: classify_complaint_severity(
            float(row['rating']) if pd.notnull(row['rating']) else 5.0,
            float(row['negative_probability']) if pd.notnull(row['negative_probability']) else 0.0
        ),
        axis=1
    )

except Exception as e:
    raise Exception(f"Error in final prediction and classification: {str(e)}")

# Analyze complaint distribution
print("\nComplaint severity distribution:")
severity_counts = combined_df['complaint_severity'].value_counts().sort_index()
print(severity_counts)
print(f"Complaint ratio: {(combined_df['complaint_severity'] > 0).mean():.2%}")

# Visualize complaint severity distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='complaint_severity', hue='park_type', data=combined_df)
plt.title('Complaint Severity Distribution')
plt.xlabel('Severity Level (0=No complaint, 3=Severe complaint)')
plt.ylabel('Count')
plt.savefig('complaint_severity_distribution.png')
plt.close()

# Analyze complaints by branch
branch_severity = combined_df.groupby(['branch', 'park_type'])['complaint_severity'].mean().reset_index()
print("\nAverage complaint severity by branch:")
print(branch_severity)

plt.figure(figsize=(14, 6))
sns.barplot(x='branch', y='complaint_severity', hue='park_type', data=branch_severity)
plt.title('Average Complaint Severity by Branch')
plt.xlabel('Branch')
plt.ylabel('Average Complaint Severity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('branch_severity.png')
plt.close()

# Design improvement recommendations
def recommend_improvements(severity, branch, negative_prob):
    """
    Provide improvement recommendations based on complaint severity and prediction probability
    """
    if severity == 0:
        return "No action needed"

    actions = {
        1: "Send apology email and small discount coupon",
        2: "Customer service team to contact and provide partial refund or free upgrade",
        3: "Management intervention, full refund or free re-visit, and root cause analysis"
    }

    base_action = actions[severity]

    # Add specific recommendations based on SHAP analysis
    if 'Disney' in branch:
        if negative_prob > 0.8:
            return f"{base_action}, focus on queue time management and facility maintenance"
        else:
            return f"{base_action}, focus on service quality improvement"
    else:  # USS
        if negative_prob > 0.8:
            return f"{base_action}, focus on visitor experience and facility availability"
        else:
            return f"{base_action}, focus on service standardization"

combined_df['recommended_action'] = combined_df.apply(
    lambda row: recommend_improvements(
        row['complaint_severity'],
        row['branch'],
        row['negative_probability']
    ),
    axis=1
)

# Analyze improvement recommendation distribution
action_counts = combined_df['recommended_action'].value_counts()
print("\nImprovement recommendation distribution:")
print(action_counts)

# Summarize analysis results
print("\nComplaint Analysis Summary:")
print(f"1. Total reviews: {len(combined_df)}")
print(f"2. Negative experience ratio: {combined_df['bad_experience'].mean():.2%}")
print(f"3. Complaint severity distribution: {severity_counts.to_dict()}")
print(f"4. BERT model performance: accuracy {accuracy_score(val_labels, predictions[-len(val_labels):]):.4f}")
print("5. Main problem areas (based on SHAP analysis):")
print("   - Please check word_importance.png for detailed word importance")
print("\n6. Complaint situation by branch:")
print(branch_severity.sort_values('complaint_severity', ascending=False))

print("\nGenerating explanations for negative reviews...")
try:
    # Get negative review samples
    negative_reviews = combined_df[combined_df['bad_experience'] == 1]['review_text'].tolist()[:100]

    # Generate explanations for each negative review
    explanations = []
    for review in tqdm(negative_reviews, desc="Analyzing negative reviews"):
        explanation = get_explanation_for_text(model, tokenizer, review)
        if explanation and explanation['prediction'] == 'negative':
            explanations.append(explanation)

    # Summarize most common negative words
    word_freq = {}
    for exp in explanations:
        for word, score in exp['important_words']:
            if word not in word_freq:
                word_freq[word] = {'count': 0, 'total_score': 0}
            word_freq[word]['count'] += 1
            word_freq[word]['total_score'] += score

    # Calculate average impact score
    for word in word_freq:
        word_freq[word]['avg_score'] = word_freq[word]['total_score'] / word_freq[word]['count']

    # Get most common negative words
    common_negative_words = sorted(
        [(word, info['count'], info['avg_score'])
         for word, info in word_freq.items() if info['count'] >= 3],
        key=lambda x: x[2],
        reverse=True
    )

    print("\nMost common words in negative reviews:")
    for word, count, score in common_negative_words[:10]:
        print(f"{word}: appeared {count} times, average impact score: {score:.4f}")

except Exception as e:
    print(f"Error in generating explanations: {str(e)}")

# done
print("\ndone.")

# add load model function
def load_bert_model(model_path='models/bert_model.pt', tokenizer_path='models/bert_tokenizer.pkl'):
    """
    load saved BERT model and tokenizer

    Args:
        model_path: path to model state dictionary
        tokenizer_path: path to tokenizer pickle file

    Returns:
        model: loaded BERT model
        tokenizer: loaded tokenizer
    """
    try:
        # load tokenizer
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)

        # initialize model
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

        # load model state
        model.load_state_dict(torch.load(model_path))

        # set to evaluation mode
        model.eval()

        print("successfully loaded model and tokenizer")
        return model, tokenizer

    except Exception as e:
        print(f"error loading model: {str(e)}")
        return None, None
