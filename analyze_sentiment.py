
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- Reusable Components ---
# Load the FinBERT model and tokenizer once, making them available for import
print("Initializing FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def get_sentiment(text):
    """
    Analyzes the sentiment of a given text using the pre-loaded FinBERT model.
    Returns positive, negative, neutral scores, and a final label.
    """
    if pd.isna(text) or text.strip() == "":
        return None, None, None, "Neutral" # Return neutral for empty/NaN text

    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        positive = predictions[:, 0].item()
        negative = predictions[:, 1].item()
        neutral = predictions[:, 2].item()

        if positive > negative and positive > neutral:
            label = "Positive"
        elif negative > positive and negative > neutral:
            label = "Negative"
        else:
            label = "Neutral"
        
        return positive, negative, neutral, label
    except Exception as e:
        print(f"Could not process text: {text[:80]}... | Error: {e}")
        return None, None, None, "Neutral"

# --- Original Script Logic ---
def analyze_news_file():
    """
    Original functionality to analyze the mediastack news file.
    """
    print("Running original news sentiment analysis...")
    try:
        df = pd.read_csv('REL_news_mediastack.csv')
    except FileNotFoundError:
        print("Error: REL_news_mediastack.csv not found. Please ensure the news data is fetched first.")
        return

    # Apply sentiment analysis
    df['positive_score'], df['negative_score'], df['neutral_score'], df['sentiment_label'] = zip(*df['description'].apply(get_sentiment))

    # If description is empty, try title
    empty_description_indices = df[df['description'].isna() | (df['description'].str.strip() == "")].index
    for idx in empty_description_indices:
        title_text = df.loc[idx, 'title']
        positive, negative, neutral, label = get_sentiment(title_text)
        df.loc[idx, 'positive_score'] = positive
        df.loc[idx, 'negative_score'] = negative
        df.loc[idx, 'neutral_score'] = neutral
        df.loc[idx, 'sentiment_label'] = label

    # Save the results
    df.to_csv('REL_news_sentiment.csv', index=False)
    print("Sentiment analysis complete. Results saved to REL_news_sentiment.csv")

if __name__ == "__main__":
    # This block runs only when the script is executed directly
    analyze_news_file()
