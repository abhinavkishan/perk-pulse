import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
import torch.nn.functional as F
import math
import difflib

# Load the data and model
df = pd.read_csv('../data/flights.csv')
num_classes = df['Card'].nunique()

model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=num_classes)

model.load_state_dict(torch.load('flight_model.pth'))
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

sentiment_analyzer = pipeline(
    "sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

offer_type_weights = {
    'free upgrade': 10.0,
    'extra baggage': 8.0,
    'priority boarding': 7.0,
    'lounge access': 6.0,
    'discount': 5.0,
}


def get_sentiment_score(text):
    result = sentiment_analyzer(text)[0]
    if result['label'] == 'POSITIVE':
        return 0.7 + (result['score'] / 3)
    else:
        return 0.7 - (result['score'] / 3)


def get_offer_type_score(offer_text):
    offer_text = offer_text.lower()
    for keyword, weight in offer_type_weights.items():
        if keyword in offer_text:
            return weight
    return 1.0


def scale_score(score):
    return 1 / (1 + math.exp(-10 * (score - 0.1)))


def predict(card_names):
    predictions = []

    for card_name in card_names:
        # Filter the DataFrame to get offers for the current card
        card_offers = df[df['Card'] == card_name]['Offer'].tolist()

        if not card_offers:  # Check if there are no offers for the card
            print(f"No offer available for {card_name}")
            predictions.append(
                {'card': card_name, 'offer': None, 'score': None})
            continue

        best_offer = None
        best_score = -float('inf')

        for offer_text in card_offers:
            sentiment_score = get_sentiment_score(offer_text)
            offer_type_score = get_offer_type_score(offer_text)
            combined_score = scale_score(sentiment_score * offer_type_score)

            if combined_score > best_score:
                best_score = combined_score
                best_offer = offer_text

        if best_offer:
            predictions.append(
                {'card': card_name, 'offer': best_offer, 'score': best_score})

    return predictions


# Example usage
card_list = ["ICICI Bank Rubyx Credit Card",
             "HSBC Bank Credit Card", "Amazon Pay ICICI Bank Credit Card"]

predictions = predict(card_list)

for prediction in predictions:
    if prediction['offer'] is None:
        print(f"{prediction['card']}: No offer available")
    else:
        print(
            f"{prediction['card']}: {prediction['offer']} (Score: {prediction['score']:.4f})")
