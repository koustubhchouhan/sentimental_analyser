import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
LABELS_PATH = os.path.join(BASE_DIR, 'sentiment_labels.pkl')


def predict_sentiment(review_text):
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)

    with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
        tfidf = pickle.load(vectorizer_file)

    with open(LABELS_PATH, 'rb') as labels_file:
        labels = pickle.load(labels_file)

    transformed = tfidf.transform([str(review_text)])
    prediction = model.predict(transformed)[0]
    probabilities = model.predict_proba(transformed)[0]

    if isinstance(prediction, str):
        sentiment = prediction.capitalize()
    else:
        sentiment = labels[int(prediction)].capitalize()

    confidence = float(max(probabilities))

    return {
        'sentiment': sentiment,
        'confidence': confidence,
        'prediction_label': str(prediction),
    }


if __name__ == '__main__':
    review = input('Enter a product review: ')
    result = predict_sentiment(review)
    print(f"Sentiment: {result['sentiment']}")
    print(f"Confidence: {result['confidence']:.2%}")
