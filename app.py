import os
import pickle

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
LABELS_PATH = os.path.join(BASE_DIR, 'sentiment_labels.pkl')


with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

with open(VECTORIZER_PATH, 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

with open(LABELS_PATH, 'rb') as labels_file:
    labels = pickle.load(labels_file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True) or {}
    review = data.get('review', '')

    if not review or not str(review).strip():
        return jsonify({
            'error': 'Please provide a review text.'
        }), 400

    transformed = tfidf.transform([str(review)])
    prediction_code = model.predict(transformed)[0]
    probabilities = model.predict_proba(transformed)[0]

    if isinstance(prediction_code, str):
        sentiment = prediction_code.capitalize()
    else:
        sentiment = labels[int(prediction_code)].capitalize()

    confidence = float(max(probabilities))

    return jsonify({
        'review': review,
        'sentiment': sentiment,
        'confidence': round(confidence, 4),
        'confidence_percent': round(confidence * 100, 2),
        'prediction_label': str(prediction_code)
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
