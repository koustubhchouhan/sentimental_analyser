import os
import pickle

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'Dataset-SA.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'sentiment_model.pkl')
VECTORIZER_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')
LABELS_PATH = os.path.join(BASE_DIR, 'sentiment_labels.pkl')


def train_and_save_model():
    df = pd.read_csv(DATA_PATH, usecols=['Review', 'Sentiment'])
    df = df.drop_duplicates()
    df = df.dropna(subset=['Review', 'Sentiment']).copy()
    df['Review'] = df['Review'].astype(str).str.strip()
    df['Sentiment'] = df['Sentiment'].astype(str).str.strip().str.lower()
    df = df[df['Sentiment'].isin(['positive', 'negative', 'neutral'])]

    # Downsample majority classes to match the minority class count.
    min_count = df['Sentiment'].value_counts().min()
    balanced_df = (
        df.groupby('Sentiment', group_keys=False)
        .apply(lambda part: part.sample(n=min_count, random_state=42))
        .reset_index(drop=True)
    )

    X = balanced_df['Review']
    y = balanced_df['Sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    model = MultinomialNB(alpha=0.5)
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    print(classification_report(y_test, y_pred))

    with open(MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)

    with open(VECTORIZER_PATH, 'wb') as file:
        pickle.dump(tfidf, file)

    with open(LABELS_PATH, 'wb') as file:
        pickle.dump(sorted(balanced_df['Sentiment'].unique().tolist()), file)

    print(f'Model saved to: {MODEL_PATH}')
    print(f'Vectorizer saved to: {VECTORIZER_PATH}')
    print(f'Labels saved to: {LABELS_PATH}')


if __name__ == '__main__':
    train_and_save_model()
