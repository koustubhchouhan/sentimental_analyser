#Phase 1: Environment Setup & Data Loading
import numpy as np
import matplotlib.pyplot as plt
import pandas
import seaborn as sns
import warnings

class pd:
    def __getattr__(self, name):
        return getattr(pandas, name)

    @staticmethod
    def read_csv(*args, **kwargs):
        return pandas.read_csv(*args, **kwargs)

    @staticmethod
    def DataFrame(*args, **kwargs):
        return pandas.DataFrame(*args, **kwargs)

    @staticmethod
    def Series(*args, **kwargs):
        return pandas.Series(*args, **kwargs)

    @staticmethod
    def concat(*args, **kwargs):
        return pandas.concat(*args, **kwargs)

    @staticmethod
    def merge(*args, **kwargs):
        return pandas.merge(*args, **kwargs)

    @staticmethod
    def pivot_table(*args, **kwargs):
        return pandas.pivot_table(*args, **kwargs)

    @staticmethod
    def to_datetime(*args, **kwargs):
        return pandas.to_datetime(*args, **kwargs)

pd = pd()

warnings.filterwarnings('ignore')
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
import nltk
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
nltk.download('stopwords')
nltk.download('punkt')
analyzer = SentimentIntensityAnalyzer()
from wordcloud import WordCloud

flipData = pd.read_csv('Dataset-SA.csv')

flipData.rename(columns={
    'product_name': 'Product_name',
    'Rate': 'Rating'
}, inplace=True)
# Ensure Rating is numeric for downstream ML logic.
flipData['Rating'] = pd.to_numeric(flipData['Rating'], errors='coerce')
flipData.dropna(subset=['Rating'], inplace=True)
flipData

#Phase 2: Data Cleaning & Exploratory Data Analysis (EDA)

flipData.drop_duplicates(inplace=True)
flipData.shape
flipData.info()
flipData.isnull().sum()
flipData.describe()
a = flipData['Rating'].value_counts()
a

#Phase 3: Lexicon-Based Sentiment Analysis (VADER)

flipData['Compound_Score'] = flipData['Review'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'])
flipData[['Review', 'Rating', 'Compound_Score']]

def categorize_sentiment(score):
  if score >= 0.05:
   return 'Positive'
  elif score <= -0.05:
    return'Negative'
  else:
   return'Neutral'
flipData['Sentiment'] = flipData['Compound_Score'].apply(categorize_sentiment)
flipData[['Review', 'Rating', 'Compound_Score', 'Sentiment']]

#Phase 4: Data Visualization

plt.figure(figsize=(7, 7))
plt.pie(a,
        labels=a.index,
        autopct='%1.1f%%', # This formats the numbers as percentages
        startangle=140,
        colors=['#66b3ff','#ff9999','#99ff99','#77dd77'])
plt.title('Percentage of Customer Sentiments', fontsize=14)
plt.show()

sentiment_counts = flipData['Sentiment'].value_counts()
plt.figure(figsize=(8, 5))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='Set2')
plt.title('Count of Customer Sentiments', fontsize=14)
plt.xlabel('Sentiment Category', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.show()

plt.figure(figsize=(10, 6))
sns.countplot(data=flipData, x='Rating', hue='Sentiment', palette='Set2')
plt.title('Sentiment Distribution Across Star Ratings', fontsize=14)
plt.xlabel('Star Rating (1 to 5)', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.show()

#Phase 5: Mismatch Analysis (Anomaly Detection)

mismatches = flipData[(flipData['Rating'] == 5)&(flipData['Sentiment'] == 'Negative')]
mismatches[['Review','Rating','Compound_Score']].head()

#Phase 6: Topic Modeling via Word Clouds

positive_reviews = " ".join(review for review in flipData[flipData['Sentiment'] == 'Positive'].Review.astype(str))
wordcloud_pos = WordCloud(width=800, height=400, background_color='white', colormap='Greens').generate(positive_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_pos, interpolation='bilinear')
plt.axis('off') # Hide the X and Y axis
plt.title('Most Common Words in Positive Reviews', fontsize=15)
plt.show()

negative_reviews = " ".join(review for review in flipData[flipData['Sentiment'] == 'Negative'].Review.astype(str))
wordcloud_neg = WordCloud(width=800, height=400, background_color='white', colormap='Reds').generate(negative_reviews)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_neg, interpolation='bilinear')
plt.axis('off')
plt.title('Most Common Words in Negative Reviews', fontsize=15)
plt.show()

#Phase 7: Predictive Machine Learning Pipeline

df_ml = flipData[flipData['Rating'] != 3].copy()
df_ml['Label'] = df_ml['Rating'].apply(lambda x: 1 if x > 3 else 0)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
X = df_ml['Review'].astype(str)
y = df_ml['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred))

#Phase 8: Review Clustering (Clustering)

X_cluster = X_train_tfidf[:500].toarray()
plt.figure(figsize=(12, 7))
plt.title("Dendrogram for Flipkart Reviews")
dendrogram = sch.dendrogram(sch.linkage(X_cluster, method='ward'))
plt.xlabel("Review Data Points")
plt.ylabel("Euclidean Distances")
plt.show()
model_cl = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
clusters = model_cl.fit_predict(X_cluster)
cluster_df = pd.DataFrame({'Review': X_train[:500], 'Cluster': clusters})
print(cluster_df.head(10))

product_stats = flipData.groupby('Product_name').agg({
    'Rating': 'mean',
    'Compound_Score': 'mean'
}).reset_index()
X_prod = product_stats[['Rating', 'Compound_Score']]
hc_prod = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
product_stats['Cluster'] = hc_prod.fit_predict(X_prod)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=product_stats, x='Rating', y='Compound_Score', hue='Cluster', palette='viridis', s=100)
plt.title('Product Clusters based on Rating vs Sentiment')
plt.show()