from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

texts = [
    "I love this product. It's amazing and works perfectly.",
    "This is the best thing I've ever bought.",
    "Absolutely wonderful! Highly recommend it.",
    "Terrible experience. I hate it.",
    "Worst service ever. I'm so disappointed.",
    "It was a waste of money and time."
]
labels = [1, 1, 1, 0, 0, 0]

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression(max_iter=1000))
])

pipeline.fit(texts, labels)
joblib.dump(pipeline, 'text_classifier.pkl')

print("✅ Model trained and saved.")
