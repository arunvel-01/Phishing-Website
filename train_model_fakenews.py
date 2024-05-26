import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier

# Read the data
df = pd.read_csv('c.csv')
df.columns = ['url', 'status']

labels = df.status  # Assuming 'status' is the column containing labels
X_train, X_test, y_train, y_test = train_test_split(df['url'], labels, test_size=0.2)  # Adjust test_size as per your requirement

tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)

# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)

# Saving vectorizer
with open('model_fakenews1.pickle', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)

# Saving model
with open('tfid1.pickle', 'wb') as f:
    pickle.dump(pac, f)
