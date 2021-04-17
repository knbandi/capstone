# -*- coding: utf-8 -*-

# Utility
import re

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import pickle
import pandas as pd

nltk.download('stopwords')

stop_words = stopwords.words("english")

stemmer = SnowballStemmer("english")

# TEXT CLENAING
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"

# DATASET
TRAIN_SIZE = 0.8

# WORD2VEC 
W2V_SIZE = 300
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

df = pd.read_pickle('pickle/df.pkl')
user_final_rating = pd.read_pickle('pickle/user_final_rating.pkl')
model_lr = pickle.load(open('pickle/model_lr.pkl', 'rb'))
tfidf = pickle.load(open('pickle/tfidf.pkl', 'rb'))

def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(TEXT_CLEANING_RE, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

def get_number_of_positive_sentiment(review_text):
    products_data = df.loc[df['name'] == review_text, ['name', 'reviews_text']]
    products_data['reviews_text'] = products_data.reviews_text.apply(lambda x: preprocess(x))
    products_data['reviews_text'] = products_data.reviews_text.apply(lambda x: preprocess(x, True))
    data = tfidf.transform(products_data.reviews_text)
    return model_lr.predict(data).sum()


def get_number_of_reviews(review_text):
    products_data = df.loc[df['name'] == review_text, ['name', 'reviews_text']]
    return len(products_data)
