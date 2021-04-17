import pandas as pd
from flask import Flask, request, render_template

import model
import pickle

# Utility

# nltk

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', users=model.df.sample().reviews_username)


@app.route('/recommend_products', methods=['POST'])
def recommend_products():
    username = request.form['username']
    try:
        recommended_product = model.user_final_rating.loc[username].sort_values(ascending=False)[0:20].reset_index()
        recommended_product['positive_review_count'] = recommended_product.product_name.apply(
            lambda review_text: model.get_number_of_positive_sentiment(review_text))
        recommended_product['review_count'] = recommended_product.product_name.apply(
            lambda review_text: model.get_number_of_reviews(review_text))
        recommended_product['positive_review_percentage'] = \
            recommended_product['positive_review_count'] / recommended_product['review_count']
        product_data = recommended_product.sort_values(by='positive_review_percentage', ascending=False)[0:5]
    except KeyError:
        return render_template('index.html', error="unknown user", users=model.df.sample().reviews_username)

    return render_template('index.html', products=product_data.product_name, users=model.df.sample().reviews_username)


if __name__ == '__main__':
    app.run(debug=True)
