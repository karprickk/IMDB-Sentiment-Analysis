from flask import Flask, request, jsonify, render_template, redirect, url_for
import mysql.connector
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import os

# Ensure nltk stopwords are downloaded
nltk.download('stopwords')

# Load trained model and tokenizer
MODEL_PATH = r"C:\IMDB Sentiment Analysis\sentiment_classifier2.h5"
TOKENIZER_PATH = r"C:\IMDB Sentiment Analysis\tokenizer.pkl"

model = tf.keras.models.load_model(MODEL_PATH)
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Stopwords and regex for cleaning text
stopwords_list = set(stopwords.words('english'))
TAG_RE = re.compile(r'<[^>]+>')

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Karthik@1234",
    database="sqlinj"
)
cursor = db.cursor()

app = Flask(__name__)

# Function to clean text before prediction
def clean_text(text):
    text = TAG_RE.sub('', text)  # Remove HTML tags
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stopwords_list])  # Remove stopwords
    return text

# Function to predict sentiment using the trained model
def predict_sentiment(text):
    cleaned_text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=100)  # Ensure this matches the training maxlen
    prediction = model.predict(padded_sequence)[0][0]

    if prediction > 0.6:
        return "positive"
    elif prediction < 0.4:
        return "negative"
    else:
        return "neutral"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/add_review', methods=['POST'])
def add_review():
    try:
        data = request.get_json()
        print("Received JSON:", data)  # Debugging

        if not data:
            return jsonify({"error": "No data received"}), 400

        movie_name = data.get('movie_name')
        review_text = data.get('review')

        if not movie_name or not review_text:
            return jsonify({"error": "Missing movie_name or review"}), 400

        sentiment = predict_sentiment(review_text)  # Ensure this function exists

        cursor.execute("INSERT INTO reviews (movie_name, review, sentiment) VALUES (%s, %s, %s)",
                       (movie_name, review_text, sentiment))
        db.commit()

        return jsonify({"message": "Review added successfully!", "sentiment": sentiment})

    except Exception as e:
        print("Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route('/get_reviews', methods=['GET'])
def get_reviews():
    movie_name = request.args.get('movie_name')
    cursor.execute("SELECT review, sentiment FROM reviews WHERE movie_name = %s", (movie_name,))
    reviews = cursor.fetchall()

    categorized_reviews = {"positive": [], "neutral": [], "negative": []}
    for review, sentiment in reviews:
        categorized_reviews[sentiment].append(review)

    return jsonify(categorized_reviews)

if __name__ == '__main__':
    app.run(debug=True)
