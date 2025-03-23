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
import subprocess


# Initialize Flask app
app = Flask(__name__)

# Load trained model and tokenizer
MODEL_PATH = os.path.join(os.getcwd(), "sentiment_classifier2.h5")
TOKENIZER_PATH = os.path.join(os.getcwd(), "tokenizer.pkl")

model = tf.keras.models.load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# Ensure NLTK stopwords are downloaded
nltk.download("stopwords")
stopwords_list = set(stopwords.words("english"))

TAG_RE = re.compile(r"<[^>]+>")


def clean_text(text):
    """Preprocess the text before feeding it to the model."""
    text = TAG_RE.sub("", text)  # Remove HTML tags
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = " ".join([word for word in text.split() if word not in stopwords_list])
    return text


def predict_sentiment(review_text):
    """Predict sentiment using the trained model."""
    cleaned_review = clean_text(review_text)
    sequence = tokenizer.texts_to_sequences([cleaned_review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)[0][0]

    if prediction >= 0.6:
        return "positive"
    elif prediction >= 0.4:
        return "neutral"
    else:
        return "negative"


# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Karthik@1234",
    database="sqlinj"
)
cursor = db.cursor()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/sentiment")
def sentiment_analysis():
    return render_template("sentiment_analysis.html")

@app.route('/recommend')
def recommend():
    subprocess.Popen(["streamlit", "run", "movie_recommender.py"], shell=True)
    return redirect("http://localhost:8501")  # Streamlit default port

@app.route("/add_review", methods=["POST"])
def add_review():
    """Handles review submission and stores sentiment in the database."""
    data = request.get_json()  # Handles JSON request
    if not data or "movie_name" not in data or "review" not in data:
        return jsonify({"error": "Invalid request data"}), 400

    movie_name = data["movie_name"]
    review_text = data["review"]
    sentiment = predict_sentiment(review_text)

    cursor.execute(
        "INSERT INTO reviews (movie_name, review, sentiment) VALUES (%s, %s, %s)",
        (movie_name, review_text, sentiment),
    )
    db.commit()

    return jsonify({"message": "Review added successfully", "sentiment": sentiment})


@app.route("/get_reviews", methods=["GET"])
def get_reviews():
    """Fetch and categorize reviews for a specific movie."""
    movie_name = request.args.get("movie_name")
    if not movie_name:
        return jsonify({"error": "Movie name is required"}), 400

    cursor.execute("SELECT review, sentiment FROM reviews WHERE movie_name = %s", (movie_name,))
    reviews = cursor.fetchall()

    categorized_reviews = {"positive": [], "neutral": [], "negative": []}
    for review, sentiment in reviews:
        categorized_reviews[sentiment].append(review)

    return jsonify(categorized_reviews)


if __name__ == "__main__":
    app.run(debug=True)
