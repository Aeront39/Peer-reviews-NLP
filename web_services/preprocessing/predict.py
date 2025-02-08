# python and keras imports
import numpy as np
from keras.preprocessing import sequence, text
from preprocessing.suggestions_and_problem_preprocessing import load_items, predict_class

import os

from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')  # For sentence tokenization

# Import nltk libraries for volume analysis
from nltk.tokenize import TreebankWordTokenizer

treebank_tokenizer = TreebankWordTokenizer()
from nltk.corpus import stopwords

# set rules for stopwords ignored in volume analysis
stop_words = set(stopwords.words('english'))
stop_words |= {'.', ',', '!', '?'}


# predict sentiment tone and score of the review
def predictSentiment(review):
    blob = TextBlob(review)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Check sentence-level sentiments for Mixed tone
    positive_sentences = negative_sentences = 0
    for sentence in blob.sentences:
        sent_polarity = sentence.sentiment.polarity
        if sent_polarity > 0.1:
            positive_sentences += 1
        elif sent_polarity < -0.1:
            negative_sentences += 1

    predicted_confidence = 0
    sentiment_tone = "Neutral"

    if polarity > 0.25:
        sentiment_tone = "Positive"
        predicted_confidence = 1
    elif polarity < -0.25:
        sentiment_tone = "Negative"
    else:
        if positive_sentences > 0 and negative_sentences > 0:
            sentiment_tone = "Mixed"
        else:
            sentiment_tone = "Neutral" if subjectivity < 0.6 else "Mixed"

    return sentiment_tone, round(polarity, 3), predicted_confidence


# predict presence and chances of suggestions in the review
def predictSuggestions(review):
    model = os.path.abspath('.')  # path to locate the saved Machine Learning models
    suggestion_model = model + "/model/suggestions_cnn_model.h5"
    suggestions_tokenizer = model + "/model/suggestions_tokenizer"
    model, tokenizer = load_items(suggestion_model, suggestions_tokenizer)
    predicted_comment, predicted_confidence = predict_class(review, model, tokenizer, 200)

    if predicted_comment == 1:
        suggestion = "Present"
    else:
        suggestion = "Absent"
        predicted_confidence = 1 - predicted_confidence
    return suggestion, predicted_confidence


# predict volume metrics of the review
def predictVolume(review):
    # tokenize the review using NLTK tokenizer
    tokens = treebank_tokenizer.tokenize(review)
    total_volume = len(tokens)
    # remove all the stopwords
    non_stop_words = [word for word in tokens if word not in stop_words]
    volume_without_stopwords = len(non_stop_words)
    return (total_volume, volume_without_stopwords)


# predict presence of praise and or criticism in the review
def predictEmotion(review):
    sid = SentimentIntensityAnalyzer()
    scores = sid.polarity_scores(review)
    compound = scores['compound']
    pos = scores['pos']
    neg = scores['neg']
    magnitude = pos + neg  # Approximate "magnitude"

    praise = criticism = "None"
    predicted_confidence = 0

    # Adjust thresholds based on your use case
    if 0.3 <= magnitude < 0.6:
        if compound > 0.1:
            praise = "Low"
        elif compound < -0.1:
            criticism = "Low"
        else:
            praise = criticism = "Low"
    elif magnitude >= 0.6:
        if compound > 0.1:
            praise = "High"
            predicted_confidence = 1
        elif compound < -0.1:
            criticism = "High"
        else:
            praise = criticism = "High"
            predicted_confidence = 1

    return praise, criticism, predicted_confidence


# predict presence of problem in the review

def predictProblem(review):
    model = os.path.abspath('.')  # path to locate the saved Machine Learning models
    problems_model = model + "/model/problems_cnn_model.h5"
    problems_tokenizer = model + "/model/problems_tokenizer"
    model, tokenizer = load_items(problems_model, problems_tokenizer)
    predicted_comment, predicted_confidence = predict_class(review, model, tokenizer, 400)
    problem = "None"
    if predicted_comment == 1:
        problem = "Present"
    else:
        problem = "Absent"
        predicted_confidence = 1 - predicted_confidence
    return problem, predicted_confidence
