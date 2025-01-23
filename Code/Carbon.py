import pandas as pd
import numpy as np
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from codecarbon import EmissionsTracker


nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

model = joblib.load("../models/Random Forest_best_model.joblib")
vectorizer = joblib.load("../models/vectorizer.joblib")
label_encoder = joblib.load("../models/label_encoder.joblib")

def Clean_message(text):
    """Nettoyage du texte"""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Supprime les stop words
    text = ' '.join(lemmatizer.lemmatize(word) for word in text.split())  # Lemmatization
    return text

def predict_message(message):
    treated_message = vectorizer.transform([Clean_message(message)])
    y_pred = model.predict(treated_message)
    label = label_encoder.inverse_transform(y_pred)[0]
    return label

message = "You have won an Iphone 6 ! to claim your reward, please click on the following link : http://www.notphishingtrust.com"

tracker = EmissionsTracker()
tracker.start()
y_pred = predict_message(message)
tracker.stop()

print(f"Prédiction : {y_pred}")
