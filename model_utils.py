import pickle
import re

# --- Load your trained model and vectorizer ---
with open("mental_health_model.pkl", "rb") as f:
    clf = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

def predict_emotion(user_text):
    text = clean_text(user_text)
    vec = vectorizer.transform([text])
    pred = clf.predict(vec)[0]
    return pred
