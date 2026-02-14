import joblib
from scipy.sparse import hstack

def predict_job(text, numeric_features):

    model = joblib.load("../models/xgboost_model.pkl")
    tfidf = joblib.load("../models/tfidf_vectorizer.pkl")

    text_vector = tfidf.transform([text])

    final_input = hstack([text_vector, numeric_features])

    prediction = model.predict(final_input)
    probability = model.predict_proba(final_input)[0][1]

    return prediction, probability
