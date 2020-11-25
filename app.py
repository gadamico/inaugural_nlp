import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from flask import Flask, request, render_template, jsonify


with open('voting_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('countvec.pkl', 'rb') as g:
    cvec = pickle.load(g)

app = Flask(__name__, static_url_path="")

@app.route('/', methods=['GET', 'POST'])
def index():
    """Return the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Return a random prediction."""
    data = str(request.get_json())
    stem = nltk.stem.SnowballStemmer('english')
    stops = nltk.corpus.stopwords.words('english')
    stops.extend(list(string.punctuation))
    stops.extend(list(string.digits))
    
    cleaned = []
    print(data)
    for word in data.split():
        if word not in stops and not word.isnumeric():
            cleaned.append(word)
    speech_cleaned = [stem.stem(word.lower().replace('\'', ''))
                     for word in nltk.word_tokenize(' '.join(cleaned))]
    vect = cvec.transform([' '.join(speech_cleaned)])
    
    prediction = model.predict(vect)
    
    if prediction == 'R':
        return "That was written by a Republican!"
    if prediction == 'D':
        return "That was written by a Democrat!"
    if prediction == 'W':
        return "That was written by a Whig!"
    if prediction == 'DR':
        return "That was written by a Democratic-Republican!"
    else:
        return "That was written by a Whig!"