import feedparser
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
import nltk
nltk.download('popular')
NewsFeed = feedparser.parse(
    "https://www.newindianexpress.com/World/rssfeed/?id=171&getXmlFeed=true")
app = Flask(__name__)
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
vocab = pickle.load(open('vocab.pickle', 'rb'))
idf = pickle.load(open('idf.pickle', 'rb'))
logreg = pickle.load(open('logreg.pickle', 'rb'))
classifiedlist = []

def extract_words(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    filtered_sentence = []
    for w in word_tokens:
        if w not in stop_words:
            stemmed = ps.stem(w)
            lemmed = lemmatizer.lemmatize(stemmed)
            filtered_sentence.append(lemmed)
    return filtered_sentence


def bagofwords(sentence, words):
    sentence_words = extract_words(sentence)
    bag = np.zeros(len(words))
    for sw in sentence_words:
        for i, word in enumerate(words):
            if word == sw:
                bag[i] += 1
    return np.array(bag)


def predictor(headline):
    word_vector = bagofwords(headline, vocab)
    word_tfidf = word_vector * idf
    prediction = logreg.predict(word_tfidf.reshape(1, -1))
    results = {1: 'Relevant', 0: 'Not Relevant'}
    return results[int(prediction)]


for item in NewsFeed.entries:
    classifiedlist.append({
        'title': item.title,
        'prediction': predictor(item.title)
    })


@app.route('/')
def home():
    return render_template('index.html', feed=classifiedlist)


@app.route('/classify-a-headline')
def classify_a_headline():
    return render_template('classify_input.html')


@app.route('/classify', methods=['GET', 'POST'])
def classify():
    if request.method == 'POST':
        if request.form['textArea'] == '':
            return redirect(url_for('classify_a_headline'))
        else:
            return render_template('result.html', payload={
                'prediction': predictor(request.form['textArea']),
                'headline': request.form['textArea']
            })
    else:
        return redirect(url_for('classify_a_headline'))

if __name__ == '__main__':
    app.run()