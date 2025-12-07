from flask import Flask, render_template, request
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

application = Flask(__name__)
app = application

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

port_stem = PorterStemmer()
stop_words = set(stopwords.words("english"))
pattern = re.compile(r"[^a-zA-Z]")

def stemming(content):
    content = pattern.sub(" ", content)
    content = content.lower()
    words = content.split()
    words = [port_stem.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    tweet_text = ""

    if request.method == "POST":
        tweet_text = request.form.get("tweet", "")

        # 1. preprocess
        processed = stemming(tweet_text)

        # 2. vectorize
        vec = vectorizer.transform([processed])

        # 3. predict
        pred = model.predict(vec)[0]   # 0 or 1

        # 4. map to label
        sentiment = "Positive Tweet" if pred == 1 else "Negative Tweet"

        prediction = f"Predicted sentiment: {sentiment}"

    return render_template("home.html", prediction=prediction, tweet=tweet_text)


if __name__ == "__main__":
    app.run(debug=True)