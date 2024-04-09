from joblib import load

lr_path = "./lr_classifier.joblib"
vectorizer_path = "./tfidf_vectorizer.joblib"

vectorizer=""
lr = ""

def init():
    global vectorizer, lr
    vectorizer = load(vectorizer_path)
    lr = load(lr_path)

def predict(song_text):
    global vectorizer, lr
    X = [song_text]
    X = vectorizer.transform(X)
    ans = lr.predict(X)
    return ans

