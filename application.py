from flask import Flask, request
import joblib
application = Flask(__name__)

Vectorizer = joblib.load("model/vectorizer.pkl")
spamorham_model = joblib.load("model/spam_ham_model.pkl")

@application.route('/')
def hello_world():
    return "Welcome"

@application.route('/spamorham',methods=['GET','POST'])
def spamorham():
    message = request.args.get("message")
    vect_message = Vectorizer.transform([message])
    result = spamorham_model.predict(vect_message)[0]
    return result

if __name__ == '__main__':
    application.run()