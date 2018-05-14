from collections import Counter
from flask import Flask, request
import pickle
import pandas as pd
app = Flask(__name__)

@app.route('/')
def welcome_page():
    return '''
        <H1>Welcome to the Consumer Review Insights, click on the link below to classify your text: </H1>
            <A href="/submit">Click Here</A>
        '''

# Form page to submit text
@app.route('/submit')
def submission_page():
    return '''
        <H1>Enter Text to Classify Here</H1>
        <form action="/predict" method='POST' >
            <input type="text" name="user_input" />
            <input type="submit" />
        </form>
        '''

# My text classifier app
@app.route('/predict', methods=['POST'] )
def text_classifier():

    text = [str(request.form['user_input'])]
    with open('data/model.pkl', 'rb') as f:
        model = pickle.load(f)
    prediction =  model.predict(text)

    page = f'This text belongs to <br> {prediction}'
    return page

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
