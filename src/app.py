'''
app.py contains code necessary to run the webapp including
Results from Naive Bayes Model and Word2Vec
Currently, the topic model is loaded from a static file
generated earlier.

currently app.py uses three templates - review.html, w2v.html and lda.html
'''
import psycopg2
from flask import Flask, render_template, request
from naive_review_analyzer import NaiveReviewAnalyzer
from load_review_data import ReviewDataLoader
from trigrams import Trigrams
from w2v_review_analyzer import W2VReviewAnalyzer
from random import random
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd
import pickle

rdl = ReviewDataLoader()
nra = NaiveReviewAnalyzer()
tr = Trigrams()
tr.load_model()

'''
Steps below are to load and cache all of the data
necessary to run the app fast including loading
product data, model etc.
Following steps:
    1) retrieves all products from the database
    2) Loads model
'''
#df_all = rdl.retrieve_all_reviews()
#tr.build_trigrams(df_all)
w2v = W2VReviewAnalyzer(pd.DataFrame())
w2v.model = pickle.load(open('../static/w2v.pkl','rb'))

app = Flask(__name__)

#Following code load data needed for product dropdown
conn = psycopg2.connect(dbname='product_reviews', user='postgres', password='', host='localhost')
cursor = conn.cursor()
sql = "select category, brand_name, model from reviews group by category, brand_name, model"
cursor.execute(sql)
rows = cursor.fetchall()

@app.route('/', methods=['GET'])
def index():
    '''
    renders the initial template filled with data for pull-dropdown
    '''
    return render_template('reviews.html', data=rows)

@app.route('/w2v', methods=['GET'])
def similar_words_index():
    return render_template('w2v.html')

@app.route('/w2v_results', methods=['POST'])
def similar_words_results():
    '''
    parse input keywords and display results from word2vec model
    '''
    plus_words = ''
    minus_words = ''
    plus = request.form['plus']
    if (len(plus.strip()) > 0):
        plus_words = [x.strip() for x in plus.split(",")]
    minus = request.form['minus']
    if (len(minus.strip()) > 0):
        minus_words = [x.strip() for x in minus.split(",")]
    sim_words = w2v.model.wv.most_similar(positive=plus_words, negative=minus_words, topn=15)
    sim_words = [(x[0],round(float(x[1]),2)) for x in sim_words]
    return render_template('w2v.html', plus=plus, minus=minus, sim_words=sim_words)

@app.route('/solve', methods=['POST'])
def submit():
    '''
    parse model and aggregate information such as no of positive
    and negative reviews, and relative probabilities from Naive Bayes model
    '''
    mdl = request.form['model']
    df_mdl = rdl.retrieve_reviews(mdl)
    df = tr.transform(df_mdl)
    #df = tr.df_prod[tr.df_prod['model'] == mdl]
    total_count = df['rating'].count()
    df_neg = df[df['rating'].apply(lambda x: x in [1,2])]
    neg_count = df_neg['rating'].count()
    df_pos = df[df['rating'].apply(lambda x: x in [4,5])]
    pos_count = df_pos['rating'].count()
    nra.create_bow(df)
    token_dict_neg, sent_neg = nra.create_word_list(rating=[1,2])
    token_dict_pos, sent_pos = nra.create_word_list(rating=[4,5])

    #Build features from positive and negative reviews
    max_count = 10

    objects = [x[0] for i,x in enumerate(token_dict_pos) if i < max_count]
    y_pos = list(range(len(objects)-1,-1,-1))
    performance = [x[1] for i,x in enumerate(token_dict_pos) if i < max_count]
    print ('objects:',objects)
    print ('performance:',performance)

    fig, ax = plt.subplots(1,1, figsize=(12,8))
    ax.barh(y_pos, performance, align='center', alpha=0.5, color='teal')
    plt.yticks(y_pos, objects, fontsize=30)
    plt.xticks(fontsize=30)
    ax.set_xlabel('Relative Probability', fontsize=36)
    ax.set_title('Positive Product Features', fontsize=36)
    ax.set_xlim(0,100)
    plt.tight_layout()
    plt.savefig(f'static/images/plot_pos_{mdl}.png')

    objects = [x[0] for i,x in enumerate(token_dict_neg) if i < max_count]
    y_neg = list(range(len(objects)-1,-1,-1))
    performance = [x[1] for i,x in enumerate(token_dict_neg) if i < max_count]

    fig, ax = plt.subplots(1,1, figsize=(12,8))
    ax.barh(y_neg, performance, align='center', alpha=0.5, color='firebrick')
    plt.yticks(y_neg, objects, fontsize=30)
    plt.xticks(fontsize=30)
    ax.set_xlabel('Relative Probability', fontsize=36)
    ax.set_title('Negative Product Features', fontsize=36)
    ax.set_xlim(0,100)
    plt.tight_layout()
    plt.savefig(f'static/images/plot_neg_{mdl}.png')
    plot_neg_img = f'<img width="500" height="400" src="static/images/plot_neg_{mdl}.png" />'
    plot_pos_img = f'<img  width="500" height="400" src="static/images/plot_pos_{mdl}.png" />'

    return render_template('reviews.html',
                           data=rows,
                           mdl=mdl,
                           token_dict_neg=token_dict_neg,
                           token_dict_pos=token_dict_pos,
                           sent_neg=sent_neg,
                           sent_pos=sent_pos,
                           total_count=total_count,
                           neg_count=neg_count,
                           pos_count=pos_count,
                           plot_neg_img=plot_neg_img,
                           plot_pos_img=plot_pos_img)

@app.route('/lda', methods=['GET'])
def lda_display():
    return render_template('lda.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
