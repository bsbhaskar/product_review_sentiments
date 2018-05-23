import psycopg2
from flask import Flask, render_template, request, jsonify
from naive_review_analyzer import NaiveReviewAnalyzer
from LdaReviewAnalyzer import LdaReviewAnalyzer
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
#lda = LdaReviewAnalyzer(num_topics=5)
tr = Trigrams()
print ('Initialized')
df_all = rdl.retrieve_all_reviews()
print ('Loaded dataframe')
tr.build_trigrams2(df_all)
print ('built trigrams')
# lda.build_vectorize(tr.df_prod)
# lda.fit(df_all, random_state=22)
w2v = W2VReviewAnalyzer(pd.DataFrame())
w2v.model = pickle.load(open('../static/w2v.pkl','rb'))
print ('loaded w2v model')
app = Flask(__name__)
conn = psycopg2.connect(dbname='product_reviews', user='postgres', password='', host='localhost')
cursor = conn.cursor()
sql = "select category, brand_name, model from reviews group by category, brand_name, model"
cursor.execute(sql)
rows = cursor.fetchall()

@app.route('/test', methods=['GET'])
def test():
    return render_template('test.html')

@app.route('/', methods=['GET'])
def index():
    return render_template('reviews.html', data=rows)

@app.route('/w2v', methods=['GET'])
def similar_words_index():
    return render_template('w2v.html')

@app.route('/w2v_results', methods=['POST'])
def similar_words_results():
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
    mdl = request.form['model']
    #df = rdl.retrieve_reviews(mdl)
    df = tr.df_prod[tr.df_prod['model'] == mdl]
    total_count = df['rating'].count()
    df_neg = df[df['rating'].apply(lambda x: x in [1,2])]
    neg_count = df_neg['rating'].count()
    df_pos = df[df['rating'].apply(lambda x: x in [4,5])]
    pos_count = df_pos['rating'].count()
    nra.create_bow(df)
    token_dict_neg, sent_neg = nra.create_word_list(rating=[1,2])
    token_dict_pos, sent_pos = nra.create_word_list(rating=[4,5])

    # lda.transform(mdl, rating=[1,2])
    # topic_dict_neg = lda.get_topics()
    # lda.save_topic_model('templates/lda_neg.html')
    #
    # lda.transform(mdl, rating=[4,5])
    # topic_dict_pos = lda.get_topics()
    # lda.save_topic_model('templates/lda_pos.html')
    print ('data:',rows)
    print ('----------------------------------')
    print ('mld:',mdl)
    print ('----------------------------------')
    # print ('topic_dict_pos:',topic_dict_pos)
    # print ('----------------------------------')
    print ('token_dict_pos:',token_dict_pos)
    print ('----------------------------------')
    # print ('topic_dict_neg:',topic_dict_neg)
    # print ('----------------------------------')
    print ('token_dict_neg:',token_dict_neg)
    print ('----------------------------------')
    print ('Count:',total_count, neg_count, pos_count)

    max_count = 10

    objects = [x[0] for i,x in enumerate(token_dict_pos) if i < max_count]
    y_pos = list(range(len(objects)-1,-1,-1))
    #y_pos = list(range(len(objects)))
    performance = [x[1] for i,x in enumerate(token_dict_pos) if i < max_count]
    print ('objects:',objects)
    print ('performance:',performance)

    fig, ax = plt.subplots(1,1, figsize=(12,8))
    ax.barh(y_pos, performance, align='center', alpha=0.5, color='teal')
    plt.yticks(y_pos, objects, fontsize=30)
    plt.xticks(fontsize=30)
    ax.set_xlabel('Relative Probability', fontsize=36)
    ax.set_title('Positive Keywords', fontsize=36)
    ax.set_xlim(0,100)
    plt.tight_layout()
    plt.savefig(f'static/images/plot_pos_{mdl}.png')

    objects = [x[0] for i,x in enumerate(token_dict_neg) if i < max_count]
    y_neg = list(range(len(objects)-1,-1,-1))
    #y_neg = list(range(len(objects)))
    performance = [x[1] for i,x in enumerate(token_dict_neg) if i < max_count]

    fig, ax = plt.subplots(1,1, figsize=(12,8))
    ax.barh(y_neg, performance, align='center', alpha=0.5, color='firebrick')
    plt.yticks(y_neg, objects, fontsize=30)
    plt.xticks(fontsize=30)
    ax.set_xlabel('Relative Probability', fontsize=36)
    ax.set_title('Negative Keywords', fontsize=36)
    ax.set_xlim(0,100)
    plt.tight_layout()
    plt.savefig(f'static/images/plot_neg_{mdl}.png')
    plot_neg_img = f'<img width="500" height="400" src="static/images/plot_neg_{mdl}.png" />'
    plot_pos_img = f'<img  width="500" height="400" src="static/images/plot_pos_{mdl}.png" />'

    #return render_template('reviews.html')
    return render_template('reviews.html',
                           data=rows,
                           mdl=mdl,
                           # topic_dict_neg=topic_dict_neg,
                           # topic_dict_pos=topic_dict_pos,
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
