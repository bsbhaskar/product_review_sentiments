import psycopg2
from flask import Flask, render_template, request, jsonify
from naive_review_analyzer import NaiveReviewAnalyzer
from LdaReviewAnalyzer import LdaReviewAnalyzer
from load_review_data import ReviewDataLoader
from random import random
import matplotlib.pyplot as plt
from io import BytesIO

rdl = ReviewDataLoader()
nra = NaiveReviewAnalyzer()
lda = LdaReviewAnalyzer(num_topics=5)
df_all = rdl.retrieve_all_reviews()
lda.build_vectorize(df_all)
lda.fit(df_all, random_state=22)

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

@app.route('/plot.png', methods=['GET'])
def get_graph():
    token_dict_pos = request.args['token_dict_pos']
    print (token_dict_pos)
    print (type(token_dict_pos))
    token_dict_pos = token_dict_pos[2,-3]
    list = [token for token in token_dict_pos.split('), (')]
    print ('list:',list)

    # plt.figure()
    # image = BytesIO()
    #
    # objects = [x[0] for x in token_dict_pos]
    # print ('objects:',objects)
    # y_pos = range(len(objects))
    # performance = [x[1] for x in token_dict_pos]
    # print ('performance:',performance)

    # fig, ax = plt.subplots(1,1, figsize=(10,15))
    # ax.barh(y_pos, performance, align='center', alpha=0.5)
    # plt.yticks(y_pos, objects, fontsize=20)
    # ax.set_xlabel('Relative Probability', fontsize=20)
    # ax.set_title('Positive Keywords')
    # plt.savefig(image)

    # plt.figure()
    # n = 10
    # plt.plot(range(n), [random() for i in range(n)])
    # image = BytesIO()
    # plt.savefig(image)
    return image.getvalue(), 200, {'Content-Type': 'image/png'}

@app.route('/solve', methods=['POST'])
def submit():
    mdl = request.form['model']
    df = rdl.retrieve_reviews(mdl)
    total_count = df['rating'].count()
    df_neg = df[df['rating'].apply(lambda x: x in [1,2])]
    neg_count = df_neg['rating'].count()
    df_pos = df[df['rating'].apply(lambda x: x in [4,5])]
    pos_count = df_pos['rating'].count()
    nra.create_bow(df)
    token_dict_neg, sent_neg = nra.create_word_list(rating=[1,2])
    token_dict_pos, sent_pos = nra.create_word_list(rating=[4,5])

    lda.transform(mdl, rating=[1,2])
    topic_dict_neg = lda.get_topics()
    lda.save_topic_model('templates/lda_neg.html')

    lda.transform(mdl, rating=[4,5])
    topic_dict_pos = lda.get_topics()
    lda.save_topic_model('templates/lda_pos.html')
    print ('data:',rows)
    print ('----------------------------------')
    print ('mld:',mdl)
    print ('----------------------------------')
    print ('topic_dict_pos:',topic_dict_pos)
    print ('----------------------------------')
    print ('token_dict_pos:',token_dict_pos)
    print ('----------------------------------')
    print ('topic_dict_neg:',topic_dict_neg)
    print ('----------------------------------')
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

    fig, ax = plt.subplots(1,1, figsize=(5,8))
    ax.barh(y_pos, performance, align='center', alpha=0.5)
    plt.yticks(y_pos, objects, fontsize=15)
    plt.xticks(fontsize=15)
    ax.set_xlabel('Relative Probability', fontsize=20)
    ax.set_title('Positive Keywords', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'static/images/plot_pos_{mdl}.png')

    objects = [x[0] for i,x in enumerate(token_dict_neg) if i < max_count]
    y_neg = list(range(len(objects)-1,-1,-1))
    #y_neg = list(range(len(objects)))
    performance = [x[1] for i,x in enumerate(token_dict_neg) if i < max_count]

    fig, ax = plt.subplots(1,1, figsize=(5,8))
    ax.barh(y_neg, performance, align='center', alpha=0.5)
    plt.yticks(y_neg, objects, fontsize=15)
    plt.xticks(fontsize=15)
    ax.set_xlabel('Relative Probability', fontsize=20)
    ax.set_title('Negative Keywords', fontsize=20)
    plt.tight_layout()
    plt.savefig(f'static/images/plot_neg_{mdl}.png')
    plot_neg_img = f'<img src="static/images/plot_neg_{mdl}.png" />'
    plot_pos_img = f'<img src="static/images/plot_pos_{mdl}.png" />'

    #return render_template('reviews.html')
    return render_template('reviews.html',
                           data=rows,
                           mdl=mdl,
                           topic_dict_neg=topic_dict_neg,
                           topic_dict_pos=topic_dict_pos,
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

@app.route('/lda_pos', methods=['GET'])
def lda_display_pos():
    return render_template('lda_pos.html')

@app.route('/lda_neg', methods=['GET'])
def lda_display_neg():
    return render_template('lda_neg.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
