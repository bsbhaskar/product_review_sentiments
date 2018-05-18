import psycopg2
from flask import Flask, render_template, request, jsonify
from naive_review_analyzer import NaiveReviewAnalyzer
from LdaReviewAnalyzer import LdaReviewAnalyzer
from load_review_data import ReviewDataLoader

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
    token_dict_neg = nra.create_word_list(rating=[1,2])
    token_dict_pos = nra.create_word_list(rating=[4,5])

    lda.transform(mdl, rating=[1,2])
    topic_dict_neg = lda.get_topics()
    lda.save_topic_model('templates/lda_neg.html')

    lda.transform(mdl, rating=[4,5])
    topic_dict_pos = lda.get_topics()
    lda.save_topic_model('templates/lda_pos.html')

    return render_template('reviews.html', data=rows,mdl=mdl, topic_dict_neg=topic_dict_neg, topic_dict_pos=topic_dict_pos, token_dict_neg=token_dict_neg, token_dict_pos=token_dict_pos, total_count=total_count, neg_count=neg_count, pos_count=pos_count)

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
    app.run(host='0.0.0.0', threaded=True)
