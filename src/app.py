import psycopg2
from flask import Flask, render_template, request, jsonify
from naive_review_analyzer import NaiveReviewAnalyzer
from LdaReviewAnalyzer import LdaReviewAnalyzer
from load_review_data import ReviewDataLoader

rdl = ReviewDataLoader()
nra = NaiveReviewAnalyzer()

app = Flask(__name__)
conn = psycopg2.connect(dbname='product_reviews', user='postgres', password='', host='localhost')
cursor = conn.cursor()
sql = "select category, brand_name, model from reviews group by category, brand_name, model"
cursor.execute(sql)
rows = cursor.fetchall()

@app.route('/test', methods=['GET'])
def test():
    return render_template('test.html')

@app.route('/index', methods=['GET'])
def index():
    return render_template('reviews.html', data=rows)

@app.route('/solve', methods=['POST'])
def submit():
    mdl = request.form['model']
    df = rdl.retrieve_reviews(mdl)
    nra.create_bow(df)
    token_dict = nra.create_word_list()
    lda = LdaReviewAnalyzer(df)
    lda.fit_transform(num_topics=3)
    topic_dict = lda.get_topics()
    lda.save_topic_model()
    return render_template('reviews.html', data=rows,mdl=mdl, token_dict=token_dict,topic_dict=topic_dict)

@app.route('/lda', methods=['GET'])
def lda_display():
    return render_template('lda.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
