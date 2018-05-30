import sys
import os
import pandas as pd
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
from nltk.corpus import stopwords
import unicodedata
import string
import numpy as np
import spacy
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import ipywidgets
import pyLDAvis.sklearn
import pickle

class LdaReviewAnalyzer():

    """
    This class builds classifies documents and words into topics using
    Latent Dirichlet Allocation algorithm from sklearn. It is important to keep the random state constant for get reproducible results
    pyLDAvis library is used to visualize the topics
    """

    def __init__(self, num_topics=5, wp):
        '''
        df is dataframe with atleaast one column named reviews which is the corpus used for analyzing topics. The dataframe may contain products and ratings if necessary for segmenting the topics
        wp is the WordProcessor class from word_processing.py
        '''
        self.num_topics = num_topics
        self.wp = wp


    def build_vectorize(self, df_prod):

        self.vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                             stop_words=self.wp.stop_words, lowercase=True,
                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
        self.vectorizer.fit(df_prod['reviews'])

    def vectorize(self, product='all', rating=[1,2,3,4,5]):

        if (product == 'all'):
            df_prod = self.df_prod[self.df_prod['rating'].apply(lambda x: x in rating)]
        else:
            df_prod = self.df_prod[self.df_prod['model'] == product]
            df_prod = df_prod[df_prod['rating'].apply(lambda x: x in rating)]

        data_vectorized = self.vectorizer.transform(df_prod['reviews'])
        return data_vectorized

    def fit(self, df, product='all',rating=[1,2,3,4,5], random_state=40):

        self.df_prod = df
        data_vectorized = self.vectorize(product,rating)
        self.lda_model = LatentDirichletAllocation(n_topics=self.num_topics, max_iter=10, learning_method='online', random_state=random_state)
        self.lda_model.fit(data_vectorized)

    def transform(self, product='all',rating=[1,2,3,4,5]):

        self.data_vectorized = self.vectorize(product,rating)
        self.lda_results = self.lda_model.transform(self.data_vectorized)
        return self.lda_results

    def get_topics(self, top_n=10):
        tpc_dict = {}
        for idx, topic in enumerate(self.lda_model.components_):
            tpc = ("Topic %d:" % (idx))
            tpc_list = [(self.vectorizer.get_feature_names()[i], topic[i])
                            for i in topic.argsort()[:-top_n - 1:-1]]
            tpc_dict[tpc] = tpc_list
        return tpc_dict

    def save_topic_model(self, loc='templates/lda.html'):

        #pyLDAvis.enable_notebook()
        panel = pyLDAvis.sklearn.prepare(self.lda_model, self.data_vectorized, self.vectorizer, mds='tsne', lambda_step=0.5, R=15)
        pyLDAvis.save_html(panel,loc)

    def display_topic_model(self):

        pyLDAvis.enable_notebook()
        panel = pyLDAvis.sklearn.prepare(self.lda_model, self.data_vectorized, self.vectorizer, mds='tsne', lambda_step=0.5, R=15)
        return pyLDAvis.display(panel)


    def save_lda_analyzer(self, loc='static/lda.pkl'):

        with open(loc, 'wb') as f:
            pickle.dump(self, f)
