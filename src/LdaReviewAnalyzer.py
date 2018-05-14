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

class LdaReviewAnalyzer():

    #This class builds classifies documents and words into topics using
    #Latent Dirichlet Allocation algorithm from sklearn
    #pyLDAvis library is used to visualize the topics

    def __init__(self, df):
        # df is dataframe with atleaast one column named reviews which is the # # corpus used for analyzing topics. The dataframe may contain products # and ratings if necessary for segmenting the topics
        self.df_prod = df

    def vectorize(self, product=all, rating=[0,1]):

        if (product == 'all'):
            df_prod = self.df_prod[self.df_prod['rating'].apply(lambda x: x in rating)]
        else:
            df_prod = self.df_prod[self.df_prod['product'] == product]
            df_prod = df_prod[df_prod['rating'].apply(lambda x: x in rating)]

        self.vectorizer = CountVectorizer(min_df=5, max_df=0.9,
                             stop_words='english', lowercase=True,
                             token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}')
        self.data_vectorized = self.vectorizer.fit_transform(df_prod['reviews'])


    def fit_transform(self, num_topics=5, product='all',rating=[0,1]):

        self.vectorize(product,rating)
        self.lda_model = LatentDirichletAllocation(n_topics=num_topics, max_iter=10, learning_method='online')
        self.lda_results = self.lda_model.fit_transform(self.data_vectorized)

    def print_topics(self, top_n=10):

        for idx, topic in enumerate(self.lda_model.components_):
            print("Topic %d:" % (idx))
            print([(self.vectorizer.get_feature_names()[i], topic[i])
                            for i in topic.argsort()[:-top_n - 1:-1]])

    def display_topic_model(self):

        pyLDAvis.enable_notebook()
        panel = pyLDAvis.sklearn.prepare(self.lda_model, self.data_vectorized, self.vectorizer, mds='tsne')
        return pyLDAvis.display(panel)
