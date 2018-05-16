import nltk
import spacy
import string
import numpy as np
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
nltk.download('stopwords')
nltk.download('wordnet')

class NaiveReviewAnalyzer:

    def __init__(self):

        # df = dataframe with atleast three columns rating, product, review
        # review - is the corpus of documents
        # rating - consists of binary label - 1 for pos reviews & 0 for neg reviews
        self.stop_words = set(stopwords.words('english'))
        self.punctuation = set(string.punctuation)
        self.lemmatize = WordNetLemmatizer()
        self.nlp = spacy.load("en")


    def clean_document(self, doc):

        doc_no_stopwords = " ".join([i for i in doc.lower().split() if i not in self.stop_words])
        doc_no_punctuation = "".join(i for i in doc_no_stopwords if i not in self.punctuation)
        doc_Lemmatized = " ".join(self.lemmatize.lemmatize(i) for i in doc_no_punctuation.split())

        return doc

    def create_bow(self, df):
        
        self.df = df
        self.df['bow'] = self.df['reviews'].apply(lambda x: self.clean_document(x))

    def create_word_list(self, product='all', rating=[5,1]):

        if (product == 'all'):
            df_prod = self.df[self.df['rating'].apply(lambda x: x in rating)]
        else:
            df_prod = self.df[self.df['product'] == product]
            df_prod = df_prod[self.df['rating'].apply(lambda x: x in rating)]

        tfidf = TfidfVectorizer(stop_words='english',max_features=10000)
        X_descr_vectors = tfidf.fit_transform(df_prod['bow'])
        nb = MultinomialNB()
        nb.fit(X_descr_vectors, df_prod['rating'].transpose())
        y_hat = nb.predict_proba(X_descr_vectors)
        arr = np.argsort(nb.feature_log_prob_[0])[-20:-1]

        list_of_words = []
        for i in arr:
            list_of_words.append(tfidf.get_feature_names()[i])

        tokens = self.nlp(' '.join(list_of_words))
        list_of_tokens = {}
        for token in tokens:
            value = list_of_tokens.get(token.pos_,[])
            value.append(token.orth_)
            list_of_tokens[token.pos_] = value

        return list_of_tokens
