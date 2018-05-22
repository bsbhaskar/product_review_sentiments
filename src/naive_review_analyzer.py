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
        custom_stop_words = set(['samsung','one','amazon','sony','star','stars','middle','black','use','tv','white','dont','night','room','way','purchased','vanns','think','got','thought','way','great','set','nice','son','half','line','tv','picture','screen','hour','day','week','month','time','work','days','months','weeks'])
        self.stop_words = self.stop_words.union(custom_stop_words)
        self.punctuation = set(string.punctuation)
        self.lemmatize = WordNetLemmatizer()
        self.nlp = spacy.load("en")


    def clean_document(self, doc):

        doc_no_stopwords = " ".join([i for i in doc.lower().split() if i not in self.stop_words])
        doc_no_punctuation = "".join(i for i in doc_no_stopwords if i not in self.punctuation)
        doc_Lemmatized = " ".join(self.lemmatize.lemmatize(i) for i in doc_no_punctuation.split())

        return doc_no_stopwords

    def create_bow(self, df):

        self.df = df
        self.df['bow'] = self.df['reviews'].apply(lambda x: self.clean_document(x))

    def create_word_list(self, product='all', rating=[1,5]):

        if (product == 'all'):
            df_prod = self.df[self.df['rating'].apply(lambda x: x in rating)]
        else:
            df_prod = self.df[self.df['product'] == product]
            df_prod = df_prod[self.df['rating'].apply(lambda x: x in rating)]

        tfidf = TfidfVectorizer(stop_words=self.stop_words,max_features=10000)
        X_descr_vectors = tfidf.fit_transform(df_prod['bow'])
        nb = MultinomialNB()
        nb.fit(X_descr_vectors, df_prod['rating'].transpose())
        y_hat = nb.predict_proba(X_descr_vectors)
        arr = np.argsort(nb.feature_log_prob_[0])[-50:-1]

        list_of_words = []
        list_of_prob = []
        max_prob = np.exp(nb.feature_log_prob_[0][arr[-1]])

        for i in arr:
            list_of_words.append(tfidf.get_feature_names()[i])
            list_of_prob.append(int((np.exp(nb.feature_log_prob_[0][i])/max_prob)*100))


        tokens = self.nlp(' '.join(list_of_words))

        list_of_tokens = {}
        list_of_nouns = []
        list_of_adjs = []
        list_of_verbs = []
        for i, word in enumerate(list_of_words):
            tokens = self.nlp(word)
            value = list_of_tokens.get(tokens[0].pos_,[])
            value.append((tokens[0].orth_,list_of_prob[i]))
            list_of_tokens[tokens[0].pos_] = value
            if (tokens[0].pos_ in ['NOUN']):
                list_of_nouns.append((word,list_of_prob[i]))
            elif (tokens[0].pos_ in ['ADJ']):
                list_of_adjs.append((word,list_of_prob[i]))
            elif (tokens[0].pos_ in ['VERB']):
                list_of_verbs.append((word,list_of_prob[i]))

        return list(reversed(list_of_nouns)),list(reversed(list_of_adjs))
