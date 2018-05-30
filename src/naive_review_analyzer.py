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

    '''
    The class is used to extract key features from positive and negative reviews. First create bag of words by calling bow(). Create word list returns list of nouns or key features and adjectives or sentiments.
    '''

    def __init__(self, wp):

        '''
        df = dataframe with atleast three columns rating, product, review
        review - is the corpus of documents
        rating - consists of binary label - 1 for pos reviews & 0 for neg reviews
        '''
        self.wp = wp


    def clean_document(self, doc):

        doc_no_stopwords = " ".join([i for i in doc.lower().split() if i not in self.wp.stop_words])
        doc_lemmatized = " ".join(self.wp.lemmatize.lemmatize(i) for i in doc_no_stopwords.split())
        return doc_lemmatized

    def create_bow(self, df):

        self.df = df
        self.df['bow'] = self.df['reviews'].apply(lambda x: self.clean_document(x))

    def create_word_list(self, product='all', rating=[1,5]):
        '''
        For a given product, identifies key features within a given corpus of documentsself.
        Step 1 is to vectorize the documents using TFIDF
        Step 2 is to fit Multinomial Naive Bayes algorithm
        Step 3 is to get log probabilities of features or sort them in ascending order
        Step 4 is to return list of Nouns as Features and list of adjectives as sentiments in descending order.
        '''
        if (product == 'all'):
            df_prod = self.df[self.df['rating'].apply(lambda x: x in rating)]
        else:
            df_prod = self.df[self.df['product'] == product]
            df_prod = df_prod[self.df['rating'].apply(lambda x: x in rating)]

        tfidf = TfidfVectorizer(stop_words=self.wp.stop_words,max_features=10000)
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


        tokens = self.wp.nlp(' '.join(list_of_words))

        list_of_tokens = {}
        list_of_nouns = []
        list_of_adjs = []
        list_of_verbs = []
        for i, word in enumerate(list_of_words):
            tokens = self.wp.nlp(word)
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
