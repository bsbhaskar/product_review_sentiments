import sys
import os
import string
import pandas as pd
import numpy as np
import spacy
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases, Word2Vec
from nltk.corpus import stopwords
from load_review_data import ReviewDataLoader
import warnings
warnings.filterwarnings("ignore")

class Trigrams():
    '''
    This class trains model for bigrams and trigrams by analyzing the full document. The model can then be used to clean up the original reviewsself
    '''

    def __init__(self):

        self.nlp = spacy.load("en")
        self.stop_words = set(stopwords.words('english'))
        custom_stop_words = set(['is','have','was','has','been','','samsung','one','amazon','sony','star','stars','middle','black','use','white','dont','night','room','way','purchased','vanns','think','got','thought','way','set','nice','son','half','line','tv','picture','screen','work'])
        self.stop_words = self.stop_words.union(custom_stop_words)
        self.punctuation = set(string.punctuation)
        self.lemmatize = WordNetLemmatizer()

    def load_model(self):
        self.bigram_model = pickle.load(open('../static/bigram_model.pkl','rb'))
        self.trigram_model = pickle.load(open('../static/trigram_model.pkl','rb'))

    def clean_document(self, doc):
        words = []
        for word in doc.lower().split():
            if ((word not in self.stop_words) and (word not in self.punctuation) and (len(word) > 1)):
                words.append(self.lemmatize.lemmatize(word))
        return words

    def create_bow(self):

        self.df_prod['bow'] = self.df_prod['reviews'].apply(lambda x: self.clean_document(x))
        self.build_trigrams()
        return self.df_prod

    def keep_token(self, token):

        is_punct = token.is_punct or token.is_space
        is_stopword = token in self.stop_words
        is_pos_ = token.pos_ in ['NOUN','ADJ','PROPN',"VERB"]
        #is_pos_ = token.pos_ in ['NOUN','ADJ']
        return is_pos_ and not is_punct and not is_stopword

    def lemmatized_sentence_corpus(self, documents):

        corpus_lemma = []
        for doc in documents:
            clean_doc = ' '.join(self.clean_document(doc))
            parsed_doc = self.nlp(clean_doc)
            for sent in parsed_doc.sents:
                corpus_lemma.append(' '.join([token.lemma_ for token in sent if self.keep_token(token)]))
        return corpus_lemma

    def fit(self, df):

        self.df_prod = df
        self.bigram_model = Phrases([doc.split(" ") for doc in self.lemmatized_sentence_corpus(self.df_prod['reviews'].values)], min_count=2)
        bigram_sentences = []
        for unigram_sentence in self.lemmatized_sentence_corpus(self.df_prod['reviews'].values):
            bigram_sentences.append(' '.join(self.bigram_model[unigram_sentence.split(" ")]))
        self.trigram_model = Phrases([doc.split(" ") for doc in bigram_sentences], min_count=2)

    def transform(self, df):
        trigrams_doc = []
        for doc in df['reviews'].values:
            clean_doc = ' '.join(self.clean_document(doc))
            parsed_doc = self.nlp(clean_doc)
            bigram_doc = ' '.join(self.bigram_model[(token.lemma_ for token in parsed_doc if self.keep_token(token))])
            trigram_doc = ' '.join(self.trigram_model[(token for token in bigram_doc.split(" "))])
            trigrams_doc.append(trigram_doc)
        df['reviews'] = trigrams_doc
        return df

    def get_trigrams(self, doc):
        parsed_doc = self.nlp(doc)
        bigram_doc = ' '.join(self.bigram_model[(token.lemma_ for token in parsed_doc if self.keep_token(token))])
        trigram_doc = ' '.join(self.trigram_model[(token for token in bigram_doc.split(" "))])
        return ' '.join(self.clean_document(trigram_doc))

if __name__ == '__main__':
    rdl = ReviewDataLoader()
    df_all = rdl.retrieve_all_reviews()
    tr = Trigrams()
    tr.build_trigrams(df_all)
    with open('../static/bigram_model.pkl', 'wb') as f:
        pickle.dump(tr.bigram_model, f)
    with open('../static/trigram_model.pkl', 'wb') as f:
        pickle.dump(tr.trigram_model, f)
