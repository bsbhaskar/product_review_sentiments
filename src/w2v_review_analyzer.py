import sys
import os
import string
import pandas as pd
from gensim.models import Phrases, Word2Vec
import warnings
warnings.filterwarnings("ignore")

class W2VReviewAnalyzer():

    def __init__(self, df, wp):

        '''
        this class builds a Word2Vec Model using Gensim library on full corpus of documents. 
        '''

        self.df_prod = df
        self.wp = wp

    def create_bow(self):

        self.df_prod['bow'] = self.df_prod['reviews'].apply(lambda x: self.wp.clean_document(x))
        self.build_trigrams()
        return self.df_prod

    def keep_token(self, token):

        is_punct = token.is_punct or token.is_space
        is_stopword = token in self.wp.stop_words
        is_pos_ = token.pos_ in ['NOUN','ADJ','PROPN',"VERB"]
        return is_pos_ and not is_punct and not is_stopword

    def lemmatized_sentence_corpus(self, documents):

        corpus_lemma = []
        for doc in documents:
            parsed_doc = self.wp.nlp(doc)
            for sent in parsed_doc.sents:
                corpus_lemma.append(' '.join([token.lemma_ for token in sent if self.keep_token(token)]))
        return corpus_lemma

    def build_trigrams(self):

        self.bigram_model = Phrases([doc.split(" ") for doc in self.lemmatized_sentence_corpus(self.df_prod['reviews'].values)], min_count=2)
        bigram_sentences = []
        for unigram_sentence in self.lemmatized_sentence_corpus(self.df_prod['reviews'].values):
            bigram_sentences.append(' '.join(self.bigram_model[unigram_sentence.split(" ")]))
        self.trigram_model = Phrases([doc.split(" ") for doc in bigram_sentences], min_count=2)

        self.trigrams_doc = []
        for doc in self.df_prod['reviews'].values:
            parsed_doc = self.wp.nlp(doc)
            bigram_doc = ' '.join(self.bigram_model[(token.lemma_ for token in parsed_doc if self.keep_token(token))])
            trigram_doc = ' '.join(self.trigram_model[(token for token in bigram_doc.split(" "))])
            self.trigrams_doc.append(self.trigram_model[(token for token in bigram_doc.split(" "))])

    def fit(self):

        self.build_trigrams()
        self.model = Word2Vec(self.trigrams_doc, size=100, window=5, min_count=5, workers=4)

    def fit2(self):

        self.model = Word2Vec(self.df_prod['bow'].values, size=100, window=5, min_count=5, workers=4)

    def most_similar(pos=[], neg=[]):

        return self.model.wv.most_similar(positive=pos, negative=neg)
