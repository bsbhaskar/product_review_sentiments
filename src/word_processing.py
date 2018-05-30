import string
import spacy
import pickle
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases, Word2Vec
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")

class WordProcessor():
    '''
    This is a utility class that loads data related to processing words.
    '''

    def __init__(self):

        '''
        Initialize loading of keywords and vocab and custom keywords. In the future, the custom keywords will be moved to a file.
        '''
        self.nlp = spacy.load("en")
        self.stop_words = set(stopwords.words('english'))
        custom_stop_words = set(['is','have','was','has','been','','samsung','one','amazon','sony','star','stars','middle','black','use','white','dont','night','room','way','purchased','vanns','think','got','thought','way','set','nice','son','half','line','tv','picture','screen','work','hour','day','week','month','time','work','days','months','weeks','pron'])
        self.stop_words = self.stop_words.union(custom_stop_words)
        self.punctuation = set(string.punctuation)
        self.lemmatize = WordNetLemmatizer()

    def clean_document(self, doc):
        '''
        returns list of words from a document post word_processing
        '''        
        words = []
        for word in doc.lower().split():
            if ((word not in self.stop_words) and (word not in self.punctuation) and (len(word) > 1)):
                words.append(self.lemmatize.lemmatize(word))
        return words
