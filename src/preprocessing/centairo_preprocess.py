


#------------------------------------------------------------------------------
# Third party imports
#------------------------------------------------------------------------------
import pandas as pd
import spacy
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import *
import dill, copy, pickle
import re
import unicodedata
import numpy as np
import h5py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
import string

#------------------------------------------------------------------------------
# Third party models
#------------------------------------------------------------------------------

nlp = spacy.load("en")
nltk.download('stopwords')
nltk.download('punkt')
stopwords = stopwords.words('english')
stemmer = PorterStemmer()
sentiment_analyzer = VS()

#------------------------------------------------------------------------------
# Main model
#------------------------------------------------------------------------------

class CentairoPreprocessor(object):
    
    def __init__(self, config={}):
        self.config = config
        self.tfidf_vectorizer = None
        self.pos_vectorizer = None
        
    def tokenize(self, doc):
        _doc = nltk.word_tokenize(doc)
        return [stemmer.stem(word) for word in _doc]
    
    def get_pos_tags(self, docs):
        res = []
        for doc in docs:
            tokens = self.tokenize(self.strip_string(doc))
            tags = nltk.pos_tag(tokens)
            tag_list = [x[1] for x in tags]
            tag_str = " ".join(tag_list)
            res.append(tag_str)
        return res
    
    def train_tfidf_vec(self, docs, ngram_range=(1,3), max_features=5000):
        self.tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, 
                                          use_idf=True, smooth_idf=True, preprocessor=self.strip_string,
                                          decode_error='replace', min_df=5,max_df=0.501,
                                         tokenizer=self.tokenize, ngram_range=ngram_range,
                                         norm='l2', sublinear_tf=False)
        self.tfidf_vectorizer.fit_transform(docs)
    
    def train_pos_vec(self,docs):
        self.pos_vectorizer = TfidfVectorizer(tokenizer=None, lowercase=False,
                                preprocessor=None, ngram_range=(1, 3),
                                stop_words=None, use_idf=False, smooth_idf=False,
                                norm=None, decode_error='replace',max_features=5000,
                                min_df=5, max_df=0.501)
        pos = self.get_pos_tags(docs)
        self.pos_vectorizer.fit_transform(pos)
    
    def train_models(self, docs):
        print("Training TFIDF vectorizer..")
        self.train_tfidf_vec(docs)
        print("Training POS vevtorizer..")
        self.train_pos_vec(docs)
        print("Models trained, ready to save..")
        
    def strip_string(self, text_string):
        space_pattern = '\s+'
        giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
            '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        mention_regex = '@[\w\-]+'
        parsed_text = re.sub(space_pattern, ' ', text_string)
        parsed_text = parsed_text.translate(str.maketrans('','',string.punctuation))
        parsed_text = re.sub(giant_url_regex, '', parsed_text)
        parsed_text = re.sub(mention_regex, '', parsed_text)
        parsed_text = ''.join([i for i in parsed_text if not i.isdigit()])
        #parsed_text = parsed_text.code("utf-8", errors='ignore')
        return parsed_text
  
    def get_sentiment(self, docs):
        res = []
        for doc in docs:
            sent = sentiment_analyzer.polarity_scores(doc)
            res.append([sent['compound'], sent['pos'], sent['neg'], sent['neu']])
        return res
            
    def doc2vec(self, docs):
        vecs = []
        for doc in docs:
            _tmp = nlp(doc)
            vecs.append(_tmp.vector)
        return vecs
    
    def transform_docs(self, docs):
        if self.tfidf_vectorizer == None:
            print("No trained tfidf vectorizer found...")
            return
        if self.pos_vectorizer == None:
            print("No trained POS vectorizer found...")
            return
        if type(docs) == str:
            docs = [docs]
        vecs = self.doc2vec(docs)
        tfidf_vecs = self.tfidf_vectorizer.transform(docs).toarray()
        pos_vecs = self.pos_vectorizer.transform(docs).toarray()
        sentiment = self.get_sentiment(docs)
        return np.concatenate([vecs, tfidf_vecs, pos_vecs, sentiment], axis=1)
        
    def load_models(self, tfidf_filename, pos_file_name):
        try:
            with open(tfidf_filename, 'rb') as instream:
                self.tfidf_vectorizer = dill.load(instream)
            print("Successfully loaded {}".format(tfidf_filename))
        except:
            print("Could not load {}, does the file exist?".format(tfidf_filename))
            return
        try:
            with open(pos_file_name, 'rb') as instream:
                self.pos_vectorizer = dill.load(instream)
            print("Successfully loaded {}".format(pos_file_name))
        except:
            print("Could not load {}, does the file exist?".format(pos_file_name))
    
    def save_models(self, tfidf_filename, pos_file_name):
        if self.tfidf_vectorizer == None:
            print("No trained model found")
            return
        try:
            output = open(tfidf_filename, 'wb')
            dill.dump(self.tfidf_vectorizer, output)
            print("Model saved to {}".format(tfidf_filename))
        except:
            print("Failed saving model to {}".format(tfidf_filename))
        if self.pos_vectorizer == None:
            print("No trained model found")
            return
        try:
            output = open(pos_file_name, 'wb')
            dill.dump(self.pos_vectorizer, output)
            print("Model saved to {}".format(pos_file_name))
        except:
            print("Failed saving model to {}".format(pos_file_name)) 