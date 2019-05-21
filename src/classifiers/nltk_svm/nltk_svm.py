"""
Module to classify any string.
..module: nltk_svm
..author: Adam Petranovich <petranovich@centairo.com>
"""

#------------------------------------------------------------------------------
# Required Imports
#------------------------------------------------------------------------------

from utilities import *

#------------------------------------------------------------------------------
# Test Config 
#------------------------------------------------------------------------------
test_config = {'data_path': '../../data'}

#------------------------------------------------------------------------------
# Main Class
#------------------------------------------------------------------------------

class NltkSvm(object):

    def __init__(self, config):
        self.model = joblib.load('{}/final_model.pkl'.format(config['data_path']))
        self.tf_vectorizer = joblib.load('{}/final_tfidf.pkl'.format(config['data_path']))
        self.idf_vector = joblib.load('{}/final_idf.pkl'.format(config['data_path']))
        self.pos_vectorizer = joblib.load('{}/final_pos.pkl'.format(config['data_path']))

    def analyze_messages(self, message):
        res = {}
        X = transform_inputs(message, self.tf_vectorizer, self.idf_vector,
            self.pos_vectorizer)
        preds = predictions(X, self.model)
        res['Classification'] = class_to_name(preds[0])
        res['Sentiment'] = get_sentiment(message[0])
        return res

    
if __name__ == '__main__':
    messages = ['Take me to boob palace']
    clf = NltkSvm(test_config)
    res = clf.analyze_messages(messages)
    print(res)
