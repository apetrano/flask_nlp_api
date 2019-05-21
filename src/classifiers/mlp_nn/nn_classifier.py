

#------------------------------------------------------------------------------
# Third party imports
#------------------------------------------------------------------------------
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import numpy as np
import itertools as it
import dill, copy, pickle



class CentairoNN(object):
    
    def __init__(self, features, labels, holdout_pct=.10):
        self.features = features
        self.labels = labels
        self.holdout_pct = holdout_pct
        self.trained_model = None
        self.holdout_data = None
        self.training_data = None
        self.final_params = None
        self.results = None
        #self.parameter_space = config.get("parameter_space", 
        #            {'hidden_layer_sizes': list(it.permutations([10,10], 2))})
    
    def organize_data(self):
        data = list(zip(self.features, self.labels))
        np.random.shuffle(data)
        split_index = int(len(data)*self.holdout_pct)
        self.holdout_data = data[:split_index]
        self.training_data = data[split_index:]
    
    def tune_model_params(self):
        res = {}
        #todo - Autotune
        best_params = self.parameter_space
        self.final_params = best_params
    
    def evaluate_model(self):
        model = MLPClassifier()
        X_train, y_train = zip(*self.training_data)
        model.fit(X_train, y_train)
        X_test, y_test = zip(*self.holdout_data)
        Y_hat = model.predict(X_test)
        self.results = classification_report(y_test, Y_hat)
    
    def train_final_model(self):
        model = MLPClassifier(hidden_layer_sizes=(50,30))
        self.trained_model = model.fit(self.features, self.labels)
        
    def predict(self, features, return_probability=True):
        if self.trained_model == None:
            print("No trained model found")
            return
        if return_probability == True:
            return self.trained_model.predict_prob(features)
        else:
            return self.trained_model.predict(features)
    
    def build_classifier(self):
        print("Organizing data...")
        self.organize_data()
        print("Tuning model... TODO")
        #tune model
        print("Evaluating model...")
        self.evaluate_model()
        print("Training Final Model")
        self.train_final_model()
        print("Finished building classifier, ready to save")
        
    def load_model(self, file_name):
        try:
            with open(file_name, 'rb') as instream:
                self.trained_model = dill.load(instream)
            print("Successfully loaded {}".format(file_name))
        except:
            print("Could not load {}, does the file exist?".format(file_name))
    
    def save_model(self, file_name):
        if self.trained_model == None:
            print("No trained model found")
            return
        try:
            ouput = open(file_name, 'wb')
            dill.dump(self.trained_model, output)
            print("Model saved to {}".format(file_name))
        except:
            print("Failed saving model to {}".format(file_name))