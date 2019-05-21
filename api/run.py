#!flask/bin/python
from flask import Flask, jsonify, request, abort
from flask_cors import CORS
import sys
sys.path.append('../src/classifiers/mlp_nn')
sys.path.append('../src/preprocessing')
sys.path.append('../src/utilities')
from nn_classifer import *
from centairo_preprocess import *
from demo_utils import *

app = Flask(__name__)
CORS(app)
api_config = {'data_path': '../data'}

#initiate preprocessor and NN
pre_process = CentairoPreprocessor()
clf = CentairoNN()
sentiment_analyzer = VS()

#Load trained models
pre_process.load_models('../src/trained_models/hatespeech_tfidf_vectorizer',
	'../src/trained_models/hatespeech_pos_vectorizer')
clf.load_model('../src/trained_models/hatespeech_nn')

@app.route('/api/text_analyzer', methods=['GET', 'POST'])
def get_tasks():
	if not request.json or not 'message' in request.json:
		abort(400)
	message = request.json['message']
	processed_message = pre_process.transform_docs(message)
	predicted_classes = clf.predict(processed_message)
	res = predicted_classes[0]
	results = {}
	results['Hate Speech'] = res[0]
	results['Offensive Langauge'] = res[1]
	#results['Neither'] = res[2]

	plot = plot_centairo_results(results)
	script, div = components(plot)
	print(message)

	sentiment_plot= plot_centairo_sentiment(get_sentiment(message[0]))
	sentiment_script, plot_sentiment = components(sentiment_plot)

	api_response = {}
	api_response['classification_plot'] = div
	api_response['classification_script'] = script
	api_response['plot_sentiment'] = plot_sentiment
	api_response['sentiment_script'] = sentiment_script

	return jsonify(api_response)

if __name__ == '__main__':
    app.run(debug=True)