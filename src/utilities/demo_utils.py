from bokeh.plotting import figure, output_file, show
import numpy as np
from bokeh.embed import components
from bokeh.models import Span, Label
from bokeh.models import ColumnDataSource, FixedTicker, PrintfTickFormatter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS


def plot_centairo_results(res):
    y = np.arange(len(res.keys()))+.5
    bar_lengths = [x*100 for x in res.values()]
    plot = figure(title="Centairo Classification Results", plot_width=800, x_range=(0, 101),
              plot_height=300, y_range=[x for x in res.keys()], tools="pan,wheel_zoom,box_zoom,reset")
    plot.hbar(y, left=0, right=bar_lengths, height=.5, color="#b3de69")
    vline = Span(location=50, dimension='height', line_color='red', line_width=3, name='p')
    plot.renderers.extend([vline])
    my_label = Label(x=50, y=2.75, text='Threshold')
    plot.xaxis.formatter = PrintfTickFormatter(format="%d%%")
    plot.add_layout(my_label)
    plot.xaxis.axis_label = "Confidence Score"
    return plot


def plot_centairo_sentiment(data):
    x = np.linspace(0, 100, 100)
    source = ColumnDataSource(data=dict(x=x))
    categories = [x for x in data.keys()]
    values = [y for y in data.values()]
    p = figure(y_range=categories, plot_width=800, plot_height=300,  x_range=(0, 101), toolbar_location=None, 
           title="Centairo Sentiment and Emotion Results")
    p.square(values, categories, size=20, color="#b3de69")
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "#000000"
    p.xaxis.formatter = PrintfTickFormatter(format="%d%%")
    p.xaxis.axis_label = "Confidence Score"
    return p

def get_sentiment(message):
    sentiment_analyzer = VS()
    res = sentiment_analyzer.polarity_scores(message)
    results = {}
    results['Positive'] = res['pos']*100
    results['Negative'] = res['neg']*100
    results['Neutral'] = res['neu']*100
    return results