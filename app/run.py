import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk import pos_tag, ne_chunk
from sklearn.base import BaseEstimator, TransformerMixin
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import plotly.graph_objs as gs

app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, message):
        sentence_list = sent_tokenize(message)
        for sentence in sentence_list:
            pos_tags = pos_tag(word_tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
class LengthExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_length = pd.Series(X).apply(len)
        return pd.DataFrame(X_length)

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df =  pd.read_sql("SELECT * FROM messages", engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # First graph data
    
    genre_names = df.groupby('genre').count()['message'].reset_index().genre.tolist()
    genre_counts = df.groupby('genre').count()['message'].reset_index().message.tolist()
    
    # Second graph data  
    category_count = df.drop(['id', 'message','original','genre'], axis=1).sum(axis=0).reset_index()
    category_count.columns = ['category', 'count']
    category_count = category_count.sort_values('count')
    category_names = category_count.category.tolist()
    category_total = category_count['count'].tolist()
    
    # create visuals
      
    graph_one = []
    
    graph_one.append(
    gs.Bar(
         x=genre_names,
         y=genre_counts
        )
    )
    
    layout_one = dict(title = 'Distribution of Message Genres',
                xaxis = dict(title = 'Genre',),
                yaxis = dict(title = 'Count'))
    
    graph_two= []
    
    graph_two.append(
    gs.Bar(
    x=category_total,
    y=category_names,
    orientation='h'
        )
    )
    
    layout_two = dict(title = 'Distribution of Categories',
                xaxis = dict(title = 'Count',),
                yaxis = dict(title = 'Category Names'))
    
    graphs = []
    graphs.append(dict(data=graph_one, layout=layout_one))
    graphs.append(dict(data=graph_two, layout=layout_two))
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()