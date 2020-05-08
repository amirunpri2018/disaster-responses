import sys
import pickle

import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, RepeatedKFold

import warnings

import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

def load_data(database_filepath):
    """
    Loading a pandas DataFrame from a sqlite database
    Args:
    database_filepath: path of the sqlite database
    Returns:
    X: features (data frame)
    Y: target categories (data frame)
    label: list of category names
    category_names: index list with names of categories
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath)) 
    df =  pd.read_sql("SELECT * FROM messages", engine)
    X = df['message'].values
    y = df.drop(['id','message','original','genre'], axis=1).values
    category_names = list(df.drop(['id','message','original','genre'], axis=1).columns)
    
    return X, y, category_names

def tokenize(text):
    """
    Cleaning the text, then tokenizing and lemmatizing input text
    Args:
    text: text data as str
    Returns:
    tokens: tokenized and lemmatized text
    
    """
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())
    
    text = word_tokenize(text)
    
    tokens = [WordNetLemmatizer().lemmatize(w) for w in text if w not in stopwords.words("english")]
    
    return tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, message):
        sentence_list = nltk.sent_tokenize(message)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(word_tokenize(sentence))
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
    

def build_model():
    """
    Creating pipeline to do modeling and grid search

    Returns:
    cv: gridsearch cv object
    
    """
    
    pipeline = Pipeline([
    ('features', FeatureUnion([
    ('text_pipeline', Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer())
        ])),
        ('starting_verb', StartingVerbExtractor()),
        ('length_extractor', LengthExtractor())
    ])),
    ('clf', MultiOutputClassifier(LinearSVC(dual=False))) 
    ])
    
    parameters = {
    "clf__estimator__C": [1, 5, 10],
    "features__text_pipeline__tfidf__smooth_idf":[True,False],
    'features__text_pipeline__vect__ngram_range': [(1, 1)]
    }
    
    cv = GridSearchCV(pipeline, param_grid = parameters, scoring=["f1_weighted", "f1_micro", "f1_samples"],
        refit="f1_weighted")
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    EvaluatING the model, printing the model result, exporting the result as model_result.txt
    
    Args:
    model: model after gridsearch
    X_test: X test dataset
    Y_test: Y test dataset
    category_names: label for every category
    
    """
    
    df = pd.DataFrame.from_dict(model.cv_results_)
    
    print("Cross Validation Result on Validation Dataset")
    print("Best Score:{}".format(model.best_score_))
    print("Best Parameters:{}".format(model.best_estimator_.get_params()["clf"]))
    print("Mean Test f1 Weighted:{}".format(df["mean_test_f1_weighted"]))
    print("Mean Test f1 Micro:{}".format(df["mean_test_f1_micro"]))
    print("Mean Test f1 Samples:{}".format(df["mean_test_f1_samples"]))
    
    print("Model Result in Test Dataset")
    Y_pred = model.predict(X_test)
    print("Classification Report:{}".format(classification_report(Y_test, Y_pred, output_dict=True, target_names=category_names)))
    
    with open("model_results.txt","w") as w:
        w.write(
            "##### Cross Validation Result on Validation Dataset #####\n"
            "\n"
            "Best Score: {}\n"
            "Best Parameters: {}\n"
            "Mean Test f1 Weighted: {}\n"
            "Mean Test f1 Micro: {}\n"
            "Mean Test f1 Samples:{}\n"
            "\n"
            "\n"
            "##### Model Result in Test Dataset #####\n"
            "\n"
            "Classification Report: \n"
            "{}\n".format(
                model.best_score_,
                model.best_estimator_.get_params()["clf"],
                df["mean_test_f1_weighted"].values[0],
                df["mean_test_f1_micro"].values[0],
                df["mean_test_f1_samples"].values[0],
                str(classification_report(Y_test, Y_pred, output_dict=True, target_names=category_names)),
            )
        )
        
    print("Results stored in mode_results.txt")

def save_model(model, model_filepath):
    """
    Saving model as a .pkl or pickle file. Model_filepath is the path destination.
    Arguments:
    model: cv or model trained to save
    model_filepath: path destination
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """
    Extracting the data from databse, splitting it into train (80%) and test dataset (20%), training the model with a GridSearchCV pipeline,
    evaluating on the test set, and saving the model as a .pkl or pickle file.
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()