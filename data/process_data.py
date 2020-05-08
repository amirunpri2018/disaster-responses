import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Reading two csv files and transforming categories into appropriate columns
    Args:
    messages_filepath: path of the messages data
    categories_filepath: path of the categories data
    Returns:
    df: data merging between messages and categories (data frame)
    
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, how='inner', on=['id'])
    
    categories = df['categories'].str.split(';',expand=True)
    categories[categories[0] == 'related-2'] = 'related-1'
    
    row = categories.iloc[0]
    category_colnames = row.str[:-2]
    categories.columns = category_colnames
    
    for column in categories:
 
        categories[column] = categories[column].str.strip().str[-1]

        categories[column] = categories[column].astype('int64')
        
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis = 1)
    
    return df
        
def clean_data(df):
    """
    Cleaning the dataframe
    Args:
    df: data merging between messages and categories (data frame)
    Returns:
    df: final data after cleaning duplicates (data frame)
    
    """
    df.drop_duplicates(inplace = True)
    
    return df

def save_data(df, database_filename):
    """
    Saving the data into SQL database
    Args:
    df: data merging between messages and categories (data frame)
    database_filename: database filename (ex: database_name.db)
    Returns:
    df: final data after cleaning duplicates (data frame)
    
    """    
    engine = create_engine('sqlite:///{}'.format(database_filename)) 
    engine.execute("DROP TABLE IF EXISTS messages")
    df.to_sql('messages', engine, index=False)

def main():
    """
    Loading the data from csv files, merging the data, removing duplicate data, and saving it in SQL database.
    
    """
    
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()