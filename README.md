# Table of Contents

1. [Installation](#ins)
2. [Project Motivation](#pro)
3. [File Descriptions](#fil)
4. [Instructions](#itr)
5. [Results](#res)
6. [Limitations](#lim)
7. [Licensing, Authors and Acknowledgements](#lic)

<a name="ins"></a>
# Installation

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python. The code should run with no issues using Python versions 3.*. 

Clone this GIT repository:

https://github.com/dhaneswaramandrasa/disaster-responses.git

<a name="pro"></a>
# Project Motivation

The project is aiming to make a automated message classification that occured during a disaster that will be useful for organizations, agencies, or governments where their help is urgently needed.

<a name="fil"></a>
# File Descriptions

The full set of data is available here. There is a notebook available to explain the project.

<a name="itr"></a>
### Instructions:

To run the web app locally, follow the step:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

<a name="res"></a>
# Result

The web app of the code can be seen [here](https://medium.com/@dhaneswara.mandrasa/covid-19-this-is-why-some-countries-suffer-more-than-others-6ca7ee3e3c25).

<a name="lim"></a>
# Limitation
The precision and recall result for each categories is vary, depend on the occurence of the categories in the training dataset. For example the most common categories such as 'related' which occurs 20,000 times will produces better precision and recall than categories like 'child_alone' that only occurs 188 times.

<a name="lic"></a>
# Licensing, Authors, Acknowledgements

The dataset is taken from FigureEight. You can find the Licensing for the data and other descriptive information at the link available [here](https://appen.com/). 