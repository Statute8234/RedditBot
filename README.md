# RedditBot

[![Static Badge](https://img.shields.io/badge/openpyxl-khaki)](https://openpyxl.readthedocs.io/en/stable/)
[![Static Badge](https://img.shields.io/badge/praw-olive)](https://praw.readthedocs.io/en/stable/)
[![Static Badge](https://img.shields.io/badge/string-yellow)](https://docs.python.org/3/library/string.html)
[![Static Badge](https://img.shields.io/badge/random-chartreuse)](https://docs.python.org/3/library/random.html)
[![Static Badge](https://img.shields.io/badge/requests-green)](https://docs.python-requests.org/en/latest/)
[![Static Badge](https://img.shields.io/badge/re-lime)](https://docs.python.org/3/library/re.html)
[![Static Badge](https://img.shields.io/badge/urllib.request-teal)](https://docs.python.org/3/library/urllib.request.html)
[![Static Badge](https://img.shields.io/badge/autocorrect-aqua)](https://github.com/fsondej/autocorrect)
[![Static Badge](https://img.shields.io/badge/concurrent.futures-cyan)](https://docs.python.org/3/library/concurrent.futures.html)
[![Static Badge](https://img.shields.io/badge/numpy-turquoise)](https://numpy.org/doc/)
[![Static Badge](https://img.shields.io/badge/PIL-navy)](https://pillow.readthedocs.io/en/stable/)
[![Static Badge](https://img.shields.io/badge/BytesIO-blue)](https://docs.python.org/3/library/io.html#io.BytesIO)
[![Static Badge](https://img.shields.io/badge/secrets-indigo)](https://docs.python.org/3/library/secrets.html)
[![Static Badge](https://img.shields.io/badge/time-purple)](https://docs.python.org/3/library/time.html)
[![Static Badge](https://img.shields.io/badge/nltk-thistle)](https://www.nltk.org/)
[![Static Badge](https://img.shields.io/badge/tensorflow-plum)](https://www.tensorflow.org/api_docs)
[![Static Badge](https://img.shields.io/badge/sklearn-violet)](https://scikit-learn.org/stable/documentation.html)
[![Static Badge](https://img.shields.io/badge/pandas-orchid)](https://pandas.pydata.org/pandas-docs/stable/)


This script collects data from Reddit using the PRAW library, preprocesses it, trains models using Keras/TensorFlow, and processes images using a pre-trained ResNet50 model. It also handles file handling, handling Excel files for storage and updating data, and writes processed data to a text file.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Imports](#Imports)
- [Rating: 7/10](#Rating)
  
# About

This script combines various functionalities, including data collection from Reddit, data preprocessing and cleaning, model training and prediction, image processing, and file handling. It fetches data from a Reddit subreddit using the PRAW library, preprocesses the data, and uses pre-trained models for generating titles, scores, comments, and awards. The script also downloads images from URLs, converts them to PNG format, and predicts labels using a pre-trained model. It also handles Excel files for data storage and updating.

# Features

Copilot is an AI companion that can assist with various tasks and topics. The script you mentioned is designed to perform multiple functions related to Reddit data. It can collect data from any subreddit using the PRAW library, which is a Python wrapper for the Reddit API. The data includes post title, score, URL, comments, awards, and other metadata.
The script can preprocess and clean the data using techniques like removing stopwords, punctuation, emojis, HTML tags, and URLs, tokenizing, lemmatizing, and stemping the text data for model training and prediction. It can use pre-trained models for tasks such as generating titles, scores, comments, and awards for posts. The script can also use natural language generation models, regression or classification models, or image processing models like ResNet or VGG to predict labels for images.
Finally, the script can handle Excel files for data storage and updating. It can create, read, write, and update Excel files using the openpyxl library, store original and generated data in separate sheets or columns, and use formulas or functions to calculate metrics or statistics. The script can also format Excel files using styles, colors, or charts.

# Imports

openpyxl, praw, string, random, requests, re, urllib.request, autocorrect, concurrent.futures, numpy, PIL, io, secrets, time, nltk, word_tokenize, pos_tag, nltk.corpus, wordnet, stopwords, nltk.tokenize, sent_tokenize, nltk.stem, WordNetLemmatizer, PorterStemmer, RegexpTokenizer

# Rating

The code is designed to extract, process, and predict Reddit data using various tasks. It extracts data from Reddit using the PRAW library, processes it, and stores it in an Excel file. It also preprocesses text data using NLTK, performing spelling correction, lemmatization, and removal of unsupported characters. The code trains LSTM models for predicting Reddit posts' scores, comments, and awards, using the ResNet50 model for image classification. Images from URLs are downloaded, converted to PNG format, and processed for prediction. The code also reads and writes to Excel and text files for data storage. However, a comprehensive assessment is challenging without the context of the specific problem or project requirements. Modularizing the code for better readability, maintainability, and reusability is recommended.
