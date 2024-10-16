# RedditBot
This script collects data from Reddit using the PRAW library, preprocesses it, trains models using Keras/TensorFlow, and processes images using a pre-trained ResNet50 model. It also handles file handling, handling Excel files for storage and updating data, and writes processed data to a text file.

[![Static Badge](https://img.shields.io/badge/openpyxl-green)](https://pypi.org/project/openpyxl/)
[![Static Badge](https://img.shields.io/badge/praw-brown)](https://pypi.org/project/praw/)
[![Static Badge](https://img.shields.io/badge/string-red)](https://pypi.org/project/string/)
[![Static Badge](https://img.shields.io/badge/random-green)](https://pypi.org/project/random/)
[![Static Badge](https://img.shields.io/badge/requests-yellow)](https://pypi.org/project/requests/)
[![Static Badge](https://img.shields.io/badge/re-pink)](https://pypi.org/project/re/)
[![Static Badge](https://img.shields.io/badge/urllib-red)](https://pypi.org/project/urllib/)
[![Static Badge](https://img.shields.io/badge/autocorrect-red)](https://pypi.org/project/autocorrect/)
[![Static Badge](https://img.shields.io/badge/concurrent-purple)](https://pypi.org/project/concurrent/)
[![Static Badge](https://img.shields.io/badge/numpy-pink)](https://pypi.org/project/numpy/)
[![Static Badge](https://img.shields.io/badge/PIL-black)](https://pypi.org/project/PIL/)
[![Static Badge](https://img.shields.io/badge/io-yellow)](https://pypi.org/project/io/)
[![Static Badge](https://img.shields.io/badge/secrets-orange)](https://pypi.org/project/secrets/)
[![Static Badge](https://img.shields.io/badge/time-pink)](https://pypi.org/project/time/)
[![Static Badge](https://img.shields.io/badge/nltk-blue)](https://pypi.org/project/nltk/)
[![Static Badge](https://img.shields.io/badge/tensorflow-purple)](https://pypi.org/project/tensorflow/)
[![Static Badge](https://img.shields.io/badge/sklear-green)](https://pypi.org/project/sklear/)
[![Static Badge](https://img.shields.io/badge/pandas-black)](https://pypi.org/project/pandas/)
[![Static Badge](https://img.shields.io/badge/openpyxl,-orange)](https://pypi.org/project/openpyxl,/)
[![Static Badge](https://img.shields.io/badge/concurrent-blue)](https://pypi.org/project/concurrent/)
[![Static Badge](https://img.shields.io/badge/requests-brown)](https://pypi.org/project/requests/)
[![Static Badge](https://img.shields.io/badge/PIL-red)](https://pypi.org/project/PIL/)
[![Static Badge](https://img.shields.io/badge/io-orange)](https://pypi.org/project/io/)
[![Static Badge](https://img.shields.io/badge/numpy-gray)](https://pypi.org/project/numpy/)
[![Static Badge](https://img.shields.io/badge/keras-red)](https://pypi.org/project/keras/)
[![Static Badge](https://img.shields.io/badge/pandas-black)](https://pypi.org/project/pandas/)
[![Static Badge](https://img.shields.io/badge/sklearn-red)](https://pypi.org/project/sklearn/)
[![Static Badge](https://img.shields.io/badge/keras-gray)](https://pypi.org/project/keras/)
[![Static Badge](https://img.shields.io/badge/nltk-black)](https://pypi.org/project/nltk/)
[![Static Badge](https://img.shields.io/badge/numpy-purple)](https://pypi.org/project/numpy/)
[![Static Badge](https://img.shields.io/badge/datetime-yellow)](https://pypi.org/project/datetime/)
[![Static Badge](https://img.shields.io/badge/keras-green)](https://pypi.org/project/keras/)
## Table of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Rating: 7/10](#Rating)
  
# About

This script combines various functionalities, including data collection from Reddit, data preprocessing and cleaning, model training and prediction, image processing, and file handling. It fetches data from a Reddit subreddit using the PRAW library, preprocesses the data, and uses pre-trained models for generating titles, scores, comments, and awards. The script also downloads images from URLs, converts them to PNG format, and predicts labels using a pre-trained model. It also handles Excel files for data storage and updating.

# Features

Copilot is an AI companion that can assist with various tasks and topics. The script you mentioned is designed to perform multiple functions related to Reddit data. It can collect data from any subreddit using the PRAW library, which is a Python wrapper for the Reddit API. The data includes post title, score, URL, comments, awards, and other metadata.
The script can preprocess and clean the data using techniques like removing stopwords, punctuation, emojis, HTML tags, and URLs, tokenizing, lemmatizing, and stemping the text data for model training and prediction. It can use pre-trained models for tasks such as generating titles, scores, comments, and awards for posts. The script can also use natural language generation models, regression or classification models, or image processing models like ResNet or VGG to predict labels for images.
Finally, the script can handle Excel files for data storage and updating. It can create, read, write, and update Excel files using the openpyxl library, store original and generated data in separate sheets or columns, and use formulas or functions to calculate metrics or statistics. The script can also format Excel files using styles, colors, or charts.

# Installation
1) HTTPS - https://github.com/[User]/RedditBot.git
2) CLONE - git@github.com:{User]/RedditBot.git

# Usage

This script is useful for automated data collection and analysis on GitHub, such as running a workflow to fetch and process Reddit data, storing results in a repository, and collaborating on machine learning projects. It can also be used for version control of training scripts, datasets, and trained models. GitHub's collaborative features like pull requests and issues can be used for collaborative development. The script can also be integrated into a CI/CD pipeline for automatic testing and validation of updates. It can also be deployed using GitHub Actions for continuous deployment. An example workflow is provided, which automates the execution of the script by setting up a daily scheduled task and allowing manual triggering via the GitHub Actions interface. The script installs dependencies, runs the script, and uses GitHub secrets for sensitive information. This workflow allows for automated and continuous deployment of trained models and data processing pipelines.

# Rating

The code is designed to extract, process, and predict Reddit data using various tasks. It extracts data from Reddit using the PRAW library, processes it, and stores it in an Excel file. It also preprocesses text data using NLTK, performing spelling correction, lemmatization, and removal of unsupported characters. The code trains LSTM models for predicting Reddit posts' scores, comments, and awards, using the ResNet50 model for image classification. Images from URLs are downloaded, converted to PNG format, and processed for prediction. The code also reads and writes to Excel and text files for data storage. However, a comprehensive assessment is challenging without the context of the specific problem or project requirements. Modularizing the code for better readability, maintainability, and reusability is recommended.
