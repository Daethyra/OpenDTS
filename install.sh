#!/bin/bash

# create a virtual environment
python3 -m venv venv

# activate the virtual environment
source venv/bin/activate

# install the dependencies
pip install tweepy
pip install vaderSentiment
pip install sqlite3

# run the main program
python main.py

# deactivate the virtual environment
deactivate
