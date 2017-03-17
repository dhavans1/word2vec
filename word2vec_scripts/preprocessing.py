import os
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords

from word2vec_scripts.config import DATA_DIR


#GOAL: Convert raw string of review into a list of words (excluding all the stop words)
#input: Review data with three columns id, review and sentiment
#       size: 25K
#output: list of words

#Load review data
train_data = pd.read_csv(os.path.join(DATA_DIR,'labeledTrainData.tsv'), header=0, delimiter='\t', quoting=3)

#extracting only words from the review - eliminating html tags
#load a sample review
example_review_raw = train_data['review'][0]
#remove non-text contents - html tags
example_review_text = BeautifulSoup(example_review_raw, 'html.parser').get_text()
#eliminate non-alphabetical characters
example_review_letters_only = re.sub(r"[^A-Za-z]", " ",example_review_text)
#convert characters to lower case
lower_case = example_review_letters_only.lower()
#split the string into Words 
words = lower_case.split()

#Remove stop words that hold less meaning - a, an, etc
#download python in-built words list which also includes stopwords list
# nltk.download('all') #run this line only during the first execution
words = [word for word in words if word not in stopwords.words("english")]
