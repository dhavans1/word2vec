import os
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

from collections import Counter
from pprint import pprint
from time import time
from word2vec_scripts.config import DATA_DIR

def review_to_words(review_raw):
#GOAL: Convert raw string of review into a list of words (excluding all the stop words)
#input: raw review data
#output: string of words
    
    #Eliminate HTML tags
    review_text = BeautifulSoup(review_raw, 'html.parser').get_text()
    
    #Eliminate non-alphabetical characters
    review_letters_only = re.sub(r"[^A-Za-z]", " ",review_text)
    
    #Convert characters to lower case and split review string into set of words
    words = review_letters_only.lower().split()
    
    #Remove stop words that hold less meaning - a, an, etc
    #download python in-built words list which also includes stopwords list
    # nltk.download('all') #run this line only during the first execution
    
    #store stop words list as set
    stop_words = set(stopwords.words("english"))
    #Eliminate stop words
    meaningful_words = [word for word in words if word not in stop_words]
    
    #Join all the words into a string
    processed_review = " ".join(meaningful_words)
    
    return processed_review

def main():
    #Load review data
    #input: Review data with three columns id, review and sentiment
    #       size: 25K
    train_data = pd.read_csv(os.path.join(DATA_DIR,'labeledTrainData.tsv'), header=0, delimiter='\t', quoting=3)
    
    #Clean all reviews
    #Get total number of reviews
    total_reviews = train_data["review"].size
    
    clean_train_reviews = []
    
    #Loop, clean and store each review
    for review_index in range(total_reviews):
        #print status on completion of every 1000 reviews
        if (review_index+1)%1000 == 0:
            print("Reviews cleaned: %d/%d"%(review_index+1,total_reviews))
            
        clean_train_reviews.append(review_to_words(train_data["review"][review_index]))
        
        
    ###########################################
    #store the count of every word all reviews
#     word_count = Counter()
#     for clean_review in clean_train_reviews:
#         word_count.update(clean_review.split())
#     
#     #vector of first 5000 most common words
#     print("Total unique words %d"%(len(word_count.keys())))
#     
#     word_vector = word_count.most_common(5000)
    #pprint(word_vector)
    ###########################################
    
    #Define a Vectorizer
    vectorizer = CountVectorizer( analyzer = 'word',
                                  preprocessor = None,
                                  tokenizer = None,
                                  stop_words = None, 
                                  lowercase = False,
                                  max_features= 5000                                 
        )
    
    #Fit the vectorizer. create a vocabulary and create a Sparse matrix of word vectors of all reviews
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    
    #Features selected 
    print("Features selected:\n", vectorizer.get_feature_names())

    #Convert feature sparse matrix to numpy array    
    train_data_features = np.array(train_data_features.toarray())
    print(train_data_features.shape)
    
    #Print each feature and its count
    #Method1
    print("Feature\tCount")
    start_time = time()
    feature_counts = np.sum(train_data_features,axis=0)
    ind = 0
    for feature in vectorizer.get_feature_names():
        print(feature,"\t",feature_counts[ind])
        ind += 1
    end_time = time()
    time_taken1 = end_time-start_time
    
    
    #Method2 - Kaggle
    start_time = time()
    
    vocab = vectorizer.get_feature_names()
    dist = np.sum(train_data_features, axis=0)
    # For each, print the vocabulary word and the number of times it 
    # appears in the training set
    for tag, count in zip(vocab, dist):
        print(tag,count)
        
    end_time = time()
    time_taken2 = end_time-start_time
    
    print("execution time for procedure1: ",time_taken1)
    print("execution time for procedure2: ",time_taken2)
    print("Time difference of procedure1 from procedure2: ",time_taken1-time_taken2)
    
    
main()