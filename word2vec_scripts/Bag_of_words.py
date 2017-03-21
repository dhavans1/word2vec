import os
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from collections import Counter
from pprint import pprint
from time import time
from word2vec_scripts.config import DATA_DIR
from pandas.io.tests.parser import quoting

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
    
def train_classifier(train_data, train_class_labels, num_trees):
    #create Random Forest classifier with specified number of trees
    print("Building Random Forest Classifier...")
    forest = RandomForestClassifier(n_estimators=num_trees)
    
    #train the model with training data 
    forest = forest.fit(train_data, train_class_labels)
    print("Building Random Forest Classifier successful")

    return forest

def train(train_data_file):
    #Purpose: Pre-process training data, train classifier model
    #Input: Training data file 
    #Output: Trained classifier model
    
    print("TRAINING PHASE")
    #PRE-PROCESSING
    #Load review data
    #input: Review data with three columns id, review and sentiment
    #       size: 25K
    print("Loading Training Data...")
    train_data = pd.read_csv(train_data_file, header=0, delimiter='\t', quoting=3)

    #Clean reviews
    print("Pre-processing reviews...")
    total_train_reviews = train_data["review"].size
    
    clean_train_reviews = []
    
    #Loop, clean and store each review
    for review_index in range(total_train_reviews):
        #print status on completion of every 1000 reviews
        if (review_index+1)%1000 == 0:
            print("Reviews cleaned: %d/%d"%(review_index+1,total_train_reviews))
            
        clean_train_reviews.append(review_to_words(train_data["review"][review_index]))
    print("Pre-processing reviews successful")
    
    #BUILD CLASSIFIER 
    #Define a Vectorizer to extract features
    vectorizer = CountVectorizer( analyzer = 'word',
                                  preprocessor = None,
                                  tokenizer = None,
                                  stop_words = None, 
                                  lowercase = False,
                                  max_features= 5000                                 
        )
    
    #Fit the vectorizer. create a vocabulary and create a Sparse matrix of word vectors of all reviews
    train_data_features = vectorizer.fit_transform(clean_train_reviews)
    
    #Print Features selected 
    #print("Features selected:\n", vectorizer.get_feature_names())
    
    #Convert feature vector sparse matrix to numpy array    
    train_data_features = np.array(train_data_features.toarray())
    
    #Initialise and train the RandomForest classifier model
    #Define the number of trees in the forest    
    num_of_trees = 100
    trained_model = train_classifier(train_data_features, train_data["sentiment"], num_of_trees)

    return trained_model
    
    
def validation(test_data_file, trained_model):
    #Purpose: Validate test data using trained classifier model
    #Input: Training data file 
    #Output: Trained classifier model
    
    print("TESTING PHASE")
    #Load test data
    print("Loading Test Data...")
    test_data = pd.read_csv(test_data_file, header=0, delimiter="\t", quoting=3)
    
    #Clean reviews
    print("Pre-processing reviews...")
    clean_test_reviews = []
    
    total_test_reviews = test_data.shape[0]
    #loop through each raw review and clean
    for review_ind in range(total_test_reviews):
        #print status on completion of every 1000 reviews
        if (review_ind+1)%1000 == 0:
            print("test reviews cleaned: %d/%d"%(review_ind+1, total_test_reviews))
        clean_test_reviews.append(review_to_words(test_data['review'][review_ind]))
    print("Pre-processing reviews successful")
    
    #Create vectorizer to extract features from reviews
    #Define the vectorizer
    test_vectorizer = CountVectorizer( analyzer = "word",
                                       preprocessor = None,
                                       stop_words=None,
                                       tokenizer=None,
                                       lowercase=False,
                                       max_features=5000
        )
    
    #Fit the vectorizer into review data to get the vocabulary and transform reviews into feature vector matrix
    test_data_features = test_vectorizer.fit_transform(clean_test_reviews)
    
    #Convert the feature vector sparse matrix into array for easy operations
    test_data_features = np.array(test_data_features.toarray())
    
    print("Validating test data against trained model")
    #Validate test data using trained model
    predicted_labels = trained_model.predict(test_data_features) 
    print("Validating test data successful")
    
    return test_data, predicted_labels
        
def main():
    
    #Training
    trained_model = train(os.path.join(DATA_DIR,'labeledTrainData.tsv'))
       
    #Validation
    (test_data, results) = validation(os.path.join(DATA_DIR,"testData.tsv"), trained_model)

    #OUTPUT
    #Create a pandas Dataframe to store the review ids and results 
    output = pd.DataFrame( data = { "id":test_data["id"], "sentiment": results } )
    # Write output to CSV file
    output.to_csv(os.path.join(DATA_DIR,"Bag_of_Words_model.csv"), index=False, quoting=3)
    
main()   
    