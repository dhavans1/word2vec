import os
import pandas as pd
from gensim.models import word2vec
import nltk
from nltk.corpus import stopwords
import re
from bs4 import BeautifulSoup
import pickle 
import logging

# nltk.download() #executed in the initial run
  
from word2vec_scripts.config import DATA_DIR

def sentence_to_words(sentence, remove_stop_words=True, remove_numbers=False):
    #Goal: Parse text into list of sentences using nltk's punkt sentence tokenizer 
    #Input: Raw review text
    #Output: list of sentences
    words_list = []
    #Remove HTML tags
    sentence = BeautifulSoup(sentence, 'html.parser').get_text()
    #Removing special characters
    if remove_numbers:
        #eliminate non-alphabetical characters
        sentence = re.sub(r'[^A-Za-z]', " ", sentence)
    else:
        #eliminate non-alphanumeric characters
        sentence = re.sub(r'[^A-Za-z0-9]', " ", sentence)
    
    words_list = sentence.split()    
    #Removing Stop words
    if remove_stop_words:
        #store stopwords from nltk stopwords corpus
        stop_words = set(stopwords.words('english'))
        #eliminate stop words
        words_list = [word for word in words_list if word not in stop_words]
    
    return words_list

def review_to_sentences(review): 
    #Goal: Parse text into list of sentences using nltk's punkt sentence tokenizer 
    #Input: Raw review text
    #Output: list of sentences
    
    sentences = []
    #Initialise sentence tokenizer
    sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    #strip the review off of white spaces in the beginning and ending before passing it to the tokenizer
    sentences = sentence_tokenizer.tokenize(review.strip().lower())
#     print("sentence tokenizer output type: ",type(sentences))
    
    return sentences
 
def create_word2vec_input(reviews):
    #Goal: Convert raw review into word2vec compatible format - Cleaned list of lists of words  
    #Input: Raw review text
    #Output: List of lists of words
    
    total_reviews = len(reviews)
    print("total reviews ",total_reviews)
    print("sample review: ",reviews[0])
    
    formatted_reviews = []
    #Loop through each review
    for review_ind in range(total_reviews):
        sentences = []
        #convert review into list of sentences
        sentences = review_to_sentences(reviews[review_ind])
        
        #Loop through each sentences and  
        for sentence in sentences:
            if len(sentence) > 0:
                #parse, clean and split each sentence 
                formatted_reviews.append(sentence_to_words(sentence))
                
        #Print status  
        if (review_ind+1)%1000 == 0:
            print("Reviews formatted: %d/%d"%(review_ind+1,total_reviews))
           
    return formatted_reviews

def write_data(obj, file_name):
    #Goal: Serialise(pickle) python object structure and store it in file 
    #input: Python object and file where python object needs to be stored
    #output: Pickled byte representation stored in file
    
    pickle.dump(obj, open(os.path.join(DATA_DIR,file_name),'wb'))
    return
    
    
def load_data(file_name):
    #Goal: Marshall(unpickle) and load python object structure stored in file 
    #input: File where python object is stored
    #output: Python object
    
    return pickle.load(open(os.path.join(DATA_DIR, file_name),'rb'))

def preprocess(write_to_file=False):
    #Load training data 
    #Unlabeled Train data
    unlabeled_train_data = pd.read_csv(os.path.join(DATA_DIR, "unlabeledTrainData.tsv"), delimiter='\t', quoting=3, header=0)

    #Labeled Train data
    labeled_train_data = pd.read_csv(os.path.join(DATA_DIR, "labeledTrainData.tsv"), delimiter='\t', quoting=3, header=0)
    
    #Load Testing data from testData.tsv
    test_data = pd.read_csv(os.path.join(DATA_DIR, "testData.tsv"), delimiter='\t', quoting=3, header=0)
   
    #Create input for Word2vec. Word2vec takes list of sentences as input wherein each sentence is represented as a list of words
    #Unlabeled train data
    formatted_train_data = create_word2vec_input(unlabeled_train_data['review'])
   
    #Labeled train data
    formatted_train_data += create_word2vec_input(labeled_train_data['review'])
   
    #Test data
    formatted_test_data = create_word2vec_input(test_data['review'])
    
    if write_to_file:
        write_data(formatted_train_data, 'formattedTrainData.pkl')
        write_data(formatted_train_data, 'formattedTestData.pkl')

    return formatted_train_data,formatted_test_data

    
def train(train_data, do_train=True):
    #Goal: Initialise, train and store word2vec model
    #Input: training data - lists of sentences in word2vec input format
    #Output: word2vec model
    
    #Log details 
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)    
    
    #Parameter initialization
    num_features = 300 #Word vector dimensions
    context = 5 #Context window size
    min_word_count = 100 #minimum count of words in vocabulary
    num_threads = 5 #number of threads for parallel execution
    downsampling = 1e-3 
    
    #Initialize and train the model
    model = word2vec.Word2Vec(train_data, 
                              size = num_features,
                              min_count = min_word_count,
                              window = context,
                              workers = num_threads,
                              sample = downsampling)
    
    
    #Store the model with appropriate name
    model_name = "kaggle_word2vec_model_300features_40minWords_10contextWindow"
    model.save(model_name)
    
    return
    
def main(do_preprocess=False):
   
    #Preprocessing
    #Preprocess the data if it is the initial run
    print("Preprocessing Phase")
    
    if do_preprocess:
       (train_data,test_data) = preprocess(write_to_file=True)
    #Load pre-processed data
    else:
        train_data = load_data('formattedTrainData.pkl')
        test_data = load_data('formattedTestData.pkl')
    
    print("Preprocessing Phase successful")
    
    
    #Training
    print("Training Phase")

    train(train_data)
    
    print("Training Phase successful")
    
    #Testing
    
      
main(do_preprocess=False)
