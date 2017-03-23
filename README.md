# Kaggle word2vec

Implementing Kaggle's [word2vec tutorial](https://www.kaggle.com/c/word2vec-nlp-tutorial).

Installing Gensim:
(Issue resolved: slow version of gensim.models.word2vec is being used)

 * For python3 users:
 	* Install python3-dev - ubuntu has pip version 2 by default and although your virtual environment has python 3 setup, pip installations would result in invocation of pip version 2 which install python2 packages. So, this needs 
```
sudo apt-get install python3-dev
pip install --no-cache-dir gensim
```

( force build rather than fetching from cache if present)
