# SO / Kaggle Topic Modelling

## Dataset

**Build SO Dataset**

1. Download StackOverflow data using StackOverflow archive
2. Lightly preprocess SO, combine title, question and answers
3. Word matching using Wikipedia terms and expert judgement
4. Filter posts with at least 3 unique terms, or containing at least one main tag (expert judgement)
   1. Some trap words were things like clustering, boosting, classification, nearest neighbours etc. Clustering could refer to things like Amazon clusters, and the other words were not always data science related. A heuristic of choosing at least 3 unique terms added approximately 4000 more samples which were not obtained through the tag matching method. 

**Build Kaggle Dataset**

1. Download Kaggle dataset 06/05/2020 at Kaggle Meta dataset
2. Combine question with answers to that question

**Combined**

1. Remove samples with less than 20 words (~2000)
2. Remove samples with more than 1000 nonstop words (~3000)
3. Remove samples that are not in English
4. Remove stop words using NLTK english stopwords
5. Utilise bigrams and trigrams using Gensim
6. Lemmatisation using Spacy

## **Topic modelling**

**Selecting candidate models**

1. Gridsearch over mallet topic models using parameters:
   -  Number of Topics: 5...30
   -  Alpha: [0.01, 5, 10, 50]
   -  Optimise Interval: [0, 20, 50]
   -  Total models trained: 312
   -  Used Phoenix Job Arrays to speed up training
2. Calculated coherence score for each model and produced plot

**Manually examining candidate models**


