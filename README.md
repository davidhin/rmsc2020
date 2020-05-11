# SO / Kaggle Topic Modelling

## Dataset

**Build SO Dataset**

1. Download StackOverflow data using StackOverflow archive
2. Lightly preprocess SO, combine title, question and answers
3. Word matching using Wikipedia terms and expert judgement
4. Filter posts with at least 2 unique terms

**Build Kaggle Dataset**

1. Download Kaggle dataset 06/05/2020 at Kaggle Meta dataset
2. Lightly preprocess

## **Topic modelling**

**Selecting candidate models**

1. Remove stop words using NLTK english stopwords
2. Utilise bigrams and trigrams using Gensim
3. Lemmatisation using Spacy
4. Try number of topics 5...30 for both SO and Kaggle, using both Gensim and Mallet
5. Produce coherence plot

**Manually examining candidate models**


