#%%#######################################################################
#                                SETUP                                   #
##########################################################################
import pandas as pd
import numpy as np
import sys
import so_textprocessing as stp

# Get arguments
print(sys.argv[1] , sys.argv[3])
if str(sys.argv[2]) == "2":
    print("SKIP")
    sys.exit()

# Read Data and Group by Forum Topic ID
df_raw = pd.read_csv('ForumMessages.csv')

# Group by Forum Topic
df = df_raw.groupby('ForumTopicId').agg({ 
        'Message': lambda x: ' '.join([str(i) for i in x]),
    }).reset_index()

# Remove HTML Tags
tp = stp.TextPreprocess()
df = tp.transform_df(df, reformat="processonly", columns=['Message'])

#%%#######################################################################
#                              Get Words                                 #
##########################################################################
import os
import re
import time
import random
import itertools
from glob import glob
import logging

import numpy as np
import pandas as pd

import gensim
import spacy
import pyLDAvis
import gensim.corpora as corpora
import pyLDAvis.gensim  # don't skip this
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from gensim.models import CoherenceModel
from pprint import pprint

pd.set_option('display.max_colwidth', -1)

## logging
path = '/fast/users/a1720858/topicmodelling/'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)

# NLTK Stop words
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'would', 
                   'need', 'want', 'could', 'say', 'also', 'using', 
                   'used', 'trying', 'tried', 'try', 'may', 'get', 'got',
                   'nbsp', 'quote'])

## Sentence to words
data = df.Message.to_list()
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True)) 
data_words = list(sent_to_words(data))

#%%#######################################################################
#                           Feature Extraction                           #
##########################################################################

## Bigrams and trigrams
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) 
             if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc 
                          if token.pos_ in allowed_postags])
    return texts_out

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load('en', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, 
                                allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#%%#######################################################################
#                           Create dictionary                            #
##########################################################################

print(data_lemmatized[:1])

# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)

# Create Corpus
texts = data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
print(corpus[:1])

# Get arguments
i = int(sys.argv[1])
ldatype = sys.argv[3]
ssize = int(sys.argv[2])

print(i, ldatype, ssize)

if ldatype == "gensim":

    scores = []
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=i,
                                               random_state=100,
                                               update_every=1,
                                               chunksize=100,
                                               passes=10,
                                               alpha='auto',
                                               per_word_topics=True)

    # Print the Keyword in the 10 topics
    pprint(lda_model.print_topics())
    doc_lda = lda_model[corpus]

    # Compute Perplexity
    model_perplexity = lda_model.log_perplexity(corpus)
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                         texts=data_lemmatized, 
                                         dictionary=id2word, 
                                         coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    scores.append((i, model_perplexity, coherence_lda, ssize, ldatype))

    # Save model
    lda_model.save(path+'models/KG_lda_{}_{}_{}.model'.format(i, ssize, ldatype))

    # Visualize the topics
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, path+'kaggle_topic_modelling/lda_plots/lda_{}_{}_{}.html'.format(i, ssize, ldatype))
    pd.DataFrame(scores).to_csv(path+'kaggle_topic_modelling/lda_plots/lda_scores_{}_{}_{}.csv'.format(i, ssize, ldatype))

if ldatype == "mallet":
    
    # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
    mallet_path = path+'mallet-2.0.8/bin/mallet' # update this path
    ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, 
                                                 corpus=corpus, 
                                                 num_topics=i, 
                                                 id2word=id2word)
    pprint(ldamallet.show_topics(formatted=False))

    # Compute Coherence Score
    coherence_model_ldamallet = CoherenceModel(model=ldamallet, 
                                               texts=data_lemmatized, 
                                               dictionary=id2word, 
                                               coherence='c_v')
    coherence_ldamallet = coherence_model_ldamallet.get_coherence()
    print('\nCoherence Score: ', coherence_ldamallet)

    def mallet_to_lda(mallet_model):
        model_gensim = gensim.models.ldamodel.LdaModel(
            id2word=mallet_model.id2word, 
            num_topics=mallet_model.num_topics,
            alpha=mallet_model.alpha, 
            eta=0, 
            iterations=1000,
            gamma_threshold=0.001,
            dtype=np.float32
        )
        model_gensim.sync_state()
        model_gensim.state.sstats = mallet_model.wordtopics
        return model_gensim

    # Visualize the topics
    vis = pyLDAvis.gensim.prepare(mallet_to_lda(ldamallet), corpus, id2word)
    pyLDAvis.save_html(vis, path+'kaggle_topic_modelling/lda_plots/lda_{}_{}_{}_mallet.html'.format(i, ssize, ldatype))

    # Save model
    mallet_to_lda(ldamallet).save(path+'models/KG_lda_{}_{}_{}.model'.format(i, ssize, ldatype))

    scores = []
    scores.append((i, -1, coherence_ldamallet, ssize, ldatype))
    pd.DataFrame(scores).to_csv(path+'kaggle_topic_modelling/lda_plots/lda_scores_{}_{}_{}.csv'.format(i, ssize, ldatype))    

# %%
