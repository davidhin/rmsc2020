#%%#######################################################################
#                          SETUP / Read Data                             #
##########################################################################

import pandas as pd
import numpy as np
import sys
import so_textprocessing as stp
from gensim.models import LdaModel
from langdetect import detect
from tqdm import tqdm
tqdm.pandas()

# Read Data and Group by Forum Topic ID
df_kg = pd.read_csv('../data/kg_forummessages.csv')

# Group by Forum Topic
df_kg = df_kg.groupby('ForumTopicId').agg({
    'Message': lambda x: ' '.join([str(i) for i in x]),
}).reset_index()

# Remove HTML Tags
tp = stp.TextPreprocess()
df_kg = tp.transform_df(df_kg, reformat="processonly", columns=['Message'])
df_kg['source'] = 'kg'
df_kg = df_kg.rename(columns={'ForumTopicId':'postid'})

# Read SO
df_so = pd.read_parquet('../data/so_datascience.parquet')
df_so = df_so[(df_so.tags.str.contains("""deep-learning|neural-network|tensorflow|
                                           keras|pytorch|kaggle|machine-learning|
                                           computer-vision|caffe|opencv|
                                           pycaffe|random-forest|statistics|dplyr|
                                           k-means|cluster-analysis|pca|bigdata""")) |
               (df_so.uniq > 3)]

# Combine SO text
df_so['Message'] = df_so.apply(lambda row: "{} {} {} {}".format(
    ' '.join(row.tags.split("|")),
    row.title,
    row.question,
    row.answers
), axis=1)

# Remove unnecessary columns
df_so = df_so[['postid','Message']]
df_so['source'] = 'so'

# Combine DF
df = pd.concat([df_so,df_kg])

#%%#######################################################################
#                              Preprocess text                           #
##########################################################################
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
import gensim.corpora as corpora
import gensim
import spacy
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)

# Determine Language
df['len'] = df.Message.progress_apply(lambda x: len(x.split()))
df = df[df.len>10]
def getlang(s):
    try: return detect(s)
    except: return ""
df['lang'] = df.Message.progress_apply(getlang)
df = df[df.lang=='en']

# %% Split, stopword removal and another length filter
print("Preprocessing...")
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'would', 
                'need', 'want', 'could', 'say', 'also', 'using', 
                'used', 'trying', 'tried', 'try', 'may', 'get', 'got',
                'nbsp', 'quote', 'quot'])

def gensimsp(s):
    s_new = simple_preprocess(s, deacc=True)
    s_new = [word for word in s_new if word not in stop_words]
    return s_new
    
df.Message = df.Message.progress_apply(gensimsp)
df['len'] = df.Message.progress_apply(len)
df = df[df.len>20]
df = df[df.len<1000]

# %% Sentence to words
print("Making bi/trigrams...")
nlp = spacy.load('en', disable=['parser', 'ner'])
postags=['PROPN', 'NOUN', 'ADJ', 'VERB', 'ADV']
bigram = gensim.models.Phrases(df.Message, 
                                min_count=5, 
                                threshold=100)
trigram = gensim.models.Phrases(bigram[df.Message], threshold=100)
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

def gensimsplit(s):
    s_new = trigram_mod[bigram_mod[s]]
    s_new = nlp(" ".join(s_new))
    s_new = [t.lemma_ for t in s_new if t.pos_ in postags]
    return s_new

df.Message = df.Message.progress_apply(gensimsplit)
df.to_parquet('../data/df_lemmagram.parquet',compression='gzip',index=False)

#%%#######################################################################
#                      Make dictionary and corpus                        #
##########################################################################

print("finished")
id2word = corpora.Dictionary(df.Message)
id2word.save('id2word.dictionary')
id2word = corpora.Dictionary.load('../data/id2word.dictionary')
df['corpus'] = df.Message.progress_apply(id2word.doc2bow)
df.to_parquet('../data/corpus.parquet',compression='gzip',index=False)
