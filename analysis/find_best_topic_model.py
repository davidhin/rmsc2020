#%%#######################################################################
#                                   SETUP                                #
##########################################################################
import os
import pandas as pd
from pathlib import Path
from operator import itemgetter
from gensim.models.wrappers import LdaMallet
import logging
logging.basicConfig(level=logging.INFO)
from tqdm import tqdm
tqdm.pandas()

# Load model
KEY = '50.0_24_50'
FOLDER = "examine_{}/".format(KEY)
Path(FOLDER).mkdir(exist_ok=True)
topics_path = '../hptuning/topics/topics_{}.parquet'.format(KEY)
model_topics = pd.read_parquet(topics_path)
corpus = pd.read_parquet('../data/corpus.parquet')
raw_texts = pd.read_parquet('../data/df_lang.parquet')
raw_texts['key'] = raw_texts.postid.astype('str')+"_"+raw_texts.source
raw_texts = raw_texts[raw_texts.lang=='en']

#%%#######################################################################
#                 Generate info for manual examination                   #
##########################################################################

# Save Mallet Model Topics
try: os.symlink('../hptuning/malletmodels/', 'malletmodels')
except: print("already created symlink")
lda = LdaMallet.load('../hptuning/models/lda_{}.mallet.model'.format(KEY))
topicsdf = pd.DataFrame(lda.show_topics(50), columns=['topic','words'])
topicsdf['label'] = 'TBD'
topicsdf.to_csv(FOLDER+'topics.csv',index=None)

# Combine postid+source from corpus and topics from model_topics
new_corpus = corpus.join(model_topics, rsuffix='_r')
new_corpus['key'] = new_corpus.postid.astype('str')+"_"+new_corpus.source
new_corpus = new_corpus[['key','topics']]
new_corpus = new_corpus.iloc[new_corpus.key.drop_duplicates().index]

# Join to create final df and top posts
texts = raw_texts.set_index('key').join(new_corpus.set_index('key')).dropna()
texts = texts.reset_index()
texts = texts.iloc[texts.key.drop_duplicates().index]
texts['topic'] = texts.topics.apply(lambda x: max(x,key=itemgetter(1))[0])
texts['val'] = texts.topics.apply(lambda x: max(x,key=itemgetter(1))[1])
texts = texts.set_index('topic').join(topicsdf.set_index('topic')).reset_index()
texts = texts[['key','Message','topic','val','words']].copy()
texts.sort_values('val',ascending=False).groupby('topic').head(20)\
     .to_csv(FOLDER+'top_posts.csv',index=False)

# Number of posts in each topic
texts.groupby('topic').count().reset_index()[['topic', 'key']]\
    .to_csv(FOLDER+'counts.csv',index=False)

# Save final topics
new_corpus.to_parquet('../data/post_topics_final.parquet',
                      compression='gzip',
                      index=None)
