# %% Load Libraries
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from operator import itemgetter

import gensim
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import gensim.corpora as corpora

import pyLDAvis
import pyLDAvis.gensim  

from tqdm import tqdm
import logging
tqdm.pandas()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                    level=logging.INFO)

# Load Data
df = pd.read_parquet('../data/corpus.parquet')
id2word = corpora.Dictionary.load('../data/id2word.dictionary')
print("LOADED DATA")

# %% Train Mallet

## Load Parameters
try:
    ALPHA=float(sys.argv[1])
    NTOPICS=int(sys.argv[2])
    OPTINT=int(sys.argv[3])
except:
    ALPHA=5
    NTOPICS=20
    OPTINT=0
KEY = "{}_{}_{}".format(ALPHA,NTOPICS,OPTINT)
print(ALPHA, NTOPICS)

## Train
mallet_path = '../mallet-2.0.8/bin/mallet' # update this path
mallet_prefix = "malletmodels/m_{}/".format(KEY)
Path(mallet_prefix).mkdir(parents=True, exist_ok=True)
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path,
                                            corpus=df.corpus,
                                            num_topics=NTOPICS,
                                            id2word=id2word,
                                            alpha=ALPHA,
                                            optimize_interval=OPTINT,
                                            random_seed=42,
                                            prefix=mallet_prefix)

# Save Mallet Model
Path("models").mkdir(exist_ok=True)
ldamallet.save('models/lda_{}.mallet.model'.format(KEY))

# Compute Coherence Score

# %% Generate Visualisation
def ldaMalletConvertToldaGen(mallet_model):
    model_gensim = LdaModel(id2word=mallet_model.id2word, 
                            num_topics=mallet_model.num_topics, 
                            alpha=mallet_model.alpha, 
                            eta=0, 
                            iterations=1000, 
                            gamma_threshold=0.001, 
                            dtype=np.float32)
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim
converted_model = ldaMalletConvertToldaGen(ldamallet)

# %% Save all metrics
Path("eval").mkdir(exist_ok=True)
def get_coherence(model, texts, dictionary):
    coherence_model_ldamallet = CoherenceModel(model=model,
                                               texts=texts,
                                               dictionary=dictionary,
                                               coherence='c_v')
    return coherence_model_ldamallet.get_coherence()
perplexity = converted_model.log_perplexity(df.corpus)
coh_mallet = get_coherence(ldamallet, df.Message, id2word)
coh_convert = get_coherence(converted_model, df.Message, id2word)
scores = [(ALPHA, NTOPICS, OPTINT, coh_mallet, coh_convert, perplexity)]
pd.DataFrame(scores).to_csv('eval/score_{}.csv'.format(KEY))
print(scores)

# %% Visualize the topics
vis = pyLDAvis.gensim.prepare(converted_model, df.corpus, id2word, mds='mmds')
pyLDAvis.save_html(vis, 'eval/vis_{}.html'.format(KEY))

# %% Save Topics
Path("topics").mkdir(exist_ok=True)
df['topics'] = ldamallet[df.corpus]
df[['postid','topics']].to_parquet('topics/topics_{}.parquet'.format(KEY),
                                   compression='gzip',
                                   index=None)

# %% Inference
df['topic_max_num'] = df.topics.apply(lambda x: max(x,key=itemgetter(1))[0])
df['topic_max_val'] = df.topics.apply(lambda x: max(x,key=itemgetter(1))[1])
df[df.source=='so'].sort_values('topic_max_val')
topicwords = pd.DataFrame(ldamallet.print_topics())
topicwords.columns=['topic_max_num','words']
topicwords = topicwords.set_index('topic_max_num')
df = df.set_index('topic_max_num').join(topicwords).reset_index()

Path("words").mkdir(exist_ok=True)
wordcols=['postid','Message','topic_max_num','topic_max_val','words']
df[wordcols].to_parquet('words/words_{}.parquet'.format(KEY),
                        compression='gzip',
                        index=None)
