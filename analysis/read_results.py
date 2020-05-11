# %% Setup
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.family'] = ['Times New Roman']

# Read StackOverflow
path = '/home/david/Documents/topicmodelling/'
df = pd.concat([pd.read_csv(i) for i in glob(path+'lda_plots/*.csv')])
df = df[df["3"]==2][['0','2','3','4']].pivot('0', '4', '2').reset_index()
df.columns = ['Topic Number', 'Gensim SO', 'Mallet SO']
df_so = df.set_index("Topic Number").copy()

# Read Kaggle
path = '/home/david/Documents/topicmodelling/kaggle_topic_modelling/'
df = pd.concat([pd.read_csv(i) for i in glob(path+'lda_plots/*.csv')])
df = df[df["3"]==1][['0','2','3','4']].pivot('0', '4', '2').reset_index()
df.columns = ['Topic Number', 'Gensim KG', 'Mallet KG']
df_kaggle = df.set_index("Topic Number").copy()

# Combined
df_comb = pd.concat([df_so,df_kaggle], axis=1).reset_index()

# Produce size by side plot
fig, ax = plt.subplots(figsize=(7,4))

def lineplot(df, index, cols, ax):
    clrs = sns.color_palette("hls", 4)
    for clr, col in zip(clrs, cols):
        ax.plot(index, col, data=df, markersize=5, linewidth=2, color=clr)
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=15)
    plt.xticks(np.arange(5, 31, 2))
    plt.xlabel('Number of Topics', fontsize=17)
    plt.yticks(fontsize=15)
    plt.ylabel('Coherence', fontsize=17)
    plt.legend(bbox_to_anchor=(-0.12, -0.35, 1.12, .102), 
               loc=3, ncol=4, 
               mode="expand", borderaxespad=0, prop={'size': 13})

lineplot(df_comb, 
         'Topic Number', 
         ['Gensim SO', 'Mallet SO', 'Gensim KG', 'Mallet KG'], 
         ax)

plt.savefig('coherence.svg', 
            bbox_inches="tight", 
            dpi=300,
            format='svg')
