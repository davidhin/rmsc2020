#%%#######################################################################
#                                   SETUP                                #
##########################################################################
import pandas as pd
from operator import itemgetter
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys
import matplotlib
matplotlib.rcParams['font.family'] = ['Times New Roman']
import logging
logging.basicConfig(level=logging.INFO)
import plotly.express as px
from natsort import natsorted, index_natsorted, order_by_index
import matplotlib.colors as mc

print("Reading Data")
df = pd.read_parquet('../data/post_topics_final.parquet')
df['topic'] = df.topics.apply(lambda x: max(x,key=itemgetter(1))[0])
df['val'] = df.topics.apply(lambda x: max(x,key=itemgetter(1))[1])
labeldf = pd.read_csv('../data/labels.csv').set_index('topic')
df = df.set_index('topic').join(labeldf)
df = df.reset_index()
df[['postid','source']] = df.key.str.split('_',expand=True)

#%%#######################################################################
#                                   Counts                               #
##########################################################################
def hist(df):
    
    # Create axis
    _, ax = plt.subplots(figsize=(35,34))
        
    sns.barplot(x="key", y="label", 
                hue="source", 
                data=df,
                palette=['#acacdf', '#000080'], 
                ax=ax)
        
    # Set ticks
    plt.xticks(fontsize=50)
    plt.xlabel('')
    plt.yticks(fontsize=50)
    plt.ylabel('')

    plt.legend(fontsize=50)
    plt.savefig('../outputs/counts.png', 
            bbox_inches="tight", 
            dpi=300,
            format='png')

counts = df.groupby(['label','source']).count()[['key']].reset_index()
counts.source = counts.source.apply(lambda x: 'Kaggle' if x=='kg'
                                    else 'StackOverflow')
counts = counts.reindex(index=order_by_index(counts.index, index_natsorted(counts.label)))
hist(counts)

#%%#######################################################################
#                           Topic Relationships                          #
##########################################################################
str_contains = 'kg'

gephi = df[['key','topics']]
gephi = gephi.explode('topics')
gephi['topic'] = gephi.topics.apply(lambda x: x[0])
gephi['value'] = gephi.topics.apply(lambda x: x[1])
gephi = gephi.drop(columns=['topics'])
gephi = gephi[gephi.value>0.15]
nodes = pd.DataFrame(set(gephi.topic.tolist() + gephi.key.tolist()), columns=['Id'])
nodes['Color'] = nodes.Id
nodes['Color'] = nodes.Color.apply(lambda x: int(x) if len(str(x)) < 3 else None)
gephi.columns=['Source','Target','Weight']
gephi.to_csv('conns.csv', index=0)
nodes.to_csv('nodes.csv', index=0)
gephi = gephi[gephi.Source.str.contains(str_contains)]

upset = gephi.sort_values('Weight',ascending=0)\
        .groupby(['Source'])\
        .head(2)\
        .groupby('Source')\
        .agg({'Target':lambda x: list(x), 'Weight':'count'})\
        .sort_values('Weight')\

upset = upset[upset.apply(lambda x: len(x.Target) > 1,axis=1)]
upset.Target = upset.Target.apply(lambda x: '_'.join(natsorted(['T'+str(i) for i in x])))
upset = upset.groupby('Target').count().reset_index()
upset[['Tx','Ty']] = upset.Target.str.split('_',expand=True)

heatmap = upset.pivot(index='Ty', columns='Tx', values='Weight').fillna(0)
heatmap = heatmap[natsorted(heatmap.columns)].copy()
heatmap = heatmap.loc[natsorted(heatmap.index)]

mask = np.zeros_like(heatmap, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
for i in range(23): mask[i][i] = False
matplotlib.rcParams['font.family'] = ['Arial']

# %% Heatmap

def NonLinCdict(steps, hexcol_array):
    cdict = {'red': (), 'green': (), 'blue': ()}
    for s, hexcol in zip(steps, hexcol_array):
        rgb =matplotlib.colors.hex2color(hexcol)
        cdict['red'] = cdict['red'] + ((s, rgb[0], rgb[0]),)
        cdict['green'] = cdict['green'] + ((s, rgb[1], rgb[1]),)
        cdict['blue'] = cdict['blue'] + ((s, rgb[2], rgb[2]),)
    return cdict

hc = ['#f5f5ff', '#e5e5ff', '#ebebff', '#acacdf', '#7272bf', '#39399f', '#000080']
th = [0, 0.03, 0.2, 0.4, 0.7, 0.8, 1]
cdict = NonLinCdict(th, hc)
cm = mc.LinearSegmentedColormap('test', cdict)

## Heat map for Spec Expertise
def plot_heatmap(df):
    sns.axes_style("darkgrid")
    plt.rcParams['axes.ymargin'] = 0
    plt.rcParams['axes.xmargin'] = 0
    _, ax = plt.subplots(figsize=(31,25))
    sns.set(font_scale=3.8, style="white")
    sns.heatmap(df, square=True, 
                ax=ax, 
                mask=mask,
                linewidths=2,
                cmap=cm,
                vmax=4381)
    ax.set_xlabel('', fontsize=31, fontweight='bold')
    ax.xaxis.labelpad = 20
    ax.set_ylabel('', fontsize=21)
    ax.set_title('', fontsize=18)
    ax.tick_params(labelsize=40)
    ax.tick_params(axis='y',rotation=0)
    ax.tick_params(axis='x',rotation=45)
    plt.savefig('../outputs/heatmap_{}.png'.format(str_contains), 
        bbox_inches="tight", 
        dpi=300,
        format='png')

plot_heatmap(heatmap)
