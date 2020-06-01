#%%#######################################################################
#                                   SETUP                                #
##########################################################################
import pandas as pd
from glob import glob
from operator import itemgetter
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys
import matplotlib
matplotlib.rcParams['font.family'] = ['Arial']
import logging
logging.basicConfig(level=logging.INFO)
import plotly.express as px
from natsort import natsorted, index_natsorted, order_by_index
import matplotlib.colors as mc
from collections import Counter
from tqdm import tqdm
tqdm.pandas()
import plotly.express as px
from matplotlib.lines import Line2D

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
def hist(df, outputname='counts.png', figh=35, figw=34):
    
    # Create axis
    _, ax = plt.subplots(figsize=(figh,figw))
        
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
    plt.savefig('../outputs/{}'.format(outputname), 
            bbox_inches="tight", 
            dpi=300,
            format='png')

counts = df.groupby(['label','source']).count()[['key']].reset_index()
counts.source = counts.source.apply(lambda x: 'Kaggle' if x=='kg'
                                    else 'StackOverflow')
total_kg, total_so = df.groupby('source').count().topic.tolist()
counts['counts'] = counts.key
counts['key'] = counts.apply(lambda x: x.key/total_kg if x.source=='Kaggle' 
              else x.key/total_so, axis=1)
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

#%%#######################################################################
#                           Count ML Algorithms                          #
##########################################################################

df_lang = pd.read_parquet('../data/df_lang.parquet')
df_lang = df_lang[df_lang.lang=='en']
df_lang['key'] = df_lang.postid.astype(str) +'_'+ df_lang.source
df_lang = df_lang.set_index('key')[['Message']].copy()
df = df.set_index('key').join(df_lang).dropna().reset_index()

# %% Prepare Data
import so_textprocessing as stp
tp = stp.TextPreprocess(strlist='../data/mlalgs.csv')
df_wordmatch = df.rename(columns={'Message':'answers'})
df_wordmatch.answers = df_wordmatch.answers.progress_apply(tp.stem)
df_wordmatch = df_wordmatch.reindex(columns=[*df_wordmatch.columns.tolist(), 
                                             'tags', 'title', 'question'], 
           fill_value=' ')
df_wordmatch.to_parquet('../data/df_wordmatch.parquet',
                        compression='gzip',
                        index=None)

# %% Match Words
tp = stp.TextPreprocess(strlist='../data/mlalgs.csv')
wordmatch = tp.transform_df(df_wordmatch, columns=['Message'])

# %% Final Word Counts
wordcounts = wordmatch[['key','source','words']].copy()
wordcounts = wordcounts[wordcounts.words!='']
wordcounts.words = wordcounts.words.str.split('|')
wordcounts = wordcounts.explode('words')
mlDict = pd.read_csv('../data/mlalgsdict.csv').set_index('str')
wordcounts = wordcounts.set_index('words').join(mlDict).reset_index(drop=1)
wordcounts = wordcounts.drop_duplicates()
sokgcount = Counter(wordcounts.source)
wordcounts = wordcounts.groupby(['label','source']).count().reset_index()
wordcounts['key'] = wordcounts.apply(lambda x: x.key/sokgcount[x.source],axis=1)
wordcounts.source = wordcounts.source.apply(lambda x: 'Kaggle' if x=='kg'
                                            else 'StackOverflow')
wordcounts.key *= 100
hist(wordcounts, outputname='algorithmCount.png', figh=15)
# In total, there are Counter({'so': 45411, 'kg': 22599}) string matches
# Of ML algorithms. Duplicate mentions of a ML algorithm in a post (e.g.
# CNN and convolutional neural network) are treated as one, i.e. not counted
# more than once per post. 

#%%#######################################################################
#                          Join dates SO/Kaggle                          #
##########################################################################
dates = pd.concat([pd.read_csv(i) for i in glob('../data/sodate*')])
dates.id = dates.id.astype('str')
dates = dates.rename(columns={'id':'postid'})
dates = dates.set_index('postid')
dates.creationdate = dates.creationdate.apply(lambda x: x.split('-')[0])

kgdates = pd.read_csv('../data/kgdates.csv')
kgdates.postid = kgdates.postid.astype('str')
kgdates = kgdates.set_index('postid')

alldates = pd.concat([dates,kgdates])
alldates.creationdate = alldates.creationdate.astype(int)

dfdates = df.set_index('postid').join(alldates).dropna()
dfdates = dfdates[dfdates.creationdate!=2020]

# %% Data
plotyear = dfdates.groupby(['creationdate','label','source'])\
                  .count()\
                  .reset_index()
plotyear = plotyear.sort_values(['creationdate'])
plotyear = plotyear.reindex(index=order_by_index(plotyear.index, 
                                                 index_natsorted(plotyear.label)))

# %% Lineplot
sns.set(style="white", rc={"lines.linewidth": 6})

def lineplot(plotyear, outputname):
    _, ax = plt.subplots(figsize=(35,35))
    sns.lineplot(x="creationdate", y="key",
                    hue="label", 
                    data=plotyear, 
                    palette=px.colors.qualitative.Dark24,
                    ax=ax)
    linestyles = ['solid','dotted','dashed','dashdot']
    custom_lines = []
    for l in range(24):
        ax.lines[l].set_linestyle(linestyles[l % len(linestyles)])
        custom_lines.append(Line2D([0], [0], 
                                   color=px.colors.qualitative.Dark24[l], 
                                   lw=8, 
                                   ls=linestyles[l % len(linestyles)]))

    plt.xticks(fontsize=50)
    plt.xlabel('')
    plt.yticks(fontsize=50)
    plt.ylabel('')
    
    ax.legend(custom_lines, 
              plotyear.label.drop_duplicates().tolist(), 
              fontsize=40)
    plt.savefig('../outputs/{}'.format(outputname), 
            bbox_inches="tight", 
            dpi=300,
            format='png')

lineplot(plotyear[plotyear.source=='so'], 'so_topics_over_time.png')
lineplot(plotyear[plotyear.source=='kg'], 'kg_topics_over_time.png')


 # %%
