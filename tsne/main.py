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
matplotlib.rcParams['font.family'] = ['Arial']
import logging
logging.basicConfig(level=logging.INFO)
import plotly.express as px

print("Reading Data")
df = pd.read_parquet('../data/post_topics_final.parquet')
df['topic'] = df.topics.apply(lambda x: max(x,key=itemgetter(1))[0])
df['val'] = df.topics.apply(lambda x: max(x,key=itemgetter(1))[1])
df = df.set_index('topic').join(pd.read_csv('../data/labels.csv').set_index('topic'))
df = df.reset_index()
df['features'] = [[i[1] for i in j] for j in df.topics]

try:
    PERPLEX = int(sys.argv[1])
    N_ITER = int(sys.argv[2])
    LRATE = int(sys.argv[3])
    SAMPLE = int(sys.argv[4])
except:
    PERPLEX = 45
    N_ITER = 1000
    LRATE = 200
    SAMPLE = 10000

KEY = "{}_{}_{}_{}".format(PERPLEX,N_ITER,LRATE,SAMPLE)

#%%#######################################################################
#                                   Counts                               #
##########################################################################
# df[['postid','source']] = df.key.str.split('_',expand=True)

#%%#######################################################################
#                                   TSNE                                 #
##########################################################################
Path("pngs").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)
Path("pngs/{}".format(SAMPLE)).mkdir(parents=True,exist_ok=True)
Path("data/{}".format(SAMPLE)).mkdir(parents=True,exist_ok=True)
colors = px.colors.qualitative.Dark24

def gettsne(df):
    # Calc TSNE
    tsne = TSNE(n_components=2, 
                verbose=3, 
                perplexity=PERPLEX, 
                n_iter=N_ITER,
                learning_rate=LRATE,
                early_exaggeration=4,
                n_jobs=8,
                angle=0.07,
                init='pca')
    tsne_results = tsne.fit_transform(df.features.tolist())
    tsnedf = pd.DataFrame(tsne_results)
    tsnedf.columns=['tsne-2d-one','tsne-2d-two']
    tsnedf['Topics'] = df.topic.tolist()
    tsnedf.to_parquet('data/{}/{}.parquet'.format(SAMPLE,KEY),
                      compression='gzip',
                      index=None)
    return tsnedf

def plottsne(tsnedf):
    # Create axis
    _, ax = plt.subplots(figsize=(35,16))
        
    for i in range(24):
        sns.scatterplot(
            x="tsne-2d-one", y="tsne-2d-two",
            # hue="Topics",
            color=colors[i],
            data=tsnedf[tsnedf.Topics==i],
            s=240 if i < 10 else 380,
            alpha=0.7,
            marker=r"$ {} $".format(i),
            linewidth=0,
            label=str(i),
            ax=ax
        )
        
    # Set ticks
    plt.xticks(ticks=[], fontsize=35)
    plt.xlabel('', fontsize=45)
    plt.yticks(ticks=[], fontsize=35)
    plt.ylabel('', fontsize=45)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles[0:], 
                labels=['T{}'.format(i) for i in labels],
                bbox_to_anchor=(0, -0.15, 1, .102), 
                loc=3, 
                ncol=12, 
                mode="expand", 
                borderaxespad=0,
                handletextpad=0, 
                prop={'size': 35})
    plt.savefig('pngs/{}/{}.png'.format(SAMPLE,KEY), 
            bbox_inches="tight", 
            dpi=300,
            format='png')
    plt.savefig('../outputs/TSNE.png', 
            bbox_inches="tight", 
            dpi=300,
            format='png')

# %%
print('Starting TSNE')
calc_tsne = gettsne(df.sample(SAMPLE,random_state=43))

# %%
plottsne(calc_tsne)
sys.exit()

# %%
plottsne(pd.read_parquet('data/20000/30_1000_200_10000.parquet'))



# %%
