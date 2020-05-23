#%%#######################################################################
#                                Load Data                               #
##########################################################################
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from glob import glob
import seaborn as sns
import matplotlib
matplotlib.rcParams['font.family'] = ['Arial']

# Get Coherence
df = pd.concat([pd.read_csv(i) for i in glob('../hptuning/eval/*.csv')])
df = df[['0','1','2','3']]
df.columns=['Alpha','Number of Topics','Optimize Interval','Coherence']

# Get LL/Token Scores from Models
param = 'training MALLET LDA with'
value = '<1000> LL/token:'
lltokens = []
for f in glob('../hptuning/outputs/*.err'):
    info = [ line for line in open(f) if param in line or value in line ]
    iparams = info[0].split()[13:19]
    iscore = np.exp(-float(info[1].split()[2]))
    addinfo = [iparams[1],iparams[3],iparams[5],iscore]
    addinfo = [float(i) for i in addinfo]
    lltokens += [addinfo]
columns=['Number of Topics','Alpha','Optimize Interval', 'Perplexity']
lltokendf = pd.DataFrame(lltokens, columns=columns)

# Join scores
df = df.set_index(['Alpha','Number of Topics','Optimize Interval']).join(
        lltokendf.set_index(['Alpha','Number of Topics','Optimize Interval'])
    ).reset_index()

# Clean workspace
del addinfo,columns,f,info,iparams,iscore,lltokendf,param,value,lltokens

#%%#######################################################################
#                             Produce Plot                               #
##########################################################################

# Produce plot
fig, ax = plt.subplots(figsize=(7,4))

def lineplot(df, index, cols, ax):
    clrs = sns.color_palette("hls", 4)
    for clr, col in zip(clrs, cols):
        ax.plot(index, col, data=df, markersize=5, 
                linewidth=2, marker='o', color='#000080')
    plt.grid(True, alpha=0.3)
    plt.xticks(fontsize=15)
    plt.xticks(np.arange(5, 31, 2))
    plt.xlabel('Number of Topics', fontsize=17)
    plt.yticks(fontsize=15)
    plt.ylabel('Coherence', fontsize=17)

lineplot(df.groupby('Number of Topics').max().reset_index(), 
         'Number of Topics', 
         ['Coherence'], 
         ax)

plt.savefig('../outputs/coherence.png', 
            bbox_inches="tight", 
            dpi=300,
            format='png')
