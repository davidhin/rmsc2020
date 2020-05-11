# %%
import pandas as pd
import numpy as np
import sys
import so_textprocessing as stp

# Read Data and Group by Forum Topic ID
df_raw = pd.read_csv('kaggle_topic_modelling/ForumMessages.csv')

# Group by Forum Topic
df = df_raw.groupby('ForumTopicId').agg({ 
        'Message': lambda x: ' '.join([str(i) for i in x]),
    }).reset_index()

# Remove HTML Tags
tp = stp.TextPreprocess()
df = tp.transform_df(df, reformat="processonly", columns=['Message'])

# %%
df

# %%
