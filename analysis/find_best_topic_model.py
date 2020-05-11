#%%#######################################################################
#                          SETUP / Read Data                             #
##########################################################################

import pandas as pd
import numpy as np
import sys
import so_textprocessing as stp

# Read Data and Group by Forum Topic ID
df_kg = pd.read_csv('../data/kg_forummessages.csv')

# Group by Forum Topic
df_kg = df_kg.groupby('ForumTopicId').agg({
    'Message': lambda x: ' '.join([str(i) for i in x]),
}).reset_index()

# Remove HTML Tags
tp = stp.TextPreprocess()
df_kg = tp.transform_df(df_kg, reformat="processonly", columns=['Message'])

# Read SO
df_so = pd.read_parquet('../data/so_datascience.parquet')
df_so = df_so[df_so.uniq>=2]

#%%#######################################################################
#            Produce Info for Manual Topic Model Evaluation              #
##########################################################################

# %%
