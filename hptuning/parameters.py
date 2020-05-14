# %%
import numpy as np
import pandas as pd
from itertools import product

ALPHA = [0.01, 5, 10, 50]
NTOPICS = range(5,31)
OPTINT = [0, 20, 50]

combs = [i for i in product(ALPHA,NTOPICS,OPTINT)]
combdf = pd.DataFrame(combs)
combdf.to_csv('input.csv',index=None,header=None)
