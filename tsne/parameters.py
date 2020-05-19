# %%
import numpy as np
import pandas as pd
from itertools import product

PERPLEX = [30,50,80,150]
N_ITER = [1500]
LRATE = [500,1000]
SAMPLE = [197830]

combs = [i for i in product(PERPLEX,N_ITER,LRATE,SAMPLE)]
combdf = pd.DataFrame(combs)
combdf.to_csv('input.csv',index=None,header=None)

# %%
