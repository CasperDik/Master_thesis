import numpy as np
import pandas as pd
from numpy.polynomial import Laguerre

x = [1,2,3]
df = pd.DataFrame({'alpha': [],
                'B1' : [],
                'B2' : []})

df.loc[len(df.index)] = x
df.loc[len(df.index)] = x
df.loc[len(df.index)] = x