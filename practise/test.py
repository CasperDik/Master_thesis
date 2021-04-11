import numpy as np
import pandas as pd

df = pd.read_csv(r'C:\Users\Casper Dik\Downloads\S&P 500 Historical Data.csv')

df = df.to_numpy(dtype=float)

print(np.std(df))

