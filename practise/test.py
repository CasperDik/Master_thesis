import numpy as np
import pandas as pd

x = np.array([[12],[13],[14]])
y = np.array([[2],[3],[4]])
z = np.transpose(np.hstack((x,y)))

ints = [1,2]
string_ints = [str(int) for int in ints]

df = pd.DataFrame(data=z, index=string_ints, columns=["GBM", "MR2", "MR3"])
