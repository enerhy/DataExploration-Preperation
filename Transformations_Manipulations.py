'''---#Important commands / data transformations---''' 
import numpy as np
import pandas as pd

arr = np.ones(shape = (40, 1), dtype = 'int')
arr
arr2 = np.ones(shape = (40, 1)).astype(int)

zeros = np.zeros(shape = (40 , 2), dtype='int')
X
X_transformed = np.append(arr= np.ones((40, 1)).astype(int), values = zeros , axis = 1)
X_train_with_bo = np.append(arr = np.ones((40, 1)).astype(int), values = zeros , axis = 1)




