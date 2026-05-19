# Problem: Batch Iterator for Dataset
# URL: https://www.deep-ml.com/problems/30

import numpy as np

def batch_iterator(X, y=None, batch_size=64):
	ans = []
	for i in range(0, X.shape[0], batch_size):
		end = i + batch_size
		X_batch = X[i:end]
		if y is not None:
			y_batch = y[i:end]
			ans.append([X_batch, y_batch])
		else:
			ans.append(X_batch)
	return ans

'''
Notes
- When you append to a list, you can append a list as an element.
- Even if the end variable exceeds the length of the array, it will not cause an error. The slicing will just return the remaining elements.
'''