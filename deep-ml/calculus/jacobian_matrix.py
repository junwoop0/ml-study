# Problem: Jacobian Matrix Calculation
# URL: https://www.deep-ml.com/problems/202

import numpy as np

def jacobian_matrix(f, x: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Jacobian matrix using numerical differentiation.
	
	Args:
		f: Function that takes a list and returns a list
		x: Point at which to evaluate the Jacobian
		h: Step size for finite differences
	
	Returns:
		Jacobian matrix as list of lists
	"""
	column_len = len(x)
	row_len = len(f(x))
	arr = np.zeros((row_len, column_len), dtype = float)
	for i in range(len(arr)):
		for j in range (len(arr[0])):
			x_plus = x.copy()
			x_plus[j] += 1e-5
			arr[i][j] = (f(x_plus)[i] - f(x)[i])/(1e-5)

	return arr

'''
note
- To make 2d array using NumPy, use np.zeros
    - J = np.zeros((m,n), dtype = float)
- When the function isn’t given (or can’t be expressed) as a coefficient list, 
use numerical differentiation (don't think of numpy libraries for too long!)
- h is set to 1e-5 even if I don't define it (it's a default argument)
- I don't need to specify the rows
    - use J(:, j) to specify the j-th column, see chatgpt solution
'''