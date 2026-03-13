# Problem: Compute the Hessian Matrix
# URL: https://www.deep-ml.com/problems/218

from typing import Callable
import numpy as np

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Hessian matrix of function f at the given point using finite differences.
	
	Args:
		f: A scalar function that takes a list of floats and returns a float
		point: The point at which to compute the Hessian (list of coordinates)
		h: Step size for finite differences (default: 1e-5)
		
	Returns:
		The Hessian matrix as a list of lists (n x n where n = len(point))
	"""
	# Your code here
	size = len(point)
	x = point.copy()
	arr = np.zeros((size, size), dtype = float)
	for i in range(len(point)):
		for j in range (len(point)):
			if (i == j):
				x_plus = x.copy()
				x_plus[i] += h
				x_minus = x.copy()
				x_minus[i] -= h
				arr[i][j] = (f(x_plus) - 2*f(x) + f(x_minus)) / h**2

			if (i != j):
				all_plus = x.copy()
				all_plus[i] += h
				all_plus[j] += h
				plus_minus = x.copy()
				plus_minus[i] += h
				plus_minus[j] -= h
				minus_plus = x.copy()
				minus_plus[i] -= h
				minus_plus[j] += h
				all_minus = x.copy()
				all_minus[i] -= h
				all_minus[j] -= h
				arr[i][j] = (f(all_plus) - f(plus_minus) - f(minus_plus) + f(all_minus)) / (4*h**2)

	return arr


'''
Notes
- f: Callable[[list[float]], float] means f is a function that takes a list of floats and returns a float
- since diagnal elements and non-diagonal elements are calculated differently, we can make each case separately
- For finite difference method in mixed partial derivatives, we can use the formula:
f(x+h, y+h) - f(x+h, y-h) - f(x-h, y+h) + f(x-h, y-h) / (4*h^2)
- For second derivative with respect to the same variable, we can use the formula:
f(x+h) - 2*f(x) + f(x-h) / h^2
'''