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

'''
ChatGPT Solution -1
def compute_hessian(
    f: Callable[[list[float]], float],
    point: list[float],
    h: float = 1e-5
) -> list[list[float]]:
    """
    Compute the Hessian matrix of a scalar function f at a given point using
    central finite differences.

    Args:
        f: Scalar function mapping R^n (list/array-like) -> R
        point: Evaluation point (length n)
        h: Step size for finite differences

    Returns:
        Hessian as a list of lists (n x n)
    """
    x = np.array(point, dtype=float)
    n = x.size
    H = np.zeros((n, n), dtype=float)

    f0 = float(f(x.tolist()))  # baseline f(x)

    for i in range(n):
        # Diagonal term: d^2 f / d x_i^2
        x_ip = x.copy(); x_ip[i] += h
        x_im = x.copy(); x_im[i] -= h
        f_ip = float(f(x_ip.tolist()))
        f_im = float(f(x_im.tolist()))
        H[i, i] = (f_ip - 2.0 * f0 + f_im) / (h * h)

        # Off-diagonal terms: d^2 f / (d x_i d x_j), i < j
        for j in range(i + 1, n):
            x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
            x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h
            x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
            x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h

            f_pp = float(f(x_pp.tolist()))
            f_pm = float(f(x_pm.tolist()))
            f_mp = float(f(x_mp.tolist()))
            f_mm = float(f(x_mm.tolist()))

            val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h)
            H[i, j] = val
            H[j, i] = val  # Hessian is symmetric (for sufficiently smooth f)

    return H.tolist()

'''