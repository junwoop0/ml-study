# Problem: Newton's Method for Optimization
# URL: https://www.deep-ml.com/problems/221

from typing import Callable
import numpy as np

def newtons_method_optimization(
	gradient_func: Callable[[list[float]], list[float]],
	hessian_func: Callable[[list[float]], list[list[float]]],
	x0: list[float],
	tol: float = 1e-6,
	max_iter: int = 100
) -> list[float]:
	"""
	Find the minimum of a function using Newton's method.
	
	Args:
		gradient_func: Function that returns gradient vector at a point
		hessian_func: Function that returns Hessian matrix at a point
		x0: Initial guess (list of coordinates)
		tol: Convergence tolerance for gradient norm
		max_iter: Maximum number of iterations
		
	Returns:
		The point that minimizes the function
	"""
	x = np.array(x0)
	for i in range(max_iter):
		hessian = np.linalg.inv(np.array(hessian_func(x)))
		gradient = np.array(gradient_func(x))
		x_new = x - hessian @ gradient
		if np.linalg.norm(x_new - x) < tol:
			break
		x = x_new
	return x.tolist()

'''
Notes
- Convergence toleranance: A small positive value that determines when to stop the iterations.
- gradient_func: Callable[[list[float]], list[float]] means that the function will take a list of floats as input 
and return a list of floats (the gradient).
- To calculate the inverse of the matrix, you can use np.linalg.inv() function.
- How np.linalg.solve(H, g) works:
    - It solves the linear equation H * x = g for x
'''