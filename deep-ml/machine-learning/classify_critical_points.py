# Problem: Classify Critical Points Using Hessian Eigenvalues
# URL: https://www.deep-ml.com/problems/311

import numpy as np

def classify_critical_point(hessian: np.ndarray, tol: float = 1e-10):
    eigenvalues = np.linalg.eigvals(hessian)
    if np.all(eigenvalues > tol):
        return -1
    elif np.all(eigenvalues < -tol):
        return 1
    elif np.any((eigenvalues < tol) & (eigenvalues > -tol)):
        return None
    else:
        return 0

'''
Notes
- If you want to find eigenvalues of a matrix, you can use np.linalg.eigvals() function
- Use np.all() to check if all elements of an array satisfy a condition
- Use np.any() to check if any element of an array satisfies a condition
- To compare elements in array, use & not and for comparison
'''