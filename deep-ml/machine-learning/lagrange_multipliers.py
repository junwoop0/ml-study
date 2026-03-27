# Problem: Lagrange Multipliers for Constrained Quadratic Optimization
# URL: https://www.deep-ml.com/problems/314

import numpy as np

def lagrange_optimize(Q: np.ndarray, c: np.ndarray, a: np.ndarray, b: float) -> dict:
    """
    Solve constrained quadratic optimization using Lagrange multipliers.
    
    Minimize: f(x) = (1/2) x^T Q x + c^T x
    Subject to: a^T x = b
    
    Args:
        Q: 2x2 symmetric positive definite matrix
        c: 2-element vector (linear coefficients)
        a: 2-element vector (constraint coefficients)
        b: scalar (constraint value)
    
    Returns:
        Dictionary with 'x', 'lambda', and 'objective' keys
    """
    z = np.zeros((1,1))
    a = a.reshape(2,1)
    c = c.reshape(2,1)
    b = np.array([b])
    b = b.reshape(1,1)
    left = np.block([
        [Q, -a],
        [a.T, z]
    ])
    right = np.block([
        [-c],
        [b]
    ])
    ans = np.linalg.solve(left, right)
    x = ans[0:2, :]
    lamb = ans[-1, :]
    x = x.flatten()
    lamb = float(lamb[0])
    result = (0.5)* x.T @ Q @ x + c.T @ x
    obj = float(result[0])
    x = np.round(x, 4)
    dic = {
        "x": x.tolist(),
        "lambda": round(lamb, 4),
        "objective": round(obj, 4),
    }

    return dic

'''
Notes
- You can make big matrix including blocks by using np.block() function
- Always think of reshaping the vectors to be column vectors (n, 1) when doing matrix operations
- To make two-deminsional array into one-dimensional array, you can use .flatten() method
- You have to round to decimal places after calculating the objective value.
'''
