# Problem: Taylor Series Approximation
# URL: https://www.deep-ml.com/problems/310

import numpy as np
from math import factorial

import numpy as np
from math import factorial

def taylor_approximation(func_name: str, x: float, n_terms: int) -> float:
    """
    Compute Taylor series approximation for common functions.
    
    Args:
        func_name: Name of function ('exp', 'sin', 'cos')
        x: Point at which to evaluate
        n_terms: Number of terms in the series
    
    Returns:
        Taylor series approximation rounded to 6 decimal places
    """
    ans = 0
    for i in range(n_terms):
        if func_name == 'exp':
            temp = (x**i) / factorial(i)
            ans += temp
        if func_name == 'sin':
            num = i * 2 + 1
            temp = (x**num) * ((-1) ** i) / factorial(num)
            ans += temp
        if func_name == 'cos':
            num = i * 2
            temp = (x**num) * ((-1) ** i) / factorial(num)
            ans += temp

    return ans

'''
Notes
- If you write an expression like -1 ** i, the exponentiation will be evaluated before the negation, so it will be interpreted as -(1 ** i)
'''
