# Problem: Find Captain Redbeard's Hidden Treasure
# URL: https://www.deep-ml.com/problems/127

def find_treasure(start_x: float) -> float:
    """
    Find the x-coordinate where f(x) = x^4 - 3x^3 + 2 is minimized.

  Returns:
        float: The x-coordinate of the minimum point.
    """
    h = 0.1
    x = start_x
    for _ in range(100000):
      grad = 4 * x ** 3 - 9 * x ** 2
      new_x = x - h * grad
      new_grad = 4 * new_x**3 - 9 * new_x**2
      if (abs(new_x - x) < 1e-5) and (grad * new_grad < 0):
        break
      x = new_x
      h *= 0.99

    return x


'''
Notes
- If derivative will be zero, don't write it like f'(x) = 0, but like abs(f) < 1e-5
- You have to think of inflection points, where the second derivative is zero, to find the minimum point.
- You have to start with big h, and then decrease it to find the minimum point more accurately.
'''
