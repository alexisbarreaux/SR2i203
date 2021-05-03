import numpy as np

# https://en.wikipedia.org/wiki/Lenstra_elliptic-curve_factorization#Algorithm


def create_random_point(n: int):
    """
    Returns randomly chosen coordinates in the Z/nZ space.
    :param n: the order of the Z/nZ space.
    :return: A tuple of two int values.
    """
    return np.random.randint(n), np.random.randint(n)


def create_random_elliptic(n: int):
    """
    Returns the equation of a random elliptic curve and a non trivial point on this curve.
    The equation is of the form : y^2 = x^3 + ax + b (mod n).
    The point is a tuple (x_o, y_o) such that b=y_{0}^{2}-x_{0}^{3}-ax_{0} (mod n)

    :param n: the order of the Z/nZ space in which the curve is.
    :return: A list consisting of 4 int values which are respectively a, b, x_{0} and y_{0}
    """
    x_0, y_0 = create_random_point(n)
    a = np.random.randint(n)
    b = (pow(y_0, 2, n) - pow(x_0, 3, n) - a * x_0) % n
    return [a, b, x_0, y_0]

