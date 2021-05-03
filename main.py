import numpy as np


# https://en.wikipedia.org/wiki/Lenstra_elliptic-curve_factorization#Algorithm
# https://stackoverflow.com/questions/31074172/elliptic-curve-point-addition-over-a-finite-field-in-python


def valid_point(x: int, y: int, a: int, b: int, n: int):
    return (pow(y, 2) % n) == ((pow(x, 3) + a * x + b) % n)


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


def inv_mod_n(n: int, x: int):
    assert (np.gcd(n, x) == 1)

    # Using fermat theorem + gcd(n, x) == 1
    return pow(x, n - 2, n)


def elliptic_inverse(n: int, x_p: int, y_p: int):
    return x_p, (-y_p) % n


def elliptic_addition(n: int, a: int, b:int, x_p: int, y_p: int, x_q: int, y_q: int):
    assert (0 <= x_p < n and 0 <= x_q < n and 0 <= y_p < n and 0 <= y_q < n)

    if x_p != x_q:
        # Compute the slope and the ordinate of the line between p and q
        u = (y_p - y_q) % n
        v = (x_p - x_q)
    elif y_p != 0:
        u = (3 * pow(x_p, 2) + a) % n
        v = (2 * y_p) % n
    else:
        # Either P, Q or P + Q is the origin which we are not interested in.
        return 2, []

    try:
        v = inv_mod_n(n, v)
    except Exception as e:
        # v is a factor of n
        return 0, [np.gcd(n, v)]

    s = (u * v) % n
    x_r = (pow(s, 2) - x_p - x_q) % n
    y_r = (y_p + s * (x_r - x_p)) % n
    assert(valid_point(x_r, y_r, a, b, n))
    return 1, [x_r, y_r]


