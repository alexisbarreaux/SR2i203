import numpy as np
from math import factorial as fact

# https://en.wikipedia.org/wiki/Lenstra_elliptic-curve_factorization#Algorithm
# https://stackoverflow.com/questions/31074172/elliptic-curve-point-addition-over-a-finite-field-in-python


def valid_point(x: int, y: int, a: int, b: int, n: int):
    """
    Checks if a points is on an elliptic curve given by parameters.
    :param x: abciss
    :param y: ordinate
    :param a: first parameter of the curve
    :param b: second parameter
    :param n: modulus parameter.
    :return:
    """
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
    """
    Returns the inverse of x modulus n.
    """
    assert (np.gcd(n, x) == 1)

    # To compute the inverse of
    return pow(x, -1, n)


"""
def elliptic_inverse(n: int, x_p: int, y_p: int):
    return x_p, (-y_p) % n
"""


def elliptic_addition(n: int, a: int, b: int, x_p: int, y_p: int, x_q: int = -1, y_q: int = -1):
    """
    Returns the sum of P = (x_p, y_p) and Q = (x_q, y_q) on a curve. If Q is not given, returns 2P.
    """
    if x_q < 0:
        x_q = x_p
    if y_q < 0:
        y_q = y_p

    assert (0 <= x_p < n and 0 <= x_q < n and 0 <= y_p < n and 0 <= y_q < n)

    if x_p != x_q:
        # Compute the slope and the ordinate of the line between p and q
        u = (y_p - y_q) % n
        v = (x_p - x_q)
    elif y_p != 0:
        u = (3 * pow(x_p, 2, n) + a) % n
        v = (2 * y_p)
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
    # This y is the opposite of what we want.
    y_r = (y_p + s * (x_r - x_p)) % n
    y_r = (-y_r) % n

    assert (valid_point(x_r, y_r, a, b, n))
    assert (valid_point(x_r, -y_r, a, b, n))

    return 1, [x_r, y_r]


# This function is working but might not be needed since calculating k * P this way may go over a case where
# j * P returns an error and a factor of n, with j < k and overlooked by this method.
def elliptic_multiplication(n: int, a: int, b: int, x: int, y: int, k: int):
    """
    Method inspired of quick exponentiation used to compute k times the point P = (x, y) on an elliptic curve.
    """
    assert(type(k) == int)

    if k == 0:
        return 2, []
    elif k == 1:
        # We return P
        return 1, [x, y]
    elif k == 2:
        # We return the result of P + P
        return elliptic_addition(n, a, b, x, y)

    else:
        if k % 2 == 0:
            # We want to return k//2 * P + k//2 * P
            ret_code, l = elliptic_multiplication(n, a, b, x, y, k//2)

            # However we must check if k//2 * P is indeed a true point or not.
            if ret_code == 1:
                x2, y2 = l
                return elliptic_addition(n, a, b, x2, y2)
            else:
                return ret_code, l

        else:
            # We want to return P * k//2 * P + k//2 * P
            ret_code, l = elliptic_multiplication(n, a, b, x, y, (k - 1) // 2)
            if ret_code == 1:
                x2, y2 = l
                ret_code2, l2 = elliptic_addition(n, a, b, x2, y2)
                if ret_code2 == 1:
                    x3, y3 = l2
                    return elliptic_addition(n, a, b, x, y, x3, y3)
                else:
                    return ret_code2, l2
            else:
                return ret_code, l


def lenstra(n: int):
    """
    Method resorting to elliptic curves to try and find a factor of n.
    """
    a, b, x, y = create_random_elliptic(n)
    k = fact(20)
    ret_code, res = elliptic_multiplication(n, a, b, x, y, k)
    if ret_code == 0:
        return res
    else:
        return lenstra(n)


if __name__ == "__main__":
    print(lenstra(7919 * 7321))
