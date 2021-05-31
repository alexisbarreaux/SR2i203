import numpy as np
import matplotlib.pyplot as plt
import sys
from math import factorial as fact
from time import time

# https://en.wikipedia.org/wiki/Lenstra_elliptic-curve_factorization#Algorithm
# https://stackoverflow.com/questions/31074172/elliptic-curve-point-addition-over-a-finite-field-in-python

sys.setrecursionlimit(10000)


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


# TODO use other specific types of curves such as Edward curves ?
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

    if 4 * pow(a, 3, n) + 27 * pow(b, 2, n) == 0:
        return create_random_elliptic(n)

    return [a, b, x_0, y_0]


def plot_elliptic_real_curve(n: int):
    """
    Plots a random elliptic curve on R.
    :return:
    """
    # Thanks to : https://stackoverflow.com/questions/19756043/python-matplotlib-elliptic-curves
    [a, b, _, _] = create_random_elliptic(n)
    plt.clf()
    plt.title(f"Elliptic curve for a = {a} and b = {b}")

    y, x = np.ogrid[-10:10:100j, -10:10:100j]
    plt.contour(x.ravel(), y.ravel(), pow(y, 2) - pow(x, 3) - x * a - b, [0])
    plt.grid()
    plt.show()


def plot_elliptic_in_z(n: int, plot_points=False):
    """
    Plots an elliptic curve on Z/nZ, can also add points calculated on it.
    :param n: The n in Z/nZ
    :param plot_points: Boolean to know wether to plot points or not.
    """
    [a, b, x, y] = create_random_elliptic(n)
    abscisses = [x]
    ordinates = [y]

    possibilities = np.arange(n)
    for i in possibilities:
        for j in possibilities:
            if valid_point(i, j, a, b, n):
                abscisses.append(i)
                ordinates.append(j)

    plt.clf()
    plt.title(f"Elliptic curve on Z/pZ for a = {a} and b = {b} and p = {n}")
    plt.scatter(abscisses, ordinates)
    ax = plt.gca()
    plt.grid()

    if plot_points:
        sum_x = [x]
        sum_y = [y]
        ax.annotate("P", (x, y), xytext=(10, 10), textcoords='offset points')
        for i in range(6):
            ret, l = elliptic_addition(n, a, b, x, y)
            if ret == 1:
                x, y = l
                sum_x.append(x)
                sum_y.append(y)
                ax.annotate(f"{i + 2}P", (x, y), xytext=(10, 10), textcoords='offset points')

        plt.scatter(sum_x, sum_y, marker="x", c="r")

    plt.show()


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


def elliptic_multiplication(n: int, a: int, b: int, x: int, y: int, k: int):
    """
    Method used to compute k times the point P = (x, y) on an elliptic curve.
    """
    assert (type(k) == int)

    if k == 0:
        return 2, []
    elif k == 1:
        # We return P
        return 1, [x, y]
    elif k == 2:
        # We return the result of P + P
        return elliptic_addition(n, a, b, x, y)
    else:
        ret_code, l = elliptic_multiplication(n, a, b, x, y, k - 1)
        # However we must check if (k- 1) * P is indeed a true point or not.
        if ret_code == 1:
            x2, y2 = l
            return elliptic_addition(n, a, b, x, y, x2, y2)
        elif ret_code == 2:
            # At some point we encountered a sum such that we reached the infinite O, catching it as soon as it arises,
            # we ensure we just have to give back our point.
            return 1, [x, y]
        else:
            return ret_code, l


# This function is working but might not be needed since calculating k * P this way may go over a case where
# j * P returns an error and a factor of n, with j < k and overlooked by this method.
def quick_elliptic_multiplication(n: int, a: int, b: int, x: int, y: int, k: int):
    """
    Method inspired of quick exponentiation used to compute k times the point P = (x, y) on an elliptic curve.
    """
    assert (type(k) == int)

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
            ret_code, l = quick_elliptic_multiplication(n, a, b, x, y, k // 2)

            # However we must check if k//2 * P is indeed a true point or not.
            if ret_code == 1:
                x2, y2 = l
                return elliptic_addition(n, a, b, x2, y2)
            else:
                # Either the result was O and we can't do more or we found a factor and want to give it.
                return ret_code, l

        else:
            # We want to return P +  k//2 * P + k//2 * P
            ret_code, l = quick_elliptic_multiplication(n, a, b, x, y, (k - 1) // 2)
            if ret_code == 1:
                x2, y2 = l
                ret_code2, l2 = elliptic_addition(n, a, b, x2, y2)
                if ret_code2 == 1:
                    x3, y3 = l2
                    return elliptic_addition(n, a, b, x, y, x3, y3)
                else:
                    return ret_code2, l2
            elif ret_code == 2:
                # K//2 * P is O, thus our current result is just P
                return 1, [x, y]
            else:
                return ret_code, l


def lenstra(n: int, quick=True, use_loop=True, loop=1, fact_size=7, ret_loop=False):
    """
    Method resorting to elliptic curves to try and find a factor of n.
    Loop helps to ensure we don't have an infinite recursion
    """
    a, b, x, y = create_random_elliptic(n)

    if quick:
        k = fact(25)
        ret_code, res = quick_elliptic_multiplication(n, a, b, x, y, k)
    else:
        k = fact(fact_size)
        ret_code, res = elliptic_multiplication(n, a, b, x, y, k)

    if ret_code == 0:
        if ret_loop:
            return res, loop
        else:
            return res
    else:
        if use_loop and loop > 30:
            if ret_loop:
                return res, loop
            return res
        else:
            return lenstra(n, quick, use_loop, loop + 1,fact_size, ret_loop)


# Slow but useful method
def list_primes(n: int):
    """
    Function using our lenstra function to list primes.
    :param n: The maximum integer whose primality we want to test.
    :return:
    """
    a = np.arange(3, n + 1)
    primes = [2]
    for i in a:
        res = lenstra(int(i))
        if len(res) in [0, 2]:
            primes.append(i)

    return primes


def study_speed_lenstra(n: int, k: int):
    primes = list_primes(n)
    N = len(primes)
    times_slow = []
    times_quick = []
    x_slow = []
    x_quick = []

    for _ in range(k):
        a = np.random.randint(N)
        b = np.random.randint(N)
        # We don't want the same prime twice.
        if a == b:
            b = (a + 1) % len(primes)

        n = int(primes[a] * primes[b])

        t = time()
        res_slow = lenstra(n, False)
        if len(res_slow) == 1:
            times_slow.append(time() - t)
            x_slow.append(n)

        t = time()
        res_quick = lenstra(n)
        if len(res_quick) == 1:
            times_quick.append(time() - t)
            x_quick.append(n)

    res = np.array(times_slow) / np.array(times_quick)
    plt.clf()
    plt.title("Study on the time ratio \n between normal multiplication and our quick version.")
    plt.grid()
    plt.xlabel("Integer to factorize")
    plt.ylabel("Time difference")
    plt.scatter(x_slow, res, marker="x", c="b", label="Normal time / Quick time")
    plt.legend()
    plt.show()

    """
    res = np.array(times_slow) - np.array(times_quick)
    plt.clf()
    plt.title("Study on the time difference \n between normal multiplication and our quick version.")
    plt.grid()
    plt.xlabel("Integer to factorize")
    plt.ylabel("Time difference")
    plt.scatter(x_slow, res, marker="x", c="b", label="Normal time - Quick time")
    plt.legend()
    plt.show()
    """

    """
    plt.clf()
    plt.title("Study on the time efficiency \n of the normal multiplication and our quick version.")
    plt.grid()
    plt.xlabel("Integer to factorize")
    plt.ylabel("Time")
    plt.scatter(x_slow, times_slow, marker="x", c="r", label="Normal")
    plt.scatter(x_quick, times_quick, marker="+", c="b", label="Fast multiplication")
    plt.legend()
    plt.show()
    """
    print(len(x_slow), len(x_quick), k)

    return


def study_loops(n: int, k: int):
    primes = list_primes(n)
    N = len(primes)
    loops = []

    for _ in range(k):
        a = np.random.randint(N)
        b = np.random.randint(N)
        # We don't want the same prime twice.
        if a == b:
            b = (a + 1) % len(primes)

        n = int(primes[a] * primes[b])
        res_quick, loop = lenstra(n, True, True, 1, 7, True)
        if len(res_quick) == 1:
            loops.append(loop)

    loops = np.array(loops)
    counts = np.bincount(loops)
    freq = 100*(counts / len(loops))
    y = np.zeros(30)

    for i in range(len(freq)):
        y[i] = freq[i]

    x = np.arange(30)
    plt.clf()
    plt.title(f"Frequency of each loop size on {k} integers.")
    plt.grid()
    plt.xlabel("Number of loops")
    plt.ylabel("Frequency")
    plt.scatter(x, y, marker="x", c="b", label="Frequencies")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # print(lenstra(2 * 3))
    # plot_elliptic_in_z(59, True)
    # study_speed_lenstra(5000, 100)
    study_loops(50000, 10000)
