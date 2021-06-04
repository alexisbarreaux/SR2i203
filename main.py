import numpy as np
import matplotlib.pyplot as plt
import sys
from math import factorial as fact
from time import time
import timeit
import hashlib
from address_generation import getPublicKey as classGetpub
from address_generation import getCompressedPublicKey as classGetCompPup

# https://viresinnumeris.fr/comprendre-bitcoin-cles-et-adresses/
# https://en.wikipedia.org/wiki/Lenstra_elliptic-curve_factorization#Algorithm
# https://stackoverflow.com/questions/31074172/elliptic-curve-point-addition-over-a-finite-field-in-python

sys.setrecursionlimit(10000)
rng = np.random.default_rng()

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
    return rng.integers(n), rng.integers(n)


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
    x_0 = int(x_0)
    y_0 = int(y_0)
    a = int(rng.integers(n))
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
    assert (type(k) in [np.int64, np.int32, int])

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
    assert (type(k) in [np.int64, np.int32, int])

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
            return lenstra(n, quick, use_loop, loop + 1, fact_size, ret_loop)


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


def sha256(data):
    digest = hashlib.new("sha256")
    digest.update(data)
    return digest.digest()


def ripemd160(x):
    d = hashlib.new("ripemd160")
    d.update(x)
    return d.digest()


def b58(data):
    B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

    if data[0] == 0:
        return "1" + b58(data[1:])

    x = sum([v * (256 ** i) for i, v in enumerate(data[::-1])])
    ret = ""
    while x > 0:
        ret = B58[x % 58] + ret
        x = x // 58

    return ret


def getPublicKey(pk,
                 x_gene=0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
                 y_gene=0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
                 a=0,
                 b=7,
                 n=2 ** 256 - 2 ** 32 - 2 ** 9 - 2 ** 8 - 2 ** 7 - 2 ** 6 - 2 ** 4 - 1):
    """
    Function to get the public key from a private key on SPEC256k1
    :param privkey: the private key
    :param x_gene: absciss of the generator we want to use
    :param y_gene: ordinate of the generator we want to use
    :param a: parameter of our curve
    :param b: parameter of our curve
    :param n: parameter of our curve
    :return:
    """
    if type(pk) in [int, np.int32, np.int64]:
        pass
    elif type(pk) == bytes:
        pk = int.from_bytes(pk, "big")
    elif type(pk) == hex:
        pk = int.from_bytes(bytes.fromhex(pk), "big")
    else:
        raise (Exception("Unhandled type for pk"))

    ret, pub_key = quick_elliptic_multiplication(n, a, b, x_gene, y_gene, pk)
    if ret != 1:
        raise Exception("Error computing public key, are you sure you gave a valid curve ?")

    x_pub, y_pub = pub_key

    pub_byte = b"\x04" + x_pub.to_bytes(32, "big") + y_pub.to_bytes(32, "big")

    hash160 = ripemd160(sha256(pub_byte))
    address = b"\x00" + hash160
    address = b58(address + sha256(sha256(address))[:4])
    return address


def getCompressedPublicKey(pk,
                           x_gene=0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
                           y_gene=0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
                           a=0,
                           b=7,
                           n=2 ** 256 - 2 ** 32 - 2 ** 9 - 2 ** 8 - 2 ** 7 - 2 ** 6 - 2 ** 4 - 1):
    """
    Function to get the compressed public key from a private key on SPEC256k1
    :param privkey: the private key
    :param x_gene: absciss of the generator we want to use
    :param y_gene: ordinate of the generator we want to use
    :param a: parameter of our curve
    :param b: parameter of our curve
    :param n: parameter of our curve
    :return:
    """
    if type(pk) in [int, np.int32, np.int64]:
        pass
    elif type(pk) == bytes:
        pk = int.from_bytes(pk, "big")
    elif type(pk) == hex:
        pk = int.from_bytes(bytes.fromhex(pk), "big")
    else:
        raise (Exception("Unhandled type for pk"))

    ret, pub_key = quick_elliptic_multiplication(n, a, b, x_gene, y_gene, pk)
    x_pub, y_pub = pub_key

    pub_compressed = b""
    if y_pub % 2 == 0:
        pub_compressed += b"\x02"
    else:
        pub_compressed += b"\x03"
    pub_compressed += x_pub.to_bytes(32, "big")
    hash160 = ripemd160(sha256(pub_compressed))
    address = b"\x00" + hash160
    address = b58(address + sha256(sha256(address))[:4])
    return address


if __name__ == "__main__":
    """
    k = rng.integers(1, pow(2, 5))
    p = 2 ** 256 - 2 ** 32 - 2 ** 9 - 2 ** 8 - 2 ** 7 - 2 ** 6 - 2 ** 4 - 1
    a = 0
    b = 7
    x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    
    t = time()
    print(quick_elliptic_multiplication(p, a, b, x, y, k))
    print(time() - t)

    """
    k = rng.integers(1, pow(2, 5))
    print(classGetpub(k))
    print(classGetCompPup(k))
    print(getPublicKey(k))
    print(getCompressedPublicKey(k))
