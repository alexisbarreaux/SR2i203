from main import getCompressedPublicKey, rng
import matplotlib.pyplot as plt
import numpy as np
from time import time


def bit_to_range(n: int):
    """
    Renvoie la range des valeurs possibles connaissant le nombre de bits
    :param n: Le nombre de bits
    :return:
    """
    return pow(2, n) - 1


def naive_bruteforce(compressedPubKey, n_bits=256):
    assert (type(n_bits) in [int, np.int32, np.int64] and 0 <= n_bits <= 256)
    max_priv = bit_to_range(n_bits)
    for privKey in range(1, max_priv + 1):
        if getCompressedPublicKey(privKey.to_bytes(32, "big")) == compressedPubKey:
            return privKey

    return -1


def plot_naive(s_max=20, N=50):
    """
    Function to plot the time efficiency of the very first naive bruteforce.
    """
    size_range = np.arange(1, s_max, dtype=np.int64)
    times = np.zeros(len(size_range))
    for size in size_range:
        priv_keys = rng.integers(pow(2, size - 1), pow(2, size), N)
        for k in priv_keys:
            addr_pub = getCompressedPublicKey(k)

            t = time()
            pk = naive_bruteforce(addr_pub, size)
            if pk == k:
                times[size - 1] += time() - t
            else:
                print(k, pk)
                raise(Exception("Bruteforce failed"))

        times[size - 1] /= N


    plt.clf()
    plt.title(f"Times on {N} random keys between 2^(size -1) and 2^size.")
    plt.grid()
    plt.xlabel("Size of the private key")
    plt.ylabel("Time")
    plt.scatter(size_range, times, marker="x", c="b", label="Times")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_naive(14, 10)
    """
    # Clé associée à k = 7
    print(naive_bruteforce("19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA"))
    print(naive_bruteforce("1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e"))
    # Clé associée à k = 8
    # print(bruteforce("1CQFwcjw1dwhtkVWBttNLDtqL7ivBonGPV"))
    """