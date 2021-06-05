
from time import time
import multiprocessing
from main import getCompressedPublicKey, rng
import matplotlib.pyplot as plt
import numpy as np


def bruteforce(compressedPubKey: str, min: int, max: int, returnDict: dict) -> None:
    for pk in range(min, max+1):
        if getCompressedPublicKey(pk) == compressedPubKey:
            returnDict["pk"] = pk
            return


def bruteforce2(compressedPubKey: str, n_bits: int, n_split: int, i: int, returnDict: dict) -> None:
    for pk in range(i, pow(2, n_bits), n_split):
        if pk == 0:
            continue
        if getCompressedPublicKey(pk) == compressedPubKey:
            returnDict["pk"] = pk
            return


def ranges(N, nb):
    step = N / nb
    return [(round(step*i), round(step*(i+1))) for i in range(nb)]


def get_uniform_ranges(n_bits=256, n_split=4):
    res = ranges(pow(2, n_bits)-1, n_split)
    res[0] = res[0][0]+1, res[0][1]
    return res


def multi_processing_bruteforce(compressedPubKey: str, n_bits=256, n_split=4):
    ranges = get_uniform_ranges(n_bits, n_split)

    processes = []
    manager = multiprocessing.Manager()
    returnDict = manager.dict()

    for r in ranges:
        p = multiprocessing.Process(target=bruteforce, args=[
                                    compressedPubKey, int(r[0]), int(r[1]), returnDict])
        p.start()
        processes.append(p)

    while True:
        if "pk" in returnDict.keys():
            for p2 in processes:
                p2.kill()
            return returnDict["pk"]


def multi_processing_bruteforce2(compressedPubKey: str, n_bits=256, n_split=4):

    processes = []
    manager = multiprocessing.Manager()
    returnDict = manager.dict()

    for i in range(n_split):
        p = multiprocessing.Process(target=bruteforce2, args=[
                                    compressedPubKey, n_bits, n_split, i, returnDict])
        p.start()
        processes.append(p)

    while True:
        if "pk" in returnDict.keys():
            for p2 in processes:
                p2.kill()
            return returnDict["pk"]


def plot_multiprocessing(s_max=20, N=50):
    """
    Function to plot the time efficiency of the very first naive bruteforce.
    """
    size_range = np.arange(2, s_max, dtype=np.int64)
    times = [0] * len(size_range)

    i = 0
    for size in size_range:
        priv_keys = rng.integers(pow(2, size - 1), pow(2, size), N)
        for k in priv_keys:
            addr_pub = getCompressedPublicKey(k)

            t = time()
            pk = multi_processing_bruteforce(addr_pub, size)
            if pk == k:
                times[i] += (time() - t)
            else:
                print(k, pk)
                raise(Exception("Bruteforce failed"))

        times[i] /= N

        i += 1

    plt.clf()
    plt.title(f"Times on {N} random keys between 2^(size -1) and 2^size.")
    plt.grid()
    plt.xlabel("Size of the private key")
    plt.ylabel("Time")
    plt.scatter(size_range, times, marker="x", c="b", label="Times")
    plt.legend()
    plt.show()


def plot_multiprocessing2(s_max=20, N=50):
    """
    Function to plot the time efficiency of the very first naive bruteforce.
    """
    size_range = np.arange(2, s_max, dtype=np.int64)
    times = [0] * len(size_range)

    i = 0
    for size in size_range:
        priv_keys = rng.integers(pow(2, size - 1), pow(2, size), N)
        for k in priv_keys:
            addr_pub = getCompressedPublicKey(k)

            t = time()
            pk = multi_processing_bruteforce2(addr_pub, size)
            if pk == k:
                times[i] += (time() - t)
            else:
                print(k, pk)
                raise(Exception("Bruteforce failed"))

        times[i] /= N

        i += 1

    plt.clf()
    plt.title(f"Times on {N} random keys between 2^(size -1) and 2^size.")
    plt.grid()
    plt.xlabel("Size of the private key")
    plt.ylabel("Time")
    plt.scatter(size_range, times, marker="x", c="b", label="Times")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # print(get_uniform_ranges())
    plot_multiprocessing2(14, 10)
    # multi_processing_bruteforce("1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e", 256, 4)
