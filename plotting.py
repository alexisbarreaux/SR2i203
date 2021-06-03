from main import *

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


def study_speed_comp_addr(n: int):
    N = 60
    sizes = np.arange(1, N, dtype=np.int64)
    times = np.zeros(N - 1, dtype=np.float)
    for size in sizes:
        priv_keys = rng.integers(pow(2, size - 1), pow(2, size), n)
        for pk in priv_keys:
            t = time()
            getCompressedPublicKey(pk)
            times[size - 1] += time() - t

    times /= n

    plt.clf()
    plt.title(f"Mean time execution of getCompressedPublicKey \n on {n} random keys between 2^(size - 1) and 2^(size)")
    plt.grid()
    plt.xlabel("Size : taille en bit de la clé privée")
    plt.ylabel("Time")
    plt.scatter(sizes, times, marker="x", c="b")
    #plt.legend()
    plt.show()
    return


if __name__ == "__main__":
    # print(lenstra(2 * 3))
    # plot_elliptic_in_z(59, True)
    # study_speed_lenstra(5000, 100)
    # study_loops(50000, 10000)
    study_speed_comp_addr(100)