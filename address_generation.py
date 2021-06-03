# https://viresinnumeris.fr/comprendre-bitcoin-cles-et-adresses/
import os
import timeit
from time import time
import hashlib
import numpy as np


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


class Point:
    def __init__(self,
                 x=0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
                 y=0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
                 p=2 ** 256 - 2 ** 32 - 2 ** 9 - 2 ** 8 - 2 ** 7 - 2 ** 6 - 2 ** 4 - 1):
        self.x = x
        self.y = y
        self.p = p

    def __add__(self, other):
        return self.__radd__(other)

    def __mul__(self, other):
        return self.__rmul__(other)

    def __rmul__(self, other):
        n = self
        q = None

        for i in range(256):
            if other & (1 << i):
                q = q + n
            n = n + n

        return q

    def __radd__(self, other):
        if other is None:
            return self
        x1 = other.x
        y1 = other.y
        x2 = self.x
        y2 = self.y
        p = self.p

        if self == other:
            l = pow(2 * y2 % p, p - 2, p) * (3 * x2 * x2) % p
        else:
            l = pow(x1 - x2, p - 2, p) * (y1 - y2) % p

        newX = (l ** 2 - x2 - x1) % p
        newY = (l * x2 - l * newX - y2) % p

        return Point(newX, newY)

    def toBytes(self):
        x = self.x.to_bytes(32, "big")
        y = self.y.to_bytes(32, "big")
        return b"\x04" + x + y


def getPublicKey(pk):
    if type(pk) in [int, np.int32, np.int64]:
        pass
    elif type(pk) == bytes:
        pk = int.from_bytes(pk, "big")
    elif type(pk) == hex:
        pk = int.from_bytes(bytes.fromhex(pk), "big")
    else:
        raise (Exception("Unhandled type for pk"))

    SPEC256k1 = Point()
    hash160 = ripemd160(sha256((SPEC256k1 * pk).toBytes()))
    address = b"\x00" + hash160

    address = b58(address + sha256(sha256(address))[:4])
    return address


def getCompressedPublicKey(pk):
    if type(pk) in [int, np.int32, np.int64]:
        pass
    elif type(pk) == bytes:
        pk = int.from_bytes(pk, "big")
    elif type(pk) == hex:
        pk = int.from_bytes(bytes.fromhex(pk), "big")
    else:
        raise (Exception("Unhandled type for pk"))

    SPEC256k1 = Point()
    pub = SPEC256k1 * pk
    pub_compressed = b""
    if pub.y % 2 == 0:
        pub_compressed += b"\x02"
    else:
        pub_compressed += b"\x03"
    pub_compressed += pub.x.to_bytes(32, "big")
    hash160 = ripemd160(sha256(pub_compressed))
    address = b"\x00" + hash160

    address = b58(address + sha256(sha256(address))[:4])
    return address


def getWif(privkey):
    wif = b"\x80" + privkey
    wif = b58(wif + sha256(sha256(wif))[:4])
    return wif


def time_test():
    setup = "import numpy as np; " \
            "from address_generation import getCompressedPublicKey;" \
            " k = np.random.randint(1, pow(2, 5))"
    return print(timeit.timeit(stmt='getCompressedPublicKey(k)', setup=setup, number=5))


def profile_test():
    for _ in range(10):
        k = np.random.randint(1, pow(2, 5))
        getCompressedPublicKey(k)
    return


if __name__ == "__main__":
    """
    # k = bytes.fromhex("6ef6b8ddb7d09b14a3f5239b1d76ed943bc697765ffd242baf08e532cdbe6197")
    randomBytes = os.urandom(32)
    randomBytes = bytes.fromhex(
        "0000000000000000000000000000000000000000000000000000000000000007")
    print("Address: " + getPublicKey(randomBytes))
    print("Compressed address: " + getCompressedPublicKey(randomBytes))
    # print("Compressed address: " + getCompressedPublicKey(randomBytes))
    print("Privkey: " + getWif(randomBytes))
    """

    """
    k = pow(2, 31) # random.randint(1, pow(2, 31))
    print(k)

    SPEC256k1 = Point()
    t = time()
    pub = SPEC256k1 * k
    print(pub.x, pub.y)
    print("Time foudn", time() - t)


    p = 2 ** 256 - 2 ** 32 - 2 ** 9 - 2 ** 8 - 2 ** 7 - 2 ** 6 - 2 ** 4 - 1
    a = 0
    b = 7
    x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
    y = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
    t = time()
    print(quick_elliptic_multiplication(p, a, b, x, y, k))
    print("Time quick", time() - t)
    """
    # python -m cProfile address_generation.py