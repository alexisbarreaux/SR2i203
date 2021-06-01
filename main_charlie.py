from address_generation import getCompressedPublicKey


def bruteforce(compressedPubKey, min=1, max=2**256-1):
    for privKey in range(min, max):
        if getCompressedPublicKey(privKey.to_bytes(32, "big")) == compressedPubKey:
            return privKey


if __name__ == '__main__':
    # Clé associée à k = 7
    print(bruteforce("19ZewH8Kk1PDbSNdJ97FP4EiCjTRaZMZQA"))
    print(bruteforce("1EhqbyUMvvs7BfL8goY6qcPbD6YKfPqb7e"))
    # Clé associée à k = 8
    print(bruteforce("1CQFwcjw1dwhtkVWBttNLDtqL7ivBonGPV"))
