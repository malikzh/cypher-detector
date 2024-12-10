from algo import aes, blowfish, des
import math

def bitstring_to_bytes(s):
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder='big')

# Количество шифров на каждый алгоритм
CIPHERS_QUANTITY = 65536

ALGOS = [
    aes.AesAlgo(),
    blowfish.BlowfishAlgo(),
    des.DesAlgo(),
]

BYTES_QUANTITY = 128

PATH = './dataset'

bits = round(math.log2(CIPHERS_QUANTITY))

for algo in ALGOS:
    name = algo.NAME
    print('Generating data for {}'.format(name))

    ciphers = []
    for i in range(CIPHERS_QUANTITY):
        print('Generating cipher {} of {}'.format(i+1, CIPHERS_QUANTITY))
        text = ("{0:0" + str(bits) + "b}").format(i)
        assert len(text) == bits, "Text has {} bytes length, but must have {} bytes".format(len(text), bits)
        bstring = bitstring_to_bytes(text)

        diff = BYTES_QUANTITY - len(bstring)

        if diff > 0:
            bstring += b'\x00' * diff

        cyphertext = algo.encrypt(bstring)
        ciphers.append(cyphertext.hex())

    filepath = PATH + '/' + name + '.txt'
    print('Saving ciphers at: ' + filepath)

    with open(filepath, 'w') as f:
        f.writelines(ciphers)

