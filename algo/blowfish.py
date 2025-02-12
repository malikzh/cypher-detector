from Crypto.Cipher import Blowfish
from . import Algo

class BlowfishAlgo(Algo):
    NAME = 'Blowfish'
    KEY = bytes.fromhex('8de2a7e46e028887423b5c91b1e8065b') # 128 bit

    def encrypt(self, text: bytes) -> bytes:
        cipher = Blowfish.new(self.KEY, Blowfish.MODE_CBC)

        plen = 8 - len(text) % 8  # Padding
        padding = bytes([plen] * plen)
        text += padding
        return cipher.encrypt(text)