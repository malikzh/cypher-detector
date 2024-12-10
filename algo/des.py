from Crypto.Cipher import DES
from . import Algo

class DesAlgo(Algo):
    NAME = 'DES'
    KEY = bytes.fromhex('071c6539a1fdf779') # 8 bytes

    def encrypt(self, text: bytes) -> bytes:
        cipher = DES.new(self.KEY, DES.MODE_ECB)
        return cipher.encrypt(text)