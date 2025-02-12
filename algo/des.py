from Crypto.Cipher import DES3
from . import Algo

class DesAlgo(Algo):
    NAME = 'DES'
    KEY = bytes.fromhex('6052bfebee999bbb77a641b2ca73931a') # 128 bit

    def encrypt(self, text: bytes) -> bytes:
        cipher = DES3.new(self.KEY, DES3.MODE_ECB)
        return cipher.encrypt(text)