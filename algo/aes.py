from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from . import Algo

class AesAlgo(Algo):
    NAME = 'AES'
    KEY = bytes.fromhex('cd3a9de1d0003f31a597ba339319698e782d117a8cb4748e03bbe790f79d6b54') # 256 bit
    IV  = bytes.fromhex('8de2a7e46e028887423b5c91b1e8065b') # 128 bit

    def encrypt(self, text: bytes) -> bytes:
        cipher = Cipher(algorithms.AES(self.KEY), modes.CFB(self.IV), backend=default_backend())
        encryptor = cipher.encryptor()
        return encryptor.update(text) + encryptor.finalize()