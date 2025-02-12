from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from . import Algo

class AesAlgo(Algo):
    NAME = 'AES'
    KEY = bytes.fromhex('cd3a9de1d0003f31a597ba339319698e') # 128 bit

    def encrypt(self, text: bytes) -> bytes:
        cipher = Cipher(algorithms.AES(self.KEY), modes.ECB(), backend=default_backend())
        encryptor = cipher.encryptor()
        return encryptor.update(text) + encryptor.finalize()