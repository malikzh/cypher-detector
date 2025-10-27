from . import Encoder
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


class BlowfishEncoder(Encoder):
    def __init__(self):
        self.generate()
        self.backend = default_backend()

    def encrypt(self, text: bytes) -> bytes:
        assert len(self.key) == 32, "Key length must be 32 bytes for Blowfish"

        cipher = Cipher(algorithms.Blowfish(self.key), modes.CBC(self.iv), backend=default_backend())
        encryptor = cipher.encryptor()

        return encryptor.update(text) + encryptor.finalize()
    
    def generate_key(self):
        self.key = os.urandom(32)

    def generate_iv(self):
        self.iv = os.urandom(8)
