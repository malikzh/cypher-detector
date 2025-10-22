from . import Encoder
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend


class AesEncoder(Encoder):
    def __init__(self):
        self.generate()
        self.backend = default_backend()

    def encrypt(self, text: bytes) -> bytes:
        """Шифрование AES в режиме CBC"""
        assert len(text) % 16 == 0, "Text length must be a multiple of 16 bytes for AES"

        # Создание cipher
        cipher = Cipher(algorithms.AES(self.key), modes.CBC(self.iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        # Шифрование
        ciphertext = encryptor.update(text) + encryptor.finalize()
        
        return ciphertext
    
    def generate_key(self):
        self.key = os.urandom(32)

    def generate_iv(self):
        self.iv = os.urandom(16)