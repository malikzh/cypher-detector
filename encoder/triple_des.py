from . import Encoder
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend


class TripleDesEncoder(Encoder):
    def __init__(self):
        self.generate()
        self.backend = default_backend()

    def encrypt(self, text: bytes) -> bytes:
        assert len(text) % 8 == 0, "Text length must be a multiple of 8 bytes for 3DES"

        """Шифрование 3DES в режиме CBC"""
        # Создание cipher
        cipher = Cipher(algorithms.TripleDES(self.key), modes.CBC(self.iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        # Шифрование
        ciphertext = encryptor.update(text) + encryptor.finalize()
        
        # Возвращаем зашифрованные данные
        return ciphertext
    
    def generate_key(self):
        self.key = os.urandom(24)

    def generate_iv(self):
        self.iv = os.urandom(8)
