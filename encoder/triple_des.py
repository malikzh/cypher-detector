from . import Encoder
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend


class TripleDesEncoder(Encoder):
    def __init__(self):
        self.backend = default_backend()

    def encrypt(self, text: bytes, key: bytes) -> bytes:
        assert len(key) == 24, "Key must be 24 bytes for 3DES"
        assert len(text) % 8 == 0, "Text length must be a multiple of 8 bytes for 3DES"

        """Шифрование 3DES в режиме CBC"""
        # Генерация IV (8 байт для 3DES)
        iv = bytes(8)
        
        # Создание cipher
        cipher = Cipher(algorithms.TripleDES(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        # Шифрование
        ciphertext = encryptor.update(text) + encryptor.finalize()
        
        # Возвращаем зашифрованные данные
        return ciphertext
    
    def generate_key(self) -> bytes:
        return os.urandom(24)