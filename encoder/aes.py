from . import Encoder
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

class AesEncoder(Encoder):
    def __init__(self):
        self.backend = default_backend()

    def encrypt(self, text: bytes, key: bytes) -> bytes:
        """Шифрование AES в режиме CBC"""
        # Генерация IV (16 байт для AES)
        iv = os.urandom(16)
        
        # Создание cipher
        cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        # Padding
        padder = padding.PKCS7(128).padder()
        padded_data = padder.update(text) + padder.finalize()
        
        # Шифрование
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        return ciphertext
    
    def generate_key(self) -> bytes:
        return os.urandom(32)