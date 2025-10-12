from . import Encoder
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend

class TripleDesEncoder(Encoder):
    def __init__(self):
        self.backend = default_backend()

    def encrypt(self, text: bytes, key: bytes) -> bytes:
        """Шифрование 3DES в режиме CBC"""
        # Генерация IV (8 байт для 3DES)
        iv = os.urandom(8)
        
        # Создание cipher
        cipher = Cipher(algorithms.TripleDES(key), modes.CBC(iv), backend=self.backend)
        encryptor = cipher.encryptor()
        
        # Padding
        padder = padding.PKCS7(64).padder()
        padded_data = padder.update(text) + padder.finalize()
        
        # Шифрование
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()
        
        # Возвращаем IV + зашифрованные данные
        return ciphertext
    
    def generate_key(self) -> bytes:
        return os.urandom(24)