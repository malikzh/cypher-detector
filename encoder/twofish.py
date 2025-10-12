from . import Encoder
from .implementation.twofish import TwoFish_encrypt
import binascii
import os

class TwofishEncoder(Encoder):
    def encrypt(self, text: bytes, key: bytes) -> bytes:
        # Преобразуем text и key из bytes в hex string
        hex_text = binascii.hexlify(text).decode('utf-8')
        hex_key = binascii.hexlify(key).decode('utf-8')
        
        # Шифруем данные в hex формате
        hex_ciphertext = TwoFish_encrypt(hex_text, hex_key, "CBC")
        
        # Преобразуем результат из hex в bytes и возвращаем
        return binascii.unhexlify(hex_ciphertext.encode('utf-8'))
    
    def generate_key(self) -> bytes:
        return os.urandom(32)