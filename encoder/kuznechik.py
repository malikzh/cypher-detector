from . import Encoder
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.backends import default_backend
import gostcrypto

class KuznechikEncoder(Encoder):
    def encrypt(self, text: bytes, key: bytes) -> bytes:
        cipher_obj = gostcrypto.gostcipher.new('kuznechik',
                                        key,
                                        gostcrypto.gostcipher.MODE_CBC,
                                        pad_mode=gostcrypto.gostcipher.PAD_MODE_1)

        return b"" + cipher_obj.encrypt(text)
    
    def generate_key(self) -> bytes:
        return os.urandom(32)