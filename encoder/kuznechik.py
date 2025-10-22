from . import Encoder
import os
import gostcrypto


class KuznechikEncoder(Encoder):
    def __init__(self):
        self.iv = bytes(16)

    def encrypt(self, text: bytes, key: bytes) -> bytes:
        cipher_obj = gostcrypto.gostcipher.new('kuznechik',
                                        key,
                                        gostcrypto.gostcipher.MODE_CBC,
                                        pad_mode=gostcrypto.gostcipher.PAD_MODE_1, init_vect=self.iv)

        return b"" + cipher_obj.encrypt(text)
    
    def generate_key(self) -> bytes:
        return os.urandom(32)