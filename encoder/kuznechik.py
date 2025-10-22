from . import Encoder
import os
import gostcrypto


class KuznechikEncoder(Encoder):
    def __init__(self):
        self.generate()

    def encrypt(self, text: bytes) -> bytes:
        cipher_obj = gostcrypto.gostcipher.new('kuznechik',
                                        self.key,
                                        gostcrypto.gostcipher.MODE_CBC,
                                        pad_mode=gostcrypto.gostcipher.PAD_MODE_1, init_vect=self.iv)

        return b"" + cipher_obj.encrypt(text)
    
    def generate_key(self):
        self.key = os.urandom(32)

    def generate_iv(self):
        self.iv = os.urandom(16)
