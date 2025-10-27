from . import Encoder
import os
import gostcrypto


class KuznechikEncoder(Encoder):
    def encrypt(self, text: bytes) -> bytes:
        cipher_obj = gostcrypto.gostcipher.new('kuznechik',
                                        self.key,
                                        gostcrypto.gostcipher.MODE_CBC,
                                        pad_mode=gostcrypto.gostcipher.PAD_MODE_1, init_vect=self.iv)

        return b"" + cipher_obj.encrypt(text)

    def generate_key(self, key: bytes):
        self.key = key[:32]

    def generate_iv(self, iv: bytes):
        self.iv = iv[:16]
