__all__ = ['aes', 'blowfish.py', 'triple_des', 'kuznechik']


class Encoder(object):
    def encrypt(self, text: bytes) -> bytes:
        raise NotImplementedError
    
    def generate_key(self, key: bytes):
        raise NotImplementedError

    def generate_iv(self, iv: bytes):
        raise NotImplementedError

    def generate(self, key: bytes, iv: bytes):
        self.generate_key(key)
        self.generate_iv(iv)
    

import encoder.aes as aes
import encoder.blowfish as twofish
import encoder.triple_des as triple_des
import encoder.kuznechik as kuznechik

ENCODER_FACTORY = {
    'AES': lambda: aes.AesEncoder(),
    'Blowfish': lambda: twofish.BlowfishEncoder(),
    '3DES': lambda: triple_des.TripleDesEncoder(),
    'Kuznyechik': lambda: kuznechik.KuznechikEncoder(),
}