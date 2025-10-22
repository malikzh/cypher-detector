__all__ = ['aes', 'twofish', 'triple_des', 'kuznechik']


class Encoder(object):
    def encrypt(self, text: bytes) -> bytes:
        raise NotImplementedError
    
    def generate_key(self):
        raise NotImplementedError

    def generate_iv(self):
        raise NotImplementedError

    def generate(self):
        self.generate_key()
        self.generate_iv()
    

import encoder.aes as aes
import encoder.twofish as twofish
import encoder.triple_des as triple_des
import encoder.kuznechik as kuznechik

ENCODER_FACTORY = {
    'AES': lambda: aes.AesEncoder(),
    'Twofish': lambda: twofish.TwofishEncoder(),
    '3DES': lambda: triple_des.TripleDesEncoder(),
    'Kuznechik': lambda: kuznechik.KuznechikEncoder(),
}