__all__ = ['aes', 'twofish', 'triple_des', 'kuznechik']

class Encoder(object):
    def encrypt(self, text: bytes, key: bytes) -> bytes:
        raise NotImplementedError