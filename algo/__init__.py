__all__ = ['aes', 'blowfish', 'des']

class Algo(object):
    def encrypt(self, text: bytes) -> bytes:
        raise NotImplementedError