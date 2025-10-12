from encoder.twofish import TwofishEncoder


enc = TwofishEncoder()
key = b'0123456789abcdef'  # 16 байт для AES-128
text = b'Hello, World! This is a test message.'
ciphertext = enc.encrypt(text, key)
print("Ciphertext:", ciphertext)